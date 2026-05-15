from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from poet_mini.data.token_dataset import TokenBatcher, TokenDataConfig
from poet_mini.model.gpt import GPT, GPTConfig


# ---- 路径与配置 ----
def project_root() -> Path:
    """返回 train 工程根目录。"""
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str | Path) -> Path:
    """将配置路径解析到 train 工程根目录。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层结构必须是对象：{path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层结构必须是对象：{path}")
    return data


# ---- tokenizer 编解码 ----
def tokenizer_maps(tokenizer_data: dict[str, Any]) -> tuple[dict[str, int], list[str], list[str]]:
    """读取 tokenizer 映射。"""
    token_to_id = tokenizer_data.get("token_to_id", {})
    id_to_token = tokenizer_data.get("id_to_token", [])
    special_tokens = tokenizer_data.get("special_tokens", [])
    if not isinstance(token_to_id, dict) or not isinstance(id_to_token, list) or not isinstance(special_tokens, list):
        raise ValueError("tokenizer.json 缺少 token_to_id、id_to_token 或 special_tokens")
    return {str(key): int(value) for key, value in token_to_id.items()}, [str(item) for item in id_to_token], [str(item) for item in special_tokens]


def encode_prompt(prompt: str, token_to_id: dict[str, int], special_tokens: list[str]) -> list[int]:
    """编码 prompt，优先识别特殊 token 字符串。"""
    ids: list[int] = []
    unk_id = token_to_id.get("<UNK>", 3)
    ordered_special_tokens = sorted(special_tokens, key=len, reverse=True)
    index = 0
    while index < len(prompt):
        matched = False
        for token in ordered_special_tokens:
            if prompt.startswith(token, index):
                ids.append(token_to_id[token])
                index += len(token)
                matched = True
                break
        if matched:
            continue
        char = prompt[index]
        ids.append(token_to_id.get(char, unk_id))
        index += 1
    return ids


def decode_ids(ids: list[int], id_to_token: list[str]) -> str:
    """将 token id 解码为文本。"""
    tokens: list[str] = []
    for token_id in ids:
        if 0 <= token_id < len(id_to_token):
            tokens.append(id_to_token[token_id])
        else:
            tokens.append("<UNK>")
    return "".join(tokens)


# ---- 构建对象 ----
def resolve_device(value: str) -> torch.device:
    """解析训练设备。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_model_config(model_config: dict[str, Any], tokenizer_data: dict[str, Any]) -> GPTConfig:
    """构建 GPT 配置。"""
    model_section = model_config.get("model", {})
    if not isinstance(model_section, dict):
        raise ValueError("model.yaml 缺少 model 对象")
    vocab_size = int(model_section.get("vocab_size", tokenizer_data.get("vocab_size", 0)))
    tokenizer_vocab_size = int(tokenizer_data.get("vocab_size", vocab_size))
    if vocab_size != tokenizer_vocab_size:
        raise ValueError(f"model.vocab_size={vocab_size} 与 tokenizer vocab_size={tokenizer_vocab_size} 不一致")
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=int(model_section.get("block_size", 256)),
        n_layer=int(model_section.get("n_layer", 6)),
        n_head=int(model_section.get("n_head", 6)),
        n_embd=int(model_section.get("n_embd", 384)),
        dropout=float(model_section.get("dropout", 0.1)),
        bias=bool(model_section.get("bias", False)),
    )


def build_batcher(root: Path, train_config: dict[str, Any], model_config: GPTConfig, device: torch.device) -> TokenBatcher:
    """构建 token batch 采样器。"""
    data_section = train_config.get("data", {})
    train_section = train_config.get("train", {})
    if not isinstance(data_section, dict) or not isinstance(train_section, dict):
        raise ValueError("train.yaml 缺少 data 或 train 对象")
    config = TokenDataConfig(
        train_bin=resolve_path(root, data_section.get("train_bin", "data/processed/train.bin")),
        val_bin=resolve_path(root, data_section.get("val_bin", "data/processed/val.bin")),
        dtype=str(data_section.get("dtype", "uint16")),
        block_size=model_config.block_size,
        batch_size=int(train_section.get("batch_size", 32)),
        device=device,
    )
    return TokenBatcher(config)


# ---- 训练辅助 ----
def write_metric(path: Path, record: dict[str, Any]) -> None:
    """追加写入一条 metrics JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False))
        file.write("\n")


@torch.no_grad()
def estimate_loss(model: GPT, batcher: TokenBatcher, eval_iters: int) -> dict[str, float]:
    """估算 train / val loss。"""
    model.eval()
    result: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = batcher.get_batch(split)
            _, loss = model(x, y)
            if loss is not None:
                losses.append(float(loss.item()))
        result[split] = sum(losses) / max(len(losses), 1)
    model.train()
    return result


@torch.no_grad()
def write_sample(model: GPT, tokenizer_data: dict[str, Any], prompt: str, output_path: Path, device: torch.device, max_new_tokens: int) -> None:
    """生成并写入一个文本样例。"""
    token_to_id, id_to_token, special_tokens = tokenizer_maps(tokenizer_data)
    prompt_ids = encode_prompt(prompt, token_to_id, special_tokens)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.9, top_k=50)[0].tolist()
    text = decode_ids(output_ids, id_to_token)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, model_config: GPTConfig, train_config: dict[str, Any], step: int, best_val_loss: float, tokenizer_path: Path) -> None:
    """保存训练 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": vars(model_config),
        "train_config": train_config,
        "step": step,
        "best_val_loss": best_val_loss,
        "tokenizer_path": tokenizer_path.as_posix(),
    }
    torch.save(checkpoint, path)


# ---- CLI ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Poet-mini 最小 GPT 训练入口")
    parser.add_argument("--model-config", default="configs/model.yaml", help="模型配置路径")
    parser.add_argument("--train-config", default="configs/train.yaml", help="训练配置路径")
    parser.add_argument("--run-id", default=None, help="运行 ID，默认读取 POET_MINI_RUN_ID 或 local-dev")
    parser.add_argument("--resume", default=None, help="从指定 checkpoint 恢复训练")
    return parser.parse_args()


def main() -> int:
    """训练主入口。"""
    root = project_root()
    args = parse_args()
    run_id = args.run_id or os.environ.get("POET_MINI_RUN_ID", "local-dev")

    model_config_path = resolve_path(root, args.model_config)
    train_config_path = resolve_path(root, args.train_config)
    raw_model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)

    data_section = train_config.get("data", {})
    train_section = train_config.get("train", {})
    runtime_section = train_config.get("runtime", {})
    output_section = train_config.get("output", {})
    tokenizer_path = resolve_path(root, data_section.get("tokenizer_json", "data/processed/tokenizer.json"))
    tokenizer_data = load_json(tokenizer_path)

    seed = int(train_section.get("seed", 20260513))
    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(str(train_section.get("device", "auto")))
    model_config = build_model_config(raw_model_config, tokenizer_data)
    batcher = build_batcher(root, train_config, model_config, device)

    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_section.get("learning_rate", 3e-4)),
        betas=(float(train_section.get("beta1", 0.9)), float(train_section.get("beta2", 0.95))),
        weight_decay=float(train_section.get("weight_decay", 0.1)),
    )

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(resolve_path(root, args.resume), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    if bool(train_section.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    checkpoint_dir = resolve_path(root, output_section.get("checkpoint_dir", "outputs/checkpoints"))
    sample_dir = resolve_path(root, output_section.get("sample_dir", "outputs/samples"))
    log_dir = resolve_path(root, output_section.get("log_dir", "outputs/logs"))
    metrics_path = log_dir / "metrics.jsonl"
    max_steps = int(train_section.get("max_steps", 1000))
    eval_interval = int(runtime_section.get("eval_interval", 100))
    eval_iters = int(runtime_section.get("eval_iters", 20))
    log_interval = int(runtime_section.get("log_interval", 10))
    save_interval = int(runtime_section.get("save_interval", 500))
    sample_interval = int(runtime_section.get("sample_interval", 200))
    sample_max_new_tokens = int(runtime_section.get("sample_max_new_tokens", 120))
    sample_prompt = str(runtime_section.get("sample_prompt", "<BOS>标题：春夜\n作者：李白\n"))
    grad_clip = float(train_section.get("grad_clip", 1.0))

    print(f"[train] run_id={run_id} device={device} 参数量={model.num_parameters()} train_tokens={batcher.token_count('train')} val_tokens={batcher.token_count('val')}", flush=True)
    model.train()
    last_time = time.time()
    for step in range(start_step + 1, max_steps + 1):
        x, y = batcher.get_batch("train")
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("训练 loss 为空")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            now = time.time()
            elapsed = max(now - last_time, 1e-6)
            tokens_per_second = batcher.config.batch_size * model_config.block_size * log_interval / elapsed
            last_time = now
            metric = {
                "timestamp": now,
                "run_id": run_id,
                "step": step,
                "split": "train",
                "loss": float(loss.item()),
                "lr": float(train_section.get("learning_rate", 3e-4)),
                "tokens_per_second": round(tokens_per_second, 2),
                "device": str(device),
            }
            write_metric(metrics_path, metric)
            print(f"[train] step={step} loss={metric['loss']:.4f} tokens/s={metric['tokens_per_second']}", flush=True)

        if step % eval_interval == 0:
            losses = estimate_loss(model, batcher, eval_iters)
            for split, split_loss in losses.items():
                write_metric(metrics_path, {"timestamp": time.time(), "run_id": run_id, "step": step, "split": split, "loss": split_loss, "device": str(device)})
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, model_config, train_config, step, best_val_loss, tokenizer_path)

        if step % save_interval == 0:
            save_checkpoint(checkpoint_dir / f"step_{step:06d}.pt", model, optimizer, model_config, train_config, step, best_val_loss, tokenizer_path)
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, model_config, train_config, step, best_val_loss, tokenizer_path)

        if step % sample_interval == 0:
            write_sample(model, tokenizer_data, sample_prompt, sample_dir / f"step_{step:06d}.txt", device, sample_max_new_tokens)

    save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, model_config, train_config, max_steps, best_val_loss, tokenizer_path)
    print("[train] 最小训练闭环完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
