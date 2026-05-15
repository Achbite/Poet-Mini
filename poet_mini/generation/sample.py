from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from poet_mini.model.gpt import GPT, GPTConfig


# ---- 路径与 JSON ----
def project_root() -> Path:
    """返回 train 工程根目录。"""
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str | Path) -> Path:
    """解析相对 train 根目录的路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


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
    """编码 prompt，优先识别特殊 token。"""
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
    """解码 token id。"""
    return "".join(id_to_token[token_id] if 0 <= token_id < len(id_to_token) else "<UNK>" for token_id in ids)


# ---- CLI ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Poet-mini checkpoint 采样入口")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/latest.pt", help="checkpoint 路径")
    parser.add_argument("--tokenizer", default="data/processed/tokenizer.json", help="tokenizer.json 路径")
    parser.add_argument("--prompt", default="<BOS>标题：春夜\n作者：李白\n", help="生成 prompt")
    parser.add_argument("--output", default="outputs/samples/manual_sample.txt", help="采样输出路径")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="最大新增 token 数")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--top-k", type=int, default=50, help="top-k 采样数量，<=0 表示关闭")
    return parser.parse_args()


def main() -> int:
    """采样主入口。"""
    root = project_root()
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint_path = resolve_path(root, args.checkpoint)
    tokenizer_path = resolve_path(root, args.tokenizer)
    output_path = resolve_path(root, args.output)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_data = load_json(tokenizer_path)
    token_to_id, id_to_token, special_tokens = tokenizer_maps(tokenizer_data)
    prompt_ids = encode_prompt(args.prompt, token_to_id, special_tokens)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k if args.top_k > 0 else None)[0].tolist()
    text = decode_ids(output_ids, id_to_token)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
