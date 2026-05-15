from __future__ import annotations

import argparse
import json
import random
from array import array
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "tokenizer": {
        "add_bos": True,
        "add_eos": True,
    },
    "input": {
        "poems_jsonl": "data/interim/poems.jsonl",
    },
    "output": {
        "tokenizer_json": "data/processed/tokenizer.json",
        "train_bin": "data/processed/train.bin",
        "val_bin": "data/processed/val.bin",
        "stats_json": "outputs/logs/tokenizer_stats.json",
    },
    "split": {
        "val_ratio": 0.02,
        "seed": 20260513,
        "shuffle_samples": True,
    },
    "check": {
        "sample_decode_count": 5,
        "progress_interval": 10000,
    },
}


# ---- 配置加载 ----
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并默认配置和文件配置。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def project_root() -> Path:
    """返回 train 工程根目录。"""
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    """将配置中的相对路径解析到 train 工程根目录。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    if not config_path.exists():
        return DEFAULT_CONFIG
    if yaml is None:
        raise RuntimeError("缺少 PyYAML 依赖，请在 Docker 镜像中安装 pyyaml 后运行 token 序列化流水线")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("配置文件顶层结构必须是 YAML 对象")
    return deep_merge(DEFAULT_CONFIG, loaded)


# ---- JSON 读取 ----
def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层结构必须是对象：{path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写入 UTF-8 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def iter_jsonl_text(path: Path) -> Iterator[tuple[int, str, str]]:
    """逐行读取 JSONL 中的 text 字段。"""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, "", f"JSON 解析失败：{exc}"
                continue
            if not isinstance(record, dict):
                yield line_number, "", "JSONL 行不是对象"
                continue
            text = record.get("text", "")
            if not isinstance(text, str) or not text.strip():
                yield line_number, "", "text 字段为空或不是字符串"
                continue
            yield line_number, text, ""


# ---- token 编解码 ----
def tokenizer_maps(tokenizer_data: dict[str, Any]) -> tuple[dict[str, int], list[str], int, int, int]:
    """读取 tokenizer 映射和特殊 token ID。"""
    token_to_id = tokenizer_data.get("token_to_id", {})
    id_to_token = tokenizer_data.get("id_to_token", [])
    special_token_ids = tokenizer_data.get("special_token_ids", {})
    if not isinstance(token_to_id, dict) or not isinstance(id_to_token, list) or not isinstance(special_token_ids, dict):
        raise ValueError("tokenizer.json 缺少 token_to_id、id_to_token 或 special_token_ids")

    unk_id = int(special_token_ids.get("<UNK>", 3))
    bos_id = int(special_token_ids.get("<BOS>", 1))
    eos_id = int(special_token_ids.get("<EOS>", 2))
    return {str(key): int(value) for key, value in token_to_id.items()}, [str(item) for item in id_to_token], unk_id, bos_id, eos_id


def encode_text(text: str, token_to_id: dict[str, int], unk_id: int, bos_id: int, eos_id: int, add_bos: bool, add_eos: bool) -> tuple[list[int], int]:
    """将文本编码为 token id 序列。"""
    ids: list[int] = []
    if add_bos:
        ids.append(bos_id)

    unk_count = 0
    for char in text:
        token_id = token_to_id.get(char, unk_id)
        if token_id == unk_id and char not in token_to_id:
            unk_count += 1
        ids.append(token_id)

    if add_eos:
        ids.append(eos_id)
    return ids, unk_count


def decode_ids(ids: list[int], id_to_token: list[str]) -> str:
    """将 token id 序列解码为可读文本。"""
    tokens: list[str] = []
    for token_id in ids:
        if 0 <= token_id < len(id_to_token):
            tokens.append(id_to_token[token_id])
        else:
            tokens.append("<UNK>")
    return "".join(tokens)


# ---- 数据切分 ----
def count_samples(path: Path) -> tuple[int, int]:
    """统计可编码样本数和异常行数。"""
    sample_count = 0
    invalid_count = 0
    for _, text, error in iter_jsonl_text(path):
        if error:
            invalid_count += 1
            continue
        if text:
            sample_count += 1
    return sample_count, invalid_count


def build_val_indices(sample_count: int, val_ratio: float, seed: int, shuffle_samples: bool) -> set[int]:
    """构建验证集样本序号集合。"""
    val_count = int(sample_count * val_ratio)
    if sample_count > 1 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if val_count <= 0:
        return set()

    indices = list(range(sample_count))
    if shuffle_samples:
        random.Random(seed).shuffle(indices)
    return set(indices[:val_count])


def array_type_for_vocab(vocab_size: int) -> tuple[str, str]:
    """根据词表大小选择 token 二进制存储类型。"""
    if vocab_size <= 65535:
        return "H", "uint16"
    return "I", "uint32"


def write_token_chunk(file: Any, token_ids: list[int], array_type: str) -> None:
    """将单个样本 token id 写入二进制文件。"""
    data = array(array_type, token_ids)
    data.tofile(file)


# ---- token 序列化 ----
def prepare_tokenized_data(config: dict[str, Any], input_path: Path, tokenizer_path: Path, train_path: Path, val_path: Path, stats_path: Path) -> dict[str, Any]:
    """将 JSONL 样本编码并写入 train.bin / val.bin。"""
    tokenizer_data = load_json(tokenizer_path)
    token_to_id, id_to_token, unk_id, bos_id, eos_id = tokenizer_maps(tokenizer_data)
    vocab_size = int(tokenizer_data.get("vocab_size", len(id_to_token)))
    array_type, dtype = array_type_for_vocab(vocab_size)

    tokenizer_config = config.get("tokenizer", {})
    split_config = config.get("split", {})
    check_config = config.get("check", {})
    add_bos = bool(tokenizer_config.get("add_bos", True))
    add_eos = bool(tokenizer_config.get("add_eos", True))
    val_ratio = float(split_config.get("val_ratio", 0.02))
    seed = int(split_config.get("seed", 20260513))
    shuffle_samples = bool(split_config.get("shuffle_samples", True))
    sample_decode_count = int(check_config.get("sample_decode_count", 5))
    progress_interval = int(check_config.get("progress_interval", 10000))

    if not 0 <= val_ratio < 1:
        raise ValueError("split.val_ratio 必须满足 0 <= val_ratio < 1")

    sample_count, invalid_line_count = count_samples(input_path)
    val_indices = build_val_indices(sample_count, val_ratio, seed, shuffle_samples)
    print(f"[tokenized] 可编码样本数：{sample_count}，验证样本数：{len(val_indices)}，dtype={dtype}", flush=True)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "input": input_path.as_posix(),
        "tokenizer": tokenizer_path.as_posix(),
        "train_bin": train_path.as_posix(),
        "val_bin": val_path.as_posix(),
        "vocab_size": vocab_size,
        "dtype": dtype,
        "array_type": array_type,
        "sample_count": sample_count,
        "train_sample_count": 0,
        "val_sample_count": 0,
        "invalid_line_count": invalid_line_count,
        "train_token_count": 0,
        "val_token_count": 0,
        "unk_token_count": 0,
        "max_sample_tokens": 0,
        "average_sample_tokens": 0,
        "sample_decodes": [],
    }

    encoded_sample_index = 0
    total_sample_tokens = 0
    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        for line_number, text, error in iter_jsonl_text(input_path):
            if error:
                continue

            token_ids, unk_count = encode_text(text, token_to_id, unk_id, bos_id, eos_id, add_bos, add_eos)
            target_is_val = encoded_sample_index in val_indices
            target_file = val_file if target_is_val else train_file
            write_token_chunk(target_file, token_ids, array_type)

            token_count = len(token_ids)
            total_sample_tokens += token_count
            stats["unk_token_count"] += unk_count
            stats["max_sample_tokens"] = max(int(stats["max_sample_tokens"]), token_count)
            if target_is_val:
                stats["val_sample_count"] += 1
                stats["val_token_count"] += token_count
            else:
                stats["train_sample_count"] += 1
                stats["train_token_count"] += token_count

            if len(stats["sample_decodes"]) < sample_decode_count:
                preview_ids = token_ids[: min(200, len(token_ids))]
                stats["sample_decodes"].append(
                    {
                        "line": line_number,
                        "token_count": token_count,
                        "decoded_preview": decode_ids(preview_ids, id_to_token),
                    }
                )

            encoded_sample_index += 1
            if progress_interval > 0 and encoded_sample_index % progress_interval == 0:
                print(f"[tokenized] 已编码样本数：{encoded_sample_index}/{sample_count}", flush=True)

    stats["average_sample_tokens"] = round(total_sample_tokens / sample_count, 2) if sample_count else 0
    stats["train_file_size_bytes"] = train_path.stat().st_size if train_path.exists() else 0
    stats["val_file_size_bytes"] = val_path.stat().st_size if val_path.exists() else 0
    return stats


# ---- CLI 入口 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="将统一 JSONL 编码为 train.bin / val.bin")
    parser.add_argument("--config", default="configs/tokenizer.yaml", help="tokenizer 配置文件路径，默认相对 train 根目录")
    parser.add_argument("--input", default=None, help="覆盖配置中的 poems.jsonl 输入路径")
    parser.add_argument("--tokenizer", default=None, help="覆盖配置中的 tokenizer.json 路径")
    parser.add_argument("--train-output", default=None, help="覆盖配置中的 train.bin 输出路径")
    parser.add_argument("--val-output", default=None, help="覆盖配置中的 val.bin 输出路径")
    parser.add_argument("--stats-output", default=None, help="覆盖配置中的统计报告输出路径")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    input_config = config.get("input", {})
    output_config = config.get("output", {})
    input_path = resolve_path(root, args.input) if args.input else resolve_path(root, input_config.get("poems_jsonl", "data/interim/poems.jsonl"))
    tokenizer_path = resolve_path(root, args.tokenizer) if args.tokenizer else resolve_path(root, output_config.get("tokenizer_json", "data/processed/tokenizer.json"))
    train_path = resolve_path(root, args.train_output) if args.train_output else resolve_path(root, output_config.get("train_bin", "data/processed/train.bin"))
    val_path = resolve_path(root, args.val_output) if args.val_output else resolve_path(root, output_config.get("val_bin", "data/processed/val.bin"))
    stats_path = resolve_path(root, args.stats_output) if args.stats_output else resolve_path(root, output_config.get("stats_json", "outputs/logs/tokenizer_stats.json"))

    if not input_path.exists():
        raise FileNotFoundError(f"统一 JSONL 不存在：{input_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json 不存在：{tokenizer_path}")

    tokenized_stats = prepare_tokenized_data(config, input_path, tokenizer_path, train_path, val_path, stats_path)
    stats = load_json(stats_path) if stats_path.exists() else {}
    stats["tokenized_data"] = tokenized_stats
    write_json(stats_path, stats)

    print(json.dumps({"tokenized_data": tokenized_stats}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
