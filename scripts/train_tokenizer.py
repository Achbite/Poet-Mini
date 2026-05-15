from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "tokenizer": {
        "type": "char",
        "min_frequency": 1,
        "add_bos": True,
        "add_eos": True,
        "special_tokens": ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>"],
    },
    "input": {
        "corpus_text": "data/interim/poetry_corpus.txt",
    },
    "output": {
        "tokenizer_json": "data/processed/tokenizer.json",
        "stats_json": "outputs/logs/tokenizer_stats.json",
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
        raise RuntimeError("缺少 PyYAML 依赖，请在 Docker 镜像中安装 pyyaml 后运行 tokenizer 流水线")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("配置文件顶层结构必须是 YAML 对象")
    return deep_merge(DEFAULT_CONFIG, loaded)


# ---- 词表构建 ----
def read_special_tokens(config: dict[str, Any]) -> list[str]:
    """读取并校验特殊 token 顺序。"""
    tokenizer_config = config.get("tokenizer", {})
    raw_tokens = tokenizer_config.get("special_tokens", DEFAULT_CONFIG["tokenizer"]["special_tokens"])
    if not isinstance(raw_tokens, list):
        raise ValueError("tokenizer.special_tokens 必须是 YAML 序列")

    tokens: list[str] = []
    seen: set[str] = set()
    for item in raw_tokens:
        token = str(item).strip()
        if not token:
            raise ValueError("tokenizer.special_tokens 不能包含空 token")
        if token in seen:
            raise ValueError(f"tokenizer.special_tokens 存在重复 token：{token}")
        tokens.append(token)
        seen.add(token)
    return tokens


def count_characters(corpus_path: Path) -> Counter[str]:
    """统计语料中的字符频次。"""
    counter: Counter[str] = Counter()
    line_count = 0
    char_count = 0
    print(f"[tokenizer] 读取语料：{corpus_path.as_posix()}", flush=True)
    with corpus_path.open("r", encoding="utf-8") as file:
        for line in file:
            line_count += 1
            counter.update(line)
            char_count += len(line)
            if line_count % 100000 == 0:
                print(f"[tokenizer] 已统计行数：{line_count}，字符数：{char_count}", flush=True)

    print(f"[tokenizer] 字符统计完成，行数={line_count}，字符数={char_count}，唯一字符数={len(counter)}", flush=True)
    return counter


def build_vocab(counter: Counter[str], special_tokens: list[str], min_frequency: int) -> tuple[dict[str, int], list[str]]:
    """根据字符频次和特殊 token 构建稳定词表。"""
    token_to_id: dict[str, int] = {}
    id_to_token: list[str] = []

    for token in special_tokens:
        token_to_id[token] = len(id_to_token)
        id_to_token.append(token)

    ordinary_tokens = [item for item in counter.items() if item[1] >= min_frequency and item[0] not in token_to_id]
    ordinary_tokens.sort(key=lambda item: (-item[1], item[0]))

    for token, _ in ordinary_tokens:
        token_to_id[token] = len(id_to_token)
        id_to_token.append(token)

    return token_to_id, id_to_token


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写入 UTF-8 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_existing_stats(path: Path) -> dict[str, Any]:
    """读取已有统计文件，便于分阶段追加。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


# ---- CLI 入口 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构建 Poet-mini 字级 tokenizer")
    parser.add_argument("--config", default="configs/tokenizer.yaml", help="tokenizer 配置文件路径，默认相对 train 根目录")
    parser.add_argument("--corpus", default=None, help="覆盖配置中的纯文本语料路径")
    parser.add_argument("--output", default=None, help="覆盖配置中的 tokenizer.json 输出路径")
    parser.add_argument("--stats-output", default=None, help="覆盖配置中的统计报告输出路径")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    tokenizer_config = config.get("tokenizer", {})
    tokenizer_type = str(tokenizer_config.get("type", "char"))
    if tokenizer_type != "char":
        raise ValueError(f"第一版仅支持 char tokenizer，当前配置为：{tokenizer_type}")

    min_frequency = int(tokenizer_config.get("min_frequency", 1))
    if min_frequency < 1:
        raise ValueError("tokenizer.min_frequency 必须 >= 1")

    input_config = config.get("input", {})
    output_config = config.get("output", {})
    corpus_path = resolve_path(root, args.corpus) if args.corpus else resolve_path(root, input_config.get("corpus_text", "data/interim/poetry_corpus.txt"))
    tokenizer_path = resolve_path(root, args.output) if args.output else resolve_path(root, output_config.get("tokenizer_json", "data/processed/tokenizer.json"))
    stats_path = resolve_path(root, args.stats_output) if args.stats_output else resolve_path(root, output_config.get("stats_json", "outputs/logs/tokenizer_stats.json"))

    if not corpus_path.exists():
        raise FileNotFoundError(f"纯文本语料不存在：{corpus_path}")

    special_tokens = read_special_tokens(config)
    counter = count_characters(corpus_path)
    token_to_id, id_to_token = build_vocab(counter, special_tokens, min_frequency)

    token_frequencies = {token: counter.get(token, 0) for token in id_to_token if token not in special_tokens}
    tokenizer_data = {
        "version": 1,
        "type": "char",
        "special_tokens": special_tokens,
        "special_token_ids": {token: token_to_id[token] for token in special_tokens},
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "sep_token": "<SEP>",
        "min_frequency": min_frequency,
        "vocab_size": len(id_to_token),
    }
    write_json(tokenizer_path, tokenizer_data)

    stats = load_existing_stats(stats_path)
    stats["tokenizer"] = {
        "config": config_path.as_posix(),
        "input": corpus_path.as_posix(),
        "output": tokenizer_path.as_posix(),
        "type": "char",
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "special_token_count": len(special_tokens),
        "ordinary_token_count": len(id_to_token) - len(special_tokens),
        "vocab_size": len(id_to_token),
        "total_char_count": sum(counter.values()),
        "unique_char_count": len(counter),
        "dropped_char_count": sum(1 for _, count in counter.items() if count < min_frequency),
        "top_tokens": [{"token": token, "count": count} for token, count in counter.most_common(50)],
        "token_frequencies": token_frequencies,
    }
    write_json(stats_path, stats)

    print(json.dumps({"tokenizer": stats["tokenizer"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
