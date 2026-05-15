from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "interim_dir": "data/interim",
    "log_dir": "outputs/logs",
    "output_files": {
        "poems_jsonl": "poems.jsonl",
        "corpus_text": "poetry_corpus.txt",
        "corpus_stats": "corpus_stats.json",
    },
}

REQUIRED_FIELDS = [
    "id",
    "source_id",
    "source_name",
    "tags",
    "style_tags",
    "style_label_source",
    "author",
    "title",
    "rhythmic",
    "paragraphs",
    "text",
    "raw",
    "quality",
]

REQUIRED_TAG_FIELDS = [
    "dynasty",
    "genre",
    "corpus_type",
    "source_name",
]

REQUIRED_RAW_FIELDS = [
    "file",
    "index",
    "adapter",
]

REQUIRED_QUALITY_FIELDS = [
    "normalized_by",
    "compatible",
    "issues",
]


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
        raise RuntimeError("缺少 PyYAML 依赖，请在 Docker 镜像中安装 pyyaml 后运行数据流水线")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("配置文件顶层结构必须是 YAML 对象")
    return deep_merge(DEFAULT_CONFIG, loaded)


# ---- JSONL 读取 ----
def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str]]:
    """逐行读取 JSONL，返回行号、记录和错误信息。"""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, str(exc)
                continue

            if not isinstance(record, dict):
                yield line_number, None, "JSONL 行不是对象"
                continue

            yield line_number, record, ""


# ---- 统计逻辑 ----
def length_bucket(length: int) -> str:
    """按字符长度生成稳定的长度分桶。"""
    if length <= 20:
        return "000-020"
    if length <= 50:
        return "021-050"
    if length <= 100:
        return "051-100"
    if length <= 200:
        return "101-200"
    if length <= 500:
        return "201-500"
    return "501+"


def count_missing_nested_fields(record: dict[str, Any], field_name: str, required_fields: list[str], counter: Counter[str]) -> None:
    """统计嵌套对象字段缺失情况。"""
    nested = record.get(field_name)
    if not isinstance(nested, dict):
        counter[field_name] += 1
        return

    for required_field in required_fields:
        if required_field not in nested:
            counter[f"{field_name}.{required_field}"] += 1


def inspect_jsonl(input_path: Path) -> dict[str, Any]:
    """统计统一 JSONL 的样本质量和字段分布。"""
    print(f"[inspect] 统计 JSONL：{input_path.as_posix()}", flush=True)

    source_id_counter: Counter[str] = Counter()
    source_name_counter: Counter[str] = Counter()
    genre_counter: Counter[str] = Counter()
    dynasty_counter: Counter[str] = Counter()
    corpus_type_counter: Counter[str] = Counter()
    normalized_by_counter: Counter[str] = Counter()
    compatible_counter: Counter[str] = Counter()
    style_tag_counter: Counter[str] = Counter()
    style_source_counter: Counter[str] = Counter()
    field_missing_counter: Counter[str] = Counter()
    length_counter: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []

    total_samples = 0
    invalid_line_count = 0
    empty_text_count = 0
    total_text_chars = 0
    min_text_chars: int | None = None
    max_text_chars = 0

    for line_number, record, error in iter_jsonl(input_path):
        if error:
            invalid_line_count += 1
            if len(invalid_examples) < 10:
                invalid_examples.append({"line": line_number, "error": error})
            continue

        if record is None:
            continue

        total_samples += 1
        if total_samples % 10000 == 0:
            print(f"[inspect] 已统计样本数：{total_samples}", flush=True)
        for field_name in REQUIRED_FIELDS:

            if field_name not in record:
                field_missing_counter[field_name] += 1

        count_missing_nested_fields(record, "tags", REQUIRED_TAG_FIELDS, field_missing_counter)
        count_missing_nested_fields(record, "raw", REQUIRED_RAW_FIELDS, field_missing_counter)
        count_missing_nested_fields(record, "quality", REQUIRED_QUALITY_FIELDS, field_missing_counter)

        source_id_counter[str(record.get("source_id", ""))] += 1
        source_name_counter[str(record.get("source_name", ""))] += 1
        style_source_counter[str(record.get("style_label_source", ""))] += 1

        tags = record.get("tags", {})
        if isinstance(tags, dict):
            genre_counter[str(tags.get("genre", ""))] += 1
            dynasty_counter[str(tags.get("dynasty", ""))] += 1
            corpus_type_counter[str(tags.get("corpus_type", ""))] += 1

        quality = record.get("quality", {})
        if isinstance(quality, dict):
            normalized_by_counter[str(quality.get("normalized_by", ""))] += 1
            compatible_counter[str(quality.get("compatible", ""))] += 1

        style_tags = record.get("style_tags", [])
        if isinstance(style_tags, list):
            for style_tag in style_tags:
                style_tag_counter[str(style_tag)] += 1

        text = record.get("text", "")
        if not isinstance(text, str) or not text.strip():
            empty_text_count += 1
            continue

        text_length = len(text)
        total_text_chars += text_length
        min_text_chars = text_length if min_text_chars is None else min(min_text_chars, text_length)
        max_text_chars = max(max_text_chars, text_length)
        length_counter[length_bucket(text_length)] += 1

    average_text_chars = total_text_chars / total_samples if total_samples else 0
    return {
        "input": input_path.as_posix(),
        "total_samples": total_samples,
        "invalid_line_count": invalid_line_count,
        "empty_text_count": empty_text_count,
        "total_text_chars": total_text_chars,
        "average_text_chars": round(average_text_chars, 2),
        "min_text_chars": min_text_chars or 0,
        "max_text_chars": max_text_chars,
        "source_id_counts": dict(sorted(source_id_counter.items())),
        "source_name_counts": dict(sorted(source_name_counter.items())),
        "genre_counts": dict(sorted(genre_counter.items())),
        "dynasty_counts": dict(sorted(dynasty_counter.items())),
        "corpus_type_counts": dict(sorted(corpus_type_counter.items())),
        "normalized_by_counts": dict(sorted(normalized_by_counter.items())),
        "compatible_counts": dict(sorted(compatible_counter.items())),
        "style_tag_counts": dict(sorted(style_tag_counter.items())),
        "style_label_source_counts": dict(sorted(style_source_counter.items())),
        "length_buckets": dict(sorted(length_counter.items())),
        "missing_fields": dict(sorted(field_missing_counter.items())),
        "invalid_examples": invalid_examples,
    }


def inspect_text_file(text_path: Path) -> dict[str, Any]:
    """统计纯文本语料的基础规模。"""
    print(f"[inspect] 统计纯文本语料：{text_path.as_posix()}", flush=True)

    if not text_path.exists():
        return {
            "input": text_path.as_posix(),
            "exists": False,
        }

    line_count = 0
    non_empty_line_count = 0
    char_count = 0

    with text_path.open("r", encoding="utf-8") as file:
        for line in file:
            line_count += 1
            char_count += len(line)
            if line.strip():
                non_empty_line_count += 1

    return {
        "input": text_path.as_posix(),
        "exists": True,
        "file_size_bytes": text_path.stat().st_size,
        "line_count": line_count,
        "non_empty_line_count": non_empty_line_count,
        "char_count": char_count,
    }


# ---- CLI 入口 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="统计统一 JSONL 与纯文本语料质量")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径，默认相对 train 根目录")
    parser.add_argument("--input", default=None, help="统一 JSONL 输入路径，默认读取 data/interim/poems.jsonl")
    parser.add_argument("--text", default=None, help="纯文本语料路径，默认读取 data/interim/poetry_corpus.txt")
    parser.add_argument("--output", default=None, help="统计报告输出路径，默认写入 outputs/logs/corpus_stats.json")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    interim_dir = resolve_path(root, config.get("interim_dir", "data/interim"))
    log_dir = resolve_path(root, config.get("log_dir", "outputs/logs"))
    output_files = config.get("output_files", {})

    input_path = resolve_path(root, args.input) if args.input else interim_dir / output_files.get("poems_jsonl", "poems.jsonl")
    text_path = resolve_path(root, args.text) if args.text else interim_dir / output_files.get("corpus_text", "poetry_corpus.txt")
    output_path = resolve_path(root, args.output) if args.output else log_dir / output_files.get("corpus_stats", "corpus_stats.json")

    report = {
        "jsonl": inspect_jsonl(input_path),
        "corpus_text": inspect_text_file(text_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(json.dumps({"output": output_path.as_posix(), "jsonl": report["jsonl"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())