from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from typing import Any, Iterator

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "data_root": "/mnt/corpus",
    },
    "interim_dir": "data/interim",
    "output_files": {
        "poems_jsonl": "poems.jsonl",
        "rejected_jsonl": "rejected_records.jsonl",
        "llm_candidate_jsonl": "llm_candidates.jsonl",
    },
    "cleaning": {
        "min_text_length": 4,
        "normalize_whitespace": True,
        "keep_punctuation": True,
    },
    "incompatible_records": {
        "max_preview_chars": 1200,
        "keep_raw_record": True,
    },
    "corpus_sources": [],
}

SUPPORTED_ADAPTERS = {"chinese_poetry_json_array"}


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


# ---- 数据源解析 ----
def enabled_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    """读取启用的数据源序列。"""
    sources = config.get("corpus_sources", [])
    if not isinstance(sources, list):
        raise ValueError("corpus_sources 必须是 YAML 序列")

    enabled: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"corpus_sources[{index}] 必须是对象")
        if source.get("enabled", True):
            enabled.append(source)
    return enabled


def source_id(source: dict[str, Any], index: int) -> str:
    """获取稳定的数据源 ID。"""
    value = source.get("id", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"source_{index}"


def source_tags(source: dict[str, Any]) -> dict[str, str]:
    """读取数据源标签并转成字符串映射。"""
    tags = source.get("tags", {})
    if not isinstance(tags, dict):
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def source_adapter(source: dict[str, Any]) -> str:
    """读取数据源适配器名称。"""
    format_config = source.get("format", {})
    if not isinstance(format_config, dict):
        return ""
    return str(format_config.get("adapter", "")).strip()


def source_llm_on_incompatible(source: dict[str, Any]) -> bool:
    """读取不兼容记录是否进入 LLM 候选集。"""
    format_config = source.get("format", {})
    if not isinstance(format_config, dict):
        return False
    return bool(format_config.get("llm_on_incompatible", False))


def split_patterns(value: Any) -> list[str]:
    """解析文件匹配规则，支持列表或分号分隔字符串。"""
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def resolve_data_root(root: Path, config: dict[str, Any], override: str | None) -> Path:
    """解析语料挂载根目录。"""
    if override:
        return resolve_path(root, override)

    runtime = config.get("runtime", {})
    if isinstance(runtime, dict):
        return resolve_path(root, runtime.get("data_root", DEFAULT_CONFIG["runtime"]["data_root"]))
    return resolve_path(root, DEFAULT_CONFIG["runtime"]["data_root"])


def resolve_source_dir(data_root: Path, source: dict[str, Any]) -> Path:
    """解析单个数据源目录。"""
    path_value = source.get("path", "")
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (data_root / path).resolve()


def natural_sort_key(path: Path) -> list[Any]:
    """按文件名中的数字自然排序，保证分片顺序稳定。"""
    parts = re.split(r"(\d+)", path.as_posix())
    return [int(part) if part.isdigit() else part for part in parts]


def find_source_files(source_dir: Path, patterns: list[str]) -> list[Path]:
    """根据数据源目录和 glob 规则查找文件。"""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in source_dir.glob(pattern) if path.is_file())
    return sorted(files, key=natural_sort_key)


# ---- 通用清洗 ----
def safe_relative_path(path: Path, base_dir: Path) -> str:
    """生成稳定展示路径，无法相对化时退回绝对路径。"""
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def load_json_array(path: Path) -> list[Any]:
    """读取 JSON 数组文件。"""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("文件顶层结构不是 JSON 数组")
    return data


def clean_text(value: str, normalize_whitespace: bool) -> str:
    """清洗单个文本字段，只做低风险空白归一化。"""
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_optional_field(record: dict[str, Any], field_name: str, normalize_whitespace: bool) -> str:
    """读取并清洗可选字符串字段。"""
    value = record.get(field_name, "")
    if not isinstance(value, str):
        return ""
    return clean_text(value, normalize_whitespace)


def clean_style_tags(record: dict[str, Any], normalize_whitespace: bool) -> list[str]:
    """读取并清洗原始样本中的分类标签。"""
    value = record.get("tags", [])
    if isinstance(value, str):
        tag = clean_text(value, normalize_whitespace)
        return [tag] if tag else []
    if not isinstance(value, list):
        return []

    tags: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = clean_text(item, normalize_whitespace)
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def clean_paragraphs(record: dict[str, Any], normalize_whitespace: bool) -> tuple[list[str], int]:
    """清洗 paragraphs 字段，并返回非字符串段落数量。"""

    raw_paragraphs = record.get("paragraphs")
    if not isinstance(raw_paragraphs, list):
        return [], 0

    paragraphs: list[str] = []
    invalid_paragraph_count = 0
    for item in raw_paragraphs:
        if not isinstance(item, str):
            invalid_paragraph_count += 1
            continue

        paragraph = clean_text(item, normalize_whitespace)
        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs, invalid_paragraph_count


def format_training_text(sample: dict[str, Any]) -> str:
    """按统一规则拼接后续 tokenizer 使用的主文本。"""
    lines: list[str] = []
    tags = sample.get("tags", {}) if isinstance(sample.get("tags"), dict) else {}
    genre = str(tags.get("genre", ""))
    title = str(sample.get("title", ""))
    rhythmic = str(sample.get("rhythmic", ""))
    author = str(sample.get("author", ""))

    if genre == "ci" and rhythmic:
        lines.append(f"词牌：{rhythmic}")
    elif genre == "qu" and rhythmic:
        lines.append(f"曲牌：{rhythmic}")
    elif title:
        lines.append(f"标题：{title}")

    if author:
        lines.append(f"作者：{author}")

    paragraphs = sample.get("paragraphs", [])
    if isinstance(paragraphs, list):
        lines.extend(str(item) for item in paragraphs if str(item).strip())
    return "\n".join(lines)


def make_sample_id(current_source_id: str, raw_file: Path, raw_index: int) -> str:
    """生成稳定且跨来源不冲突的样本 ID。"""
    safe_source = re.sub(r"[^0-9a-zA-Z_-]+", "-", current_source_id).strip("-").lower()
    safe_stem = re.sub(r"[^0-9a-zA-Z]+", "-", raw_file.stem).strip("-").lower()
    if not safe_stem:
        safe_stem = hashlib.sha1(raw_file.stem.encode("utf-8")).hexdigest()[:12]
    return f"{safe_source}-{safe_stem}-{raw_index:06d}"


def record_preview(record: Any, max_chars: int) -> str:
    """生成不兼容记录预览，避免候选集过大。"""

    try:
        text = json.dumps(record, ensure_ascii=False)
    except TypeError:
        text = repr(record)
    return text[:max_chars]


def make_issue_record(
    source: dict[str, Any],
    current_source_id: str,
    source_dir: Path,
    raw_file: Path | None,
    raw_index: int | None,
    reason: str,
    record: Any,
    max_preview_chars: int,
    keep_raw_record: bool,
) -> dict[str, Any]:
    """构建不兼容记录，供人工检查或 LLM 规范化使用。"""
    issue: dict[str, Any] = {
        "source_id": current_source_id,
        "tags": source_tags(source),
        "raw": {
            "file": safe_relative_path(raw_file, source_dir) if raw_file else "",
            "index": raw_index,
            "adapter": source_adapter(source),
        },
        "quality": {
            "compatible": False,
            "issues": [reason],
        },
        "preview": record_preview(record, max_preview_chars),
    }

    if keep_raw_record:
        try:
            json.dumps(record, ensure_ascii=False)
            issue["raw_record"] = record
        except TypeError:
            issue["raw_record"] = issue["preview"]

    return issue


# ---- 样本构建 ----
def build_sample(
    record: Any,
    source: dict[str, Any],
    current_source_id: str,
    raw_file: Path,
    source_dir: Path,
    raw_index: int,
    min_text_length: int,
    normalize_whitespace: bool,
) -> tuple[dict[str, Any] | None, str, int]:
    """将一条原始记录转换为统一 JSONL 样本。"""
    if not isinstance(record, dict):
        return None, "record_not_object", 0

    paragraphs, invalid_paragraph_count = clean_paragraphs(record, normalize_whitespace)
    if not paragraphs:
        return None, "empty_paragraphs", invalid_paragraph_count

    body_text = "".join(paragraphs)
    if len(body_text) < min_text_length:
        return None, "text_too_short", invalid_paragraph_count

    tags = source_tags(source)
    style_tags = clean_style_tags(record, normalize_whitespace)
    dynasty = clean_optional_field(record, "dynasty", normalize_whitespace) or tags.get("dynasty", "")
    source_name = tags.get("source_name", current_source_id)
    sample: dict[str, Any] = {
        "id": make_sample_id(current_source_id, raw_file, raw_index),
        "source_id": current_source_id,
        "source_name": source_name,
        "tags": {
            **tags,
            "dynasty": dynasty,
        },
        "style_tags": style_tags,
        "style_label_source": "source_tags" if style_tags else "",
        "author": clean_optional_field(record, "author", normalize_whitespace),

        "title": clean_optional_field(record, "title", normalize_whitespace),
        "rhythmic": clean_optional_field(record, "rhythmic", normalize_whitespace),
        "paragraphs": paragraphs,
        "text": "",
        "raw": {
            "file": safe_relative_path(raw_file, source_dir),
            "index": raw_index,
            "adapter": source_adapter(source),
        },
        "quality": {
            "normalized_by": "rule",
            "compatible": True,
            "issues": [],
        },
    }
    sample["text"] = format_training_text(sample)
    return sample, "", invalid_paragraph_count


def iter_source_records(source_dir: Path, files: list[Path]) -> Iterator[tuple[Path, int, Any]]:
    """按文件和记录序号遍历某个数据源的原始记录。"""
    for raw_file in files:
        records = load_json_array(raw_file)
        for raw_index, record in enumerate(records):
            yield raw_file, raw_index, record


def write_jsonl_record(file: Any, record: dict[str, Any]) -> None:
    """写入单条 JSONL 记录。"""
    file.write(json.dumps(record, ensure_ascii=False))
    file.write("\n")


def build_jsonl(
    config: dict[str, Any],
    data_root: Path,
    output_path: Path,
    rejected_path: Path,
    llm_candidate_path: Path,
) -> dict[str, Any]:
    """生成统一 JSONL 文件，并返回处理统计。"""
    cleaning = config.get("cleaning", {})
    min_text_length = int(cleaning.get("min_text_length", 4))
    normalize_whitespace = bool(cleaning.get("normalize_whitespace", True))
    incompatible_config = config.get("incompatible_records", {})
    max_preview_chars = int(incompatible_config.get("max_preview_chars", 1200))
    keep_raw_record = bool(incompatible_config.get("keep_raw_record", True))

    stats: dict[str, Any] = {
        "output": output_path.as_posix(),
        "rejected_output": rejected_path.as_posix(),
        "llm_candidate_output": llm_candidate_path.as_posix(),
        "written_count": 0,
        "rejected_count": 0,
        "llm_candidate_count": 0,
        "source_counts": {},
        "genre_counts": {},
        "dynasty_counts": {},
        "skip_reasons": {},
        "invalid_paragraph_count": 0,
        "read_errors": [],
    }
    source_counter: Counter[str] = Counter()
    genre_counter: Counter[str] = Counter()
    dynasty_counter: Counter[str] = Counter()
    skip_counter: Counter[str] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    llm_candidate_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file, rejected_path.open("w", encoding="utf-8") as rejected_file, llm_candidate_path.open("w", encoding="utf-8") as llm_candidate_file:
        for index, source in enumerate(enabled_sources(config)):
            current_source_id = source_id(source, index)
            source_dir = resolve_source_dir(data_root, source)
            adapter = source_adapter(source)
            patterns = split_patterns(source.get("patterns"))
            files = find_source_files(source_dir, patterns)
            llm_on_incompatible = source_llm_on_incompatible(source)
            print(f"[build-jsonl] 数据源 {current_source_id}：目录={source_dir.as_posix()}，文件数={len(files)}", flush=True)

            if adapter not in SUPPORTED_ADAPTERS:
                issue = make_issue_record(source, current_source_id, source_dir, None, None, f"unsupported_adapter:{adapter}", {}, max_preview_chars, keep_raw_record)
                target_file = llm_candidate_file if llm_on_incompatible else rejected_file

                write_jsonl_record(target_file, issue)
                stats["llm_candidate_count" if llm_on_incompatible else "rejected_count"] += 1
                skip_counter["unsupported_adapter"] += 1
                continue

            try:
                for file_index, raw_file in enumerate(files, start=1):
                    print(f"[build-jsonl] 数据源 {current_source_id}：处理文件 {file_index}/{len(files)} {raw_file.name}", flush=True)
                    records = load_json_array(raw_file)
                    for raw_index, record in enumerate(records):
                        sample, skip_reason, invalid_paragraph_count = build_sample(
                            record=record,
                            source=source,
                            current_source_id=current_source_id,
                            raw_file=raw_file,
                            source_dir=source_dir,
                            raw_index=raw_index,
                            min_text_length=min_text_length,
                            normalize_whitespace=normalize_whitespace,
                        )
                        stats["invalid_paragraph_count"] += invalid_paragraph_count

                        if sample is None:
                            issue = make_issue_record(source, current_source_id, source_dir, raw_file, raw_index, skip_reason, record, max_preview_chars, keep_raw_record)
                            target_file = llm_candidate_file if llm_on_incompatible else rejected_file
                            write_jsonl_record(target_file, issue)
                            stats["llm_candidate_count" if llm_on_incompatible else "rejected_count"] += 1
                            skip_counter[skip_reason] += 1
                            continue

                        write_jsonl_record(output_file, sample)
                        source_counter[sample["source_name"]] += 1
                        tags = sample.get("tags", {})
                        genre_counter[str(tags.get("genre", ""))] += 1
                        dynasty_counter[str(tags.get("dynasty", ""))] += 1
                        stats["written_count"] += 1
                        if stats["written_count"] % 10000 == 0:
                            print(f"[build-jsonl] 已写入样本数：{stats['written_count']}", flush=True)
                print(f"[build-jsonl] 数据源 {current_source_id} 处理完成，累计写入={stats['written_count']}，拒绝={stats['rejected_count']}", flush=True)

            except (OSError, json.JSONDecodeError, ValueError) as exc:
                error_record = {
                    "source_id": current_source_id,
                    "source_dir": source_dir.as_posix(),
                    "error": str(exc),
                }
                stats["read_errors"].append(error_record)
                issue = make_issue_record(source, current_source_id, source_dir, None, None, f"read_error:{exc}", error_record, max_preview_chars, keep_raw_record)
                target_file = llm_candidate_file if llm_on_incompatible else rejected_file
                write_jsonl_record(target_file, issue)
                stats["llm_candidate_count" if llm_on_incompatible else "rejected_count"] += 1
                skip_counter["read_error"] += 1

    stats["source_counts"] = dict(sorted(source_counter.items()))
    stats["genre_counts"] = dict(sorted(genre_counter.items()))
    stats["dynasty_counts"] = dict(sorted(dynasty_counter.items()))
    stats["skip_reasons"] = dict(sorted(skip_counter.items()))
    return stats


# ---- CLI 入口 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="将 YAML 配置声明的语料转换为统一 JSONL")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径，默认相对 train 根目录")
    parser.add_argument("--data-root", default=None, help="覆盖配置中的语料挂载根目录")
    parser.add_argument("--output", default=None, help="统一 JSONL 输出路径，默认写入 data/interim")
    parser.add_argument("--rejected-output", default=None, help="不兼容记录输出路径，默认写入 data/interim")
    parser.add_argument("--llm-candidate-output", default=None, help="LLM 候选记录输出路径，默认写入 data/interim")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    data_root = resolve_data_root(root, config, args.data_root)
    interim_dir = resolve_path(root, config.get("interim_dir", "data/interim"))
    output_files = config.get("output_files", {})

    output_path = resolve_path(root, args.output) if args.output else interim_dir / output_files.get("poems_jsonl", "poems.jsonl")
    rejected_path = resolve_path(root, args.rejected_output) if args.rejected_output else interim_dir / output_files.get("rejected_jsonl", "rejected_records.jsonl")
    llm_candidate_path = resolve_path(root, args.llm_candidate_output) if args.llm_candidate_output else interim_dir / output_files.get("llm_candidate_jsonl", "llm_candidates.jsonl")

    stats = build_jsonl(config, data_root, output_path, rejected_path, llm_candidate_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())