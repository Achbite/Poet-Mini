from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "data_root": "/mnt/corpus",
    },
    "log_dir": "outputs/logs",
    "output_files": {
        "scan_report": "corpus_scan.json",
    },
    "scan": {
        "sample_size": 3,
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


# ---- JSON 扫描 ----
def load_json_array(path: Path) -> list[Any]:
    """读取 JSON 数组文件。"""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("文件顶层结构不是 JSON 数组")
    return data


def summarize_record(record: Any) -> dict[str, Any]:
    """提取样例记录摘要，避免扫描报告过大。"""
    if not isinstance(record, dict):
        return {"type": type(record).__name__}

    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list):
        paragraph_count = len(paragraphs)
        text_preview = "".join(item for item in paragraphs[:2] if isinstance(item, str))[:80]
    else:
        paragraph_count = 0
        text_preview = ""

    return {
        "fields": sorted(record.keys()),
        "author": record.get("author", ""),
        "title": record.get("title", ""),
        "rhythmic": record.get("rhythmic", ""),
        "paragraph_count": paragraph_count,
        "text_preview": text_preview,
    }


def safe_relative_path(path: Path, base_dir: Path) -> str:
    """生成稳定展示路径，无法相对化时退回绝对路径。"""
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def scan_file(path: Path, source_dir: Path, sample_size: int, adapter: str) -> dict[str, Any]:
    """扫描单个 JSON 文件的记录数量、字段分布和样例。"""
    result: dict[str, Any] = {
        "file": safe_relative_path(path, source_dir),
        "adapter": adapter,
        "compatible": False,
        "record_count": 0,
        "field_counts": {},
        "non_object_count": 0,
        "samples": [],
        "error": "",
    }

    if adapter not in SUPPORTED_ADAPTERS:
        result["error"] = f"不支持的 adapter：{adapter}"
        return result

    try:
        records = load_json_array(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["error"] = str(exc)
        return result

    field_counter: Counter[str] = Counter()
    non_object_count = 0
    samples: list[dict[str, Any]] = []

    for record in records:
        if isinstance(record, dict):
            field_counter.update(record.keys())
        else:
            non_object_count += 1

        if len(samples) < sample_size:
            samples.append(summarize_record(record))

    result["compatible"] = True
    result["record_count"] = len(records)
    result["field_counts"] = dict(sorted(field_counter.items()))
    result["non_object_count"] = non_object_count
    result["samples"] = samples
    return result


def build_report(config: dict[str, Any], data_root: Path, sample_size: int) -> dict[str, Any]:
    """构建全部启用数据源的扫描报告。"""
    report: dict[str, Any] = {
        "data_root": data_root.as_posix(),
        "sources": {},
        "summary": {
            "source_count": 0,
            "file_count": 0,
            "compatible_file_count": 0,
            "incompatible_file_count": 0,
            "record_count": 0,
            "llm_candidate_source_count": 0,
        },
    }

    for index, source in enumerate(enabled_sources(config)):
        current_source_id = source_id(source, index)
        tags = source_tags(source)
        adapter = source_adapter(source)
        source_dir = resolve_source_dir(data_root, source)
        patterns = split_patterns(source.get("patterns"))
        format_config = source.get("format", {}) if isinstance(source.get("format", {}), dict) else {}
        llm_on_incompatible = bool(format_config.get("llm_on_incompatible", False))

        files = find_source_files(source_dir, patterns)
        print(f"[scan] 数据源 {current_source_id}：目录={source_dir.as_posix()}，文件数={len(files)}", flush=True)
        file_reports: list[dict[str, Any]] = []
        for file_index, path in enumerate(files, start=1):
            if file_index == 1 or file_index % 25 == 0 or file_index == len(files):
                print(f"[scan] 数据源 {current_source_id}：扫描文件 {file_index}/{len(files)} {path.name}", flush=True)
            file_reports.append(scan_file(path, source_dir, sample_size, adapter))
        compatible_file_count = sum(1 for item in file_reports if item["compatible"])

        incompatible_file_count = sum(1 for item in file_reports if not item["compatible"])
        record_count = sum(int(item["record_count"]) for item in file_reports)

        report["sources"][current_source_id] = {
            "source_id": current_source_id,
            "source_dir": source_dir.as_posix(),
            "patterns": patterns,
            "tags": tags,
            "adapter": adapter,
            "llm_on_incompatible": llm_on_incompatible,
            "file_count": len(files),
            "compatible_file_count": compatible_file_count,
            "incompatible_file_count": incompatible_file_count,
            "record_count": record_count,
            "files": file_reports,
        }
        report["summary"]["source_count"] += 1
        report["summary"]["file_count"] += len(files)
        report["summary"]["compatible_file_count"] += compatible_file_count
        report["summary"]["incompatible_file_count"] += incompatible_file_count
        report["summary"]["record_count"] += record_count
        if llm_on_incompatible:
            report["summary"]["llm_candidate_source_count"] += 1

    return report


# ---- CLI 入口 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="扫描 YAML 配置声明的原始语料结构")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径，默认相对 train 根目录")
    parser.add_argument("--data-root", default=None, help="覆盖配置中的语料挂载根目录")
    parser.add_argument("--output", default=None, help="扫描报告输出路径，默认写入 outputs/logs")
    parser.add_argument("--sample-size", type=int, default=None, help="每个文件保留的样例记录数量")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    data_root = resolve_data_root(root, config, args.data_root)
    log_dir = resolve_path(root, config.get("log_dir", "outputs/logs"))
    output_name = config.get("output_files", {}).get("scan_report", "corpus_scan.json")
    output_path = resolve_path(root, args.output) if args.output else log_dir / output_name
    sample_size = args.sample_size if args.sample_size is not None else int(config.get("scan", {}).get("sample_size", 3))

    report = build_report(config, data_root, sample_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(json.dumps({"output": output_path.as_posix(), "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())