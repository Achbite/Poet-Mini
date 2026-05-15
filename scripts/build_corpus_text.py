from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "interim_dir": "data/interim",
    "output_files": {
        "poems_jsonl": "poems.jsonl",
        "corpus_text": "poetry_corpus.txt",
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


# ---- 语料导出 ----
def export_corpus_text(input_path: Path, output_path: Path) -> dict[str, Any]:
    """从统一 JSONL 导出纯文本语料。"""
    stats: dict[str, Any] = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "read_count": 0,
        "written_count": 0,
        "empty_text_count": 0,
        "invalid_line_count": 0,
        "invalid_examples": [],
    }

    print(f"[build-text] 输入：{input_path.as_posix()}", flush=True)
    print(f"[build-text] 输出：{output_path.as_posix()}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for line_number, record, error in iter_jsonl(input_path):

            if error:
                stats["invalid_line_count"] += 1
                if len(stats["invalid_examples"]) < 10:
                    stats["invalid_examples"].append({"line": line_number, "error": error})
                continue

            stats["read_count"] += 1
            text = record.get("text", "") if record else ""
            if not isinstance(text, str) or not text.strip():
                stats["empty_text_count"] += 1
                continue

            output_file.write(text.strip())
            output_file.write("\n\n")
            stats["written_count"] += 1
            if stats["written_count"] % 10000 == 0:
                print(f"[build-text] 已写入文本样本数：{stats['written_count']}", flush=True)

    print(f"[build-text] 导出完成，写入文本样本数：{stats['written_count']}", flush=True)
    return stats



# ---- CLI 入口 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从统一 JSONL 导出 tokenizer 训练用纯文本语料")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径，默认相对 train 根目录")
    parser.add_argument("--input", default=None, help="统一 JSONL 输入路径，默认读取 data/interim/poems.jsonl")
    parser.add_argument("--output", default=None, help="纯文本语料输出路径，默认写入 data/interim/poetry_corpus.txt")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    interim_dir = resolve_path(root, config.get("interim_dir", "data/interim"))
    output_files = config.get("output_files", {})
    input_path = resolve_path(root, args.input) if args.input else interim_dir / output_files.get("poems_jsonl", "poems.jsonl")
    output_path = resolve_path(root, args.output) if args.output else interim_dir / output_files.get("corpus_text", "poetry_corpus.txt")

    stats = export_corpus_text(input_path, output_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())