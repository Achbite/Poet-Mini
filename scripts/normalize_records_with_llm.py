from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "interim_dir": "data/interim",
    "output_files": {
        "llm_candidate_jsonl": "llm_candidates.jsonl",
        "llm_normalized_jsonl": "llm_normalized_records.jsonl",
    },
    "llm_normalization": {
        "enabled": False,
        "api_base_url": "https://api.openai.com/v1",
        "api_key_env": "POET_MINI_LLM_API_KEY",
        "model": "gpt-4o-mini",
        "request_timeout_seconds": 60,
        "max_retries": 2,
        "max_records": 100,
        "target_schema": "poem_record_v1",
    },
}

REQUIRED_NORMALIZED_FIELDS = [
    "source_id",
    "source_name",
    "tags",
    "author",
    "title",
    "rhythmic",
    "paragraphs",
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


# ---- 文本处理 ----
def clean_text(value: str) -> str:
    """清洗 LLM 返回的文本字段，只做空白归一化。"""
    return re.sub(r"\s+", " ", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


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


def make_sample_id(record: dict[str, Any], line_number: int) -> str:
    """根据来源信息生成 LLM 规范化样本 ID。"""
    source_id = str(record.get("source_id", "llm_source"))
    raw = record.get("raw", {}) if isinstance(record.get("raw"), dict) else {}
    raw_file = str(raw.get("file", "unknown"))
    raw_index = raw.get("index")
    safe_source = re.sub(r"[^0-9a-zA-Z_-]+", "-", source_id).strip("-").lower()
    safe_file = re.sub(r"[^0-9a-zA-Z]+", "-", Path(raw_file).stem).strip("-").lower() or "record"
    if isinstance(raw_index, int):
        return f"{safe_source}-{safe_file}-{raw_index:06d}-llm"
    return f"{safe_source}-{safe_file}-line-{line_number:06d}-llm"


# ---- Prompt 与 API ----
def build_prompt(candidate: dict[str, Any], target_schema: str) -> str:
    """构建格式规范化请求内容。"""
    source_id = candidate.get("source_id", "")
    tags = candidate.get("tags", {})
    raw = candidate.get("raw", {})
    preview = candidate.get("preview", "")
    raw_record = candidate.get("raw_record", "")
    raw_text = json.dumps(raw_record, ensure_ascii=False) if raw_record else str(preview)
    return (
        "你是古典诗词数据格式规范化助手。你的任务是把原始记录整理为目标 JSON schema，"
        "只能整理已有字段和正文，不得创作、补写、润色、改写诗句内容。"
        "如果无法确认字段，请使用空字符串或空数组。\n"
        f"目标 schema：{target_schema}\n"
        "必须严格输出 JSON 对象，字段包括："
        "source_id、source_name、tags、author、title、rhythmic、paragraphs。"
        "其中 paragraphs 必须是字符串数组，tags 必须保留原始 dynasty、genre、corpus_type、source_name 等标签。\n"
        f"source_id：{source_id}\n"
        f"tags：{json.dumps(tags, ensure_ascii=False)}\n"
        f"raw：{json.dumps(raw, ensure_ascii=False)}\n"
        f"原始记录：\n{raw_text[:4000]}"
    )


def request_chat_completion(
    api_base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_seconds: int,
) -> str:
    """调用 OpenAI 兼容 Chat Completions 接口。"""
    url = api_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出合法 JSON，不输出 Markdown。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_body = response.read().decode("utf-8")

    data = json.loads(response_body)
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("LLM 响应缺少 choices")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM 响应内容为空")
    return content.strip()


# ---- 响应校验 ----
def validate_normalized_data(data: Any) -> tuple[dict[str, Any] | None, str]:
    """校验 LLM 返回是否满足目标 schema。"""
    if not isinstance(data, dict):
        return None, "LLM 返回不是对象"

    for field_name in REQUIRED_NORMALIZED_FIELDS:
        if field_name not in data:
            return None, f"缺少字段：{field_name}"

    tags = data.get("tags")
    if not isinstance(tags, dict):
        return None, "tags 不是对象"

    paragraphs = data.get("paragraphs")
    if not isinstance(paragraphs, list):
        return None, "paragraphs 不是数组"

    cleaned_paragraphs = [clean_text(str(item)) for item in paragraphs if isinstance(item, str) and clean_text(item)]
    if not cleaned_paragraphs:
        return None, "paragraphs 为空"

    normalized: dict[str, Any] = {
        "source_id": clean_text(str(data.get("source_id", ""))),
        "source_name": clean_text(str(data.get("source_name", ""))),
        "tags": {str(key): clean_text(str(value)) for key, value in tags.items()},
        "author": clean_text(str(data.get("author", ""))),
        "title": clean_text(str(data.get("title", ""))),
        "rhythmic": clean_text(str(data.get("rhythmic", ""))),
        "paragraphs": cleaned_paragraphs,
    }
    return normalized, ""


def classify_record_failure(exc: Exception) -> str:
    """把请求异常转换为可读失败原因。"""
    return f"规范化失败：{exc}"


def normalize_candidate(
    candidate: dict[str, Any],
    line_number: int,
    llm_config: dict[str, Any],
    api_key: str,
) -> tuple[dict[str, Any] | None, str]:
    """对单条候选记录执行 LLM 规范化并校验。"""
    target_schema = str(llm_config.get("target_schema", "poem_record_v1"))
    prompt = build_prompt(candidate, target_schema)
    api_base_url = str(llm_config.get("api_base_url", DEFAULT_CONFIG["llm_normalization"]["api_base_url"]))
    model = str(llm_config.get("model", DEFAULT_CONFIG["llm_normalization"]["model"]))
    timeout_seconds = int(llm_config.get("request_timeout_seconds", 60))
    max_retries = int(llm_config.get("max_retries", 2))

    for attempt_index in range(max_retries + 1):
        try:
            content = request_chat_completion(api_base_url, api_key, model, prompt, timeout_seconds)
            data = json.loads(content)
            normalized, error = validate_normalized_data(data)
            if normalized is None:
                return None, error

            raw = candidate.get("raw", {}) if isinstance(candidate.get("raw"), dict) else {}
            source_id = normalized.get("source_id") or str(candidate.get("source_id", ""))
            tags = candidate.get("tags", {}) if isinstance(candidate.get("tags"), dict) else {}
            normalized["source_id"] = source_id
            normalized["source_name"] = normalized.get("source_name") or str(tags.get("source_name", source_id))
            normalized["tags"] = {**{str(key): str(value) for key, value in tags.items()}, **normalized.get("tags", {})}

            sample = {
                "id": make_sample_id({**candidate, "source_id": source_id}, line_number),
                "source_id": source_id,
                "source_name": normalized["source_name"],
                "tags": normalized["tags"],
                "style_tags": [],
                "style_label_source": "",
                "author": normalized["author"],
                "title": normalized["title"],
                "rhythmic": normalized["rhythmic"],
                "paragraphs": normalized["paragraphs"],
                "text": "",
                "raw": raw,
                "quality": {
                    "normalized_by": "llm",
                    "compatible": True,
                    "issues": [],
                },
            }
            sample["text"] = format_training_text(sample)
            return sample, ""
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if attempt_index >= max_retries:
                return None, classify_record_failure(exc)
            time.sleep(1 + attempt_index)

    return None, "规范化失败：未知错误"


# ---- 批量规范化 ----
def normalize_jsonl(
    input_path: Path,
    output_path: Path,
    llm_config: dict[str, Any],
    limit: int | None,
    force: bool,
) -> dict[str, Any]:
    """批量读取候选 JSONL 并写入通过校验的规范化结果。"""
    api_key_env = str(llm_config.get("api_key_env", "POET_MINI_LLM_API_KEY"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"缺少 API Key 环境变量：{api_key_env}")

    configured_limit = int(llm_config.get("max_records", 100))
    max_records = limit if limit is not None else configured_limit
    stats: dict[str, Any] = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "read_count": 0,
        "written_count": 0,
        "failed_count": 0,
        "invalid_line_count": 0,
        "skipped_limit_count": 0,
        "failures": [],
        "invalid_examples": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for line_number, candidate, error in iter_jsonl(input_path):
            if error:
                stats["invalid_line_count"] += 1
                if len(stats["invalid_examples"]) < 10:
                    stats["invalid_examples"].append({"line": line_number, "error": error})
                continue

            if candidate is None:
                continue

            stats["read_count"] += 1
            if not force and stats["written_count"] >= max_records:
                stats["skipped_limit_count"] += 1
                continue

            sample, failure = normalize_candidate(candidate, line_number, llm_config, api_key)
            if sample is None:
                stats["failed_count"] += 1
                if len(stats["failures"]) < 20:
                    stats["failures"].append({"line": line_number, "error": failure})
                continue

            output_file.write(json.dumps(sample, ensure_ascii=False))
            output_file.write("\n")
            stats["written_count"] += 1

    return stats


# ---- CLI 入口 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="通过外部 LLM API 将不兼容候选记录规范化为统一 JSONL schema")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径，默认相对 train 根目录")
    parser.add_argument("--input", default=None, help="候选 JSONL 路径，默认读取 data/interim/llm_candidates.jsonl")
    parser.add_argument("--output", default=None, help="规范化 JSONL 输出路径，默认写入 data/interim/llm_normalized_records.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="最多规范化的候选记录数量；默认使用配置中的 max_records")
    parser.add_argument("--force", action="store_true", help="忽略配置中的 max_records 限制，按 --limit 或配置继续处理")
    parser.add_argument("--allow-disabled", action="store_true", help="即使配置中 enabled=false 也允许手动执行")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    llm_config = config.get("llm_normalization", {})
    if not args.allow_disabled and not bool(llm_config.get("enabled", False)):
        print(json.dumps({"skipped": True, "reason": "llm_normalization.enabled=false"}, ensure_ascii=False, indent=2))
        return 0

    interim_dir = resolve_path(root, config.get("interim_dir", "data/interim"))
    output_files = config.get("output_files", {})
    input_path = resolve_path(root, args.input) if args.input else interim_dir / output_files.get("llm_candidate_jsonl", "llm_candidates.jsonl")
    output_path = resolve_path(root, args.output) if args.output else interim_dir / output_files.get("llm_normalized_jsonl", "llm_normalized_records.jsonl")

    stats = normalize_jsonl(input_path, output_path, llm_config, args.limit, args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
