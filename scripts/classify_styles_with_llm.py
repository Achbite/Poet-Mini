from __future__ import annotations

import argparse
import json
import os
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
        "poems_jsonl": "poems.jsonl",
        "styled_poems_jsonl": "poems_with_styles.jsonl",
    },
    "llm_classification": {
        "enabled": False,
        "api_base_url": "https://api.openai.com/v1",
        "api_key_env": "POET_MINI_LLM_API_KEY",
        "model": "gpt-4o-mini",
        "request_timeout_seconds": 60,
        "max_retries": 2,
        "style_labels": "抒情,写景,咏物,怀古,边塞,田园,送别,羁旅,闺怨,讽喻,哲理,宴饮,悼亡,爱国,未确定",
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


# ---- 路径与配置 ----
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


def split_labels(value: Any) -> list[str]:
    """解析允许输出的风格标签集合。"""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


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


# ---- 分类 Prompt ----
def build_prompt(record: dict[str, Any], style_labels: list[str]) -> str:
    """构建风格分类请求内容。"""
    label_text = "、".join(style_labels)
    title = record.get("title") or record.get("rhythmic") or ""
    author = record.get("author", "")
    text = str(record.get("text", ""))[:1200]
    return (
        "你是古典诗词风格分类助手。请只根据给定诗词内容进行多标签风格分类，"
        "必须从给定标签中选择 1 到 3 个标签，不要输出标签集合之外的内容。\n"
        f"可选标签：{label_text}\n"
        "请严格输出 JSON 对象，格式为：{\"style_tags\":[\"标签1\"],\"reason\":\"一句话理由\"}\n"
        f"标题或词牌：{title}\n"
        f"作者：{author}\n"
        f"正文：\n{text}"
    )


# ---- API 请求 ----
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


# ---- 响应解析 ----
def parse_style_response(content: str, style_labels: list[str]) -> tuple[list[str], str]:
    """解析并约束 LLM 输出的风格标签。"""
    data = json.loads(content)
    raw_tags = data.get("style_tags", [])
    reason = data.get("reason", "")
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    if not isinstance(raw_tags, list):
        raw_tags = []

    allowed_labels = set(style_labels)
    tags = [str(item).strip() for item in raw_tags if str(item).strip() in allowed_labels]
    if not tags:
        tags = ["未确定"] if "未确定" in allowed_labels else []

    return tags[:3], str(reason).strip()


# ---- 单条分类 ----
def classify_record(
    record: dict[str, Any],
    llm_config: dict[str, Any],
    style_labels: list[str],
    api_key: str,
) -> tuple[list[str], str]:
    """对单条样本执行风格分类。"""
    prompt = build_prompt(record, style_labels)
    api_base_url = str(llm_config.get("api_base_url", DEFAULT_CONFIG["llm_classification"]["api_base_url"]))
    model = str(llm_config.get("model", DEFAULT_CONFIG["llm_classification"]["model"]))
    timeout_seconds = int(llm_config.get("request_timeout_seconds", 60))
    max_retries = int(llm_config.get("max_retries", 2))

    for attempt_index in range(max_retries + 1):
        try:
            content = request_chat_completion(api_base_url, api_key, model, prompt, timeout_seconds)
            return parse_style_response(content, style_labels)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if attempt_index >= max_retries:
                return ["未确定"], f"分类失败：{exc}"
            time.sleep(1 + attempt_index)

    return ["未确定"], "分类失败：未知错误"


# ---- 批量分类 ----
def classify_jsonl(
    input_path: Path,
    output_path: Path,
    llm_config: dict[str, Any],
    limit: int | None,
    force: bool,
) -> dict[str, Any]:
    """批量读取 JSONL 并写入风格分类结果。"""
    api_key_env = str(llm_config.get("api_key_env", "POET_MINI_LLM_API_KEY"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"缺少 API Key 环境变量：{api_key_env}")

    style_labels = split_labels(llm_config.get("style_labels"))
    if not style_labels:
        style_labels = split_labels(DEFAULT_CONFIG["llm_classification"]["style_labels"])

    stats: dict[str, Any] = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "read_count": 0,
        "written_count": 0,
        "classified_count": 0,
        "skipped_existing_count": 0,
        "invalid_line_count": 0,
        "invalid_examples": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for line_number, record, error in iter_jsonl(input_path):
            if error:
                stats["invalid_line_count"] += 1
                if len(stats["invalid_examples"]) < 10:
                    stats["invalid_examples"].append({"line": line_number, "error": error})
                continue

            if record is None:
                continue

            stats["read_count"] += 1
            should_classify = force or not record.get("style_tags")
            if should_classify and (limit is None or stats["classified_count"] < limit):
                tags, reason = classify_record(record, llm_config, style_labels, api_key)
                record["style_tags"] = tags
                record["style_label_source"] = "llm"
                record["style_label_reason"] = reason
                stats["classified_count"] += 1
            else:
                stats["skipped_existing_count"] += 1

            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")
            stats["written_count"] += 1

    return stats


# ---- 参数解析 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="通过外部 LLM API 为诗词样本补充风格分类标签")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径，默认相对 train 根目录")
    parser.add_argument("--input", default=None, help="输入 JSONL 路径，默认读取 data/interim/poems.jsonl")
    parser.add_argument("--output", default=None, help="输出 JSONL 路径，默认写入 data/interim/poems_with_styles.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="最多分类的样本数量；默认处理全部缺失标签样本")
    parser.add_argument("--force", action="store_true", help="即使已有 style_tags 也重新请求 LLM 分类")
    parser.add_argument("--allow-disabled", action="store_true", help="即使配置中 enabled=false 也允许手动执行")
    return parser.parse_args()


# ---- 主入口 ----
def main() -> int:
    """脚本入口。"""
    root = project_root()
    args = parse_args()
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    llm_config = config.get("llm_classification", {})
    if not args.allow_disabled and not bool(llm_config.get("enabled", False)):
        print(json.dumps({"skipped": True, "reason": "llm_classification.enabled=false"}, ensure_ascii=False, indent=2))
        return 0

    interim_dir = resolve_path(root, config.get("interim_dir", "data/interim"))
    output_files = config.get("output_files", {})
    input_path = resolve_path(root, args.input) if args.input else interim_dir / output_files.get("poems_jsonl", "poems.jsonl")
    output_path = resolve_path(root, args.output) if args.output else interim_dir / output_files.get("styled_poems_jsonl", "poems_with_styles.jsonl")

    stats = classify_jsonl(input_path, output_path, llm_config, args.limit, args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())