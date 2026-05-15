from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - 运行环境提示
    yaml = None


PIPELINE_STEPS = [
    "scan",
    "build-jsonl",
    "build-text",
    "inspect",
]

TOKENIZER_STEPS = [
    "train-tokenizer",
    "prepare-tokenized",
]


OPTIONAL_LLM_STEPS = [
    "normalize-llm",
    "merge-normalized",
]


# ---- 工程根目录 ----
def project_root() -> Path:
    """返回 train 工程根目录。"""
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    """将配置中的相对路径解析到 train 工程根目录。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


# ---- 配置加载 ----
def load_config(config_path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    if not config_path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("缺少 PyYAML 依赖，请在 Docker 镜像中安装 pyyaml 后运行数据流水线")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("配置文件顶层结构必须是 YAML 对象")
    return loaded


# ---- Python 脚本路径 ----
def script_path(script_name: str) -> Path:
    """返回 scripts 目录下的脚本绝对路径。"""
    return Path(__file__).resolve().parent / script_name


# ---- 子进程执行 ----
def run_command(command: list[str], step_name: str) -> dict[str, object]:
    """执行单个流水线步骤，实时转发子进程输出。"""
    print(f"[pipeline] 开始步骤：{step_name}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=project_root(),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            text = line.rstrip()
            if text:
                output_lines.append(text)
                print(f"[{step_name}] {text}", flush=True)

    returncode = process.wait()
    print(f"[pipeline] 结束步骤：{step_name}，returncode={returncode}", flush=True)
    return {
        "step": step_name,
        "returncode": returncode,
        "stdout": "\n".join(output_lines).strip(),
        "stderr": "",
    }


# ---- 步骤命令构建 ----
def build_step_command(step_name: str, data_config_path: str, tokenizer_config_path: str) -> list[str]:
    """根据步骤名构建对应脚本命令。"""
    script_map = {
        "scan": ("scan_corpus.py", data_config_path),
        "build-jsonl": ("build_poems_jsonl.py", data_config_path),
        "build-text": ("build_corpus_text.py", data_config_path),
        "inspect": ("inspect_corpus.py", data_config_path),
        "normalize-llm": ("normalize_records_with_llm.py", data_config_path),
        "train-tokenizer": ("train_tokenizer.py", tokenizer_config_path),
        "prepare-tokenized": ("prepare_tokenized_data.py", tokenizer_config_path),
    }
    if step_name not in script_map:
        raise ValueError(f"未知流水线步骤：{step_name}")

    script_name, config_path = script_map[step_name]
    return [sys.executable, "-u", str(script_path(script_name)), "--config", config_path]


# ---- LLM 分类命令构建 ----
def build_llm_style_command(config_path: str) -> list[str]:
    """构建可选 LLM 风格分类命令。"""

    return [sys.executable, "-u", str(script_path("classify_styles_with_llm.py")), "--config", config_path]


# ---- LLM 规范化结果合并 ----
def iter_jsonl(path: Path) -> list[str]:

    """读取 JSONL 原始行，忽略空行。"""
    if not path.exists():
        return []

    lines: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def merge_normalized_records(config_path: str) -> dict[str, object]:
    """将 LLM 规范化结果追加合并到主 poems.jsonl。"""
    root = project_root()
    config = load_config(resolve_path(root, config_path))
    interim_dir = resolve_path(root, config.get("interim_dir", "data/interim"))
    output_files = config.get("output_files", {}) if isinstance(config.get("output_files", {}), dict) else {}

    poems_path = interim_dir / output_files.get("poems_jsonl", "poems.jsonl")
    normalized_path = interim_dir / output_files.get("llm_normalized_jsonl", "llm_normalized_records.jsonl")
    normalized_lines = iter_jsonl(normalized_path)

    if not normalized_lines:
        return {
            "step": "merge-normalized",
            "returncode": 0,
            "stdout": json.dumps({"merged_count": 0, "reason": "no_normalized_records"}, ensure_ascii=False),
            "stderr": "",
        }

    poems_path.parent.mkdir(parents=True, exist_ok=True)
    with poems_path.open("a", encoding="utf-8") as output_file:
        for line in normalized_lines:
            output_file.write(line)
            output_file.write("\n")

    return {
        "step": "merge-normalized",
        "returncode": 0,
        "stdout": json.dumps({"merged_count": len(normalized_lines), "output": poems_path.as_posix()}, ensure_ascii=False),
        "stderr": "",
    }


# ---- 参数解析 ----
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Docker 数据清洗与语料构建统一入口")
    parser.add_argument("--config", default="configs/data.yaml", help="数据配置文件路径，默认相对 train 根目录")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer.yaml", help="tokenizer 配置文件路径，默认相对 train 根目录")

    parser.add_argument(
        "--steps",
        default=",".join(PIPELINE_STEPS),
        help="要执行的步骤，使用逗号分隔，可选 scan,build-jsonl,normalize-llm,merge-normalized,build-text,inspect,train-tokenizer,prepare-tokenized",
    )

    parser.add_argument("--with-llm-normalization", action="store_true", help="在基础清洗后执行可选 LLM 格式规范化并合并结果")
    parser.add_argument("--with-llm-style", action="store_true", help="在基础清洗后执行可选 LLM 风格分类")
    parser.add_argument("--with-tokenizer", action="store_true", help="在基础清洗后执行 tokenizer 构建和 token 序列化")
    parser.add_argument("--continue-on-error", action="store_true", help="某一步失败后继续执行后续步骤")

    return parser.parse_args()


# ---- 主入口 ----
def main() -> int:
    """脚本入口。"""
    args = parse_args()
    requested_steps = [item.strip() for item in args.steps.split(",") if item.strip()]
    if args.with_llm_normalization:
        requested_steps = ["scan", "build-jsonl", *OPTIONAL_LLM_STEPS, "build-text", "inspect"]
    if args.with_tokenizer:
        requested_steps = [*requested_steps, *TOKENIZER_STEPS]

    print(f"[pipeline] 数据流水线启动，步骤：{','.join(requested_steps)}", flush=True)

    results: list[dict[str, object]] = []
    exit_code = 0
    llm_normalization_skipped = False

    for step_name in requested_steps:
        if step_name == "merge-normalized" and llm_normalization_skipped:

            result = {
                "step": "merge-normalized",
                "returncode": 0,
                "stdout": json.dumps({"skipped": True, "reason": "llm_normalization_skipped"}, ensure_ascii=False),
                "stderr": "",
            }
        elif step_name == "merge-normalized":
            result = merge_normalized_records(args.config)
        else:
            command = build_step_command(step_name, args.config, args.tokenizer_config)
            result = run_command(command, step_name)

        results.append(result)

        if step_name == "normalize-llm" and result["returncode"] == 0:
            try:
                llm_output = json.loads(str(result.get("stdout", "{}")))
                llm_normalization_skipped = bool(llm_output.get("skipped", False))
            except json.JSONDecodeError:
                llm_normalization_skipped = False

        if result["returncode"] != 0:
            exit_code = int(result["returncode"])
            if not args.continue_on_error:
                break

    if args.with_llm_style and exit_code == 0:
        llm_result = run_command(build_llm_style_command(args.config), "llm-style")
        results.append(llm_result)
        if llm_result["returncode"] != 0:
            exit_code = int(llm_result["returncode"])

    print(f"[pipeline] 数据流水线完成，exit_code={exit_code}", flush=True)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return exit_code



if __name__ == "__main__":
    raise SystemExit(main())