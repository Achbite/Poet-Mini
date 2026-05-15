from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}


@dataclass(frozen=True)
class TokenDataConfig:
    """token 数据读取配置。"""

    train_bin: Path
    val_bin: Path
    dtype: str
    block_size: int
    batch_size: int
    device: torch.device


class TokenBatcher:
    """连续 token 流 batch 构造器。"""

    def __init__(self, config: TokenDataConfig) -> None:
        """初始化训练集和验证集 token 视图。"""
        if config.dtype not in DTYPE_MAP:
            raise ValueError(f"不支持的 token dtype：{config.dtype}")
        if config.block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        if config.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        self.config = config
        self.numpy_dtype = DTYPE_MAP[config.dtype]
        self.train_tokens = self._load_tokens(config.train_bin)
        self.val_tokens = self._load_tokens(config.val_bin)
        self._check_length("train", self.train_tokens)
        self._check_length("val", self.val_tokens)

    def _load_tokens(self, path: Path) -> np.memmap:
        """以 memmap 方式读取 token 二进制文件。"""
        if not path.exists():
            raise FileNotFoundError(f"token 数据文件不存在：{path}")
        return np.memmap(path, dtype=self.numpy_dtype, mode="r")

    def _check_length(self, split: str, tokens: np.memmap) -> None:
        """检查 token 数量是否足够构造训练窗口。"""
        min_length = self.config.block_size + 1
        if len(tokens) < min_length:
            raise ValueError(f"{split} token 数量不足：{len(tokens)} < {min_length}")

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """随机采样一个 batch，并构造 next-token 训练目标。"""
        tokens = self.train_tokens if split == "train" else self.val_tokens
        max_start = len(tokens) - self.config.block_size - 1
        indices = torch.randint(max_start + 1, (self.config.batch_size,))

        x_list = []
        y_list = []
        for index in indices.tolist():
            chunk = np.asarray(tokens[index : index + self.config.block_size + 1], dtype=np.int64)
            x_list.append(torch.from_numpy(chunk[:-1].copy()))
            y_list.append(torch.from_numpy(chunk[1:].copy()))

        x = torch.stack(x_list).to(self.config.device, non_blocking=True)
        y = torch.stack(y_list).to(self.config.device, non_blocking=True)
        return x, y

    def token_count(self, split: str) -> int:
        """返回指定 split 的 token 数量。"""
        tokens = self.train_tokens if split == "train" else self.val_tokens
        return int(len(tokens))
