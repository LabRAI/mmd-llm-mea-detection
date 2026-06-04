#!/usr/bin/env python3
"""Data utilities for DATE batch-level anomaly detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class QueryDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def read_jsonl_queries(path: Path, text_field: str, limit: int | None = None) -> list[str]:
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            queries.append(text.strip())
            if limit is not None and len(queries) >= limit:
                break
    if not queries:
        raise ValueError(f"No usable '{text_field}' strings found in {path}")
    return queries


def resolve_dataset_dir(data_root: Path, dataset: str) -> Path:
    exact = data_root / dataset
    if exact.exists():
        return exact
    if data_root.exists():
        matches = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower() == dataset.lower()]
        if len(matches) == 1:
            return matches[0]
    available = sorted(p.name for p in data_root.iterdir() if p.is_dir()) if data_root.exists() else []
    available_text = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"Dataset '{dataset}' was not found under {data_root}. Available datasets: {available_text}"
    )


def default_dataset_paths(data_root: Path, dataset: str) -> tuple[Path, Path, str]:
    dataset_dir = resolve_dataset_dir(data_root, dataset)
    normal_path = dataset_dir / "normal" / "queries.jsonl"
    attacker_path = dataset_dir / "attacker" / "queries.jsonl"
    missing = [str(path) for path in (normal_path, attacker_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset must contain normal/queries.jsonl and attacker/queries.jsonl. "
            f"Missing: {', '.join(missing)}"
        )
    return normal_path, attacker_path, dataset_dir.name


def default_attacker_path(data_root: Path, dataset: str) -> tuple[Path, str]:
    dataset_dir = resolve_dataset_dir(data_root, dataset)
    attacker_path = dataset_dir / "attacker" / "queries.jsonl"
    if not attacker_path.exists():
        raise FileNotFoundError(f"Missing attacker queries: {attacker_path}")
    return attacker_path, dataset_dir.name


def generate_mask_patterns(
    num_patterns: int,
    max_content_length: int,
    mask_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_patterns < 2:
        raise ValueError("--mask-patterns must be at least 2")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("--mask-ratio must be between 0 and 1")
    if max_content_length < 1:
        raise ValueError("--max-length leaves no content tokens after special tokens")

    mask_count = max(1, int(round(max_content_length * mask_ratio)))
    patterns = np.zeros((num_patterns, max_content_length), dtype=np.bool_)
    seen: set[tuple[int, ...]] = set()
    for i in range(num_patterns):
        for _ in range(1000):
            positions = tuple(sorted(rng.choice(max_content_length, size=mask_count, replace=False).tolist()))
            if positions not in seen:
                seen.add(positions)
                patterns[i, list(positions)] = True
                break
        else:
            positions = tuple(sorted(rng.choice(max_content_length, size=mask_count, replace=False).tolist()))
            patterns[i, list(positions)] = True
    return patterns


class DateCollator:
    """Tokenize texts and apply one fixed mask pattern per sample."""

    def __init__(
        self,
        tokenizer,
        mask_patterns: np.ndarray,
        max_length: int,
        rng: np.random.Generator,
    ):
        self.tokenizer = tokenizer
        self.mask_patterns = mask_patterns
        self.max_length = max_length
        self.rng = rng

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = torch.full(input_ids.shape, -100, dtype=torch.long)
        masked_input_ids = input_ids.clone()
        mask_positions = torch.zeros(input_ids.shape, dtype=torch.bool)
        pattern_ids = torch.as_tensor(
            self.rng.integers(0, len(self.mask_patterns), size=len(texts)), dtype=torch.long
        )

        special_masks = [
            self.tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
            for row in input_ids
        ]
        for row_idx, pattern_id in enumerate(pattern_ids.tolist()):
            valid_positions = [
                pos
                for pos, is_special in enumerate(special_masks[row_idx])
                if not is_special and int(attention_mask[row_idx, pos]) == 1
            ]
            if not valid_positions:
                continue
            pattern = self.mask_patterns[pattern_id]
            selected = [valid_positions[j] for j in range(min(len(valid_positions), len(pattern))) if pattern[j]]
            if not selected:
                selected = [int(self.rng.choice(valid_positions))]
            labels[row_idx, selected] = input_ids[row_idx, selected]
            masked_input_ids[row_idx, selected] = self.tokenizer.mask_token_id
            mask_positions[row_idx, selected] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "masked_input_ids": masked_input_ids,
            "mlm_labels": labels,
            "mask_positions": mask_positions,
            "pattern_ids": pattern_ids,
        }


def iter_batches(items: list, batch_size: int, drop_last: bool) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        if batch:
            yield batch
