"""
Pretrain Dataloader for LLM pretraining from scratch.

Simple text-only dataloader that:
- Loads text from pretrain corpus (parquet files)
- Tokenizes with LLM tokenizer
- One document per sample (truncate/pad to seq_len)
- Returns input_ids and labels for causal LM loss
- Optionally mixes in ECG QA examples (with or without ECG signals)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from pathlib import Path

from ecg_bench.utils.gpu_setup import is_main
from ecg_bench.configs.constants import (
    HF_CACHE_DIR,
    ECG_RAW_TOKEN_PREFIX,
    ECG_RAW_NUM_BINS,
    ECG_RAW_MIN_VALUE,
    ECG_RAW_MAX_VALUE,
    ECG_RAW_LEAD_INDICES,
)
from ecg_bench.utils.file_manager import FileManager


def ecg_signal_to_token_string(ecg_signal: np.ndarray, max_samples: int = 500) -> str:
    """
    Convert ECG signal to a string of discrete tokens.

    Args:
        ecg_signal: ECG signal array of shape (num_samples, num_leads) or (num_leads, num_samples)
        max_samples: Maximum number of samples per lead (default 500)

    Returns:
        String of ECG tokens like "ecg_123 ecg_456 ecg_234 ..."
    """
    # Ensure shape is (num_samples, num_leads)
    if ecg_signal.shape[0] == 12:  # leads first
        ecg_signal = ecg_signal.T

    # Truncate to max_samples
    ecg_signal = ecg_signal[:max_samples, :]

    # Select only the 2 leads: II, V4
    selected_leads = ecg_signal[:, ECG_RAW_LEAD_INDICES]  # (max_samples, 2)

    # Clamp values to [-3, 3] mV
    clamped = np.clip(selected_leads, ECG_RAW_MIN_VALUE, ECG_RAW_MAX_VALUE)

    # Bin values into discrete tokens (0 to NUM_BINS-1)
    bin_width = (ECG_RAW_MAX_VALUE - ECG_RAW_MIN_VALUE) / ECG_RAW_NUM_BINS
    bin_indices = ((clamped - ECG_RAW_MIN_VALUE) / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, ECG_RAW_NUM_BINS - 1)

    # Convert to token strings, concatenating leads with space
    token_strings = []
    num_leads = bin_indices.shape[1]
    for lead_idx in range(num_leads):
        lead_tokens = [f"{ECG_RAW_TOKEN_PREFIX}{idx}" for idx in bin_indices[:, lead_idx]]
        token_strings.extend(lead_tokens)
        if lead_idx < num_leads - 1:
            token_strings.append(" ")  # Space between leads

    return " ".join(token_strings)


class PretrainDataset(Dataset):
    """
    Dataset for pretraining on raw text.

    Each sample is a single document, tokenized and truncated/padded to seq_len.
    Labels are the same as input_ids (for causal LM, the loss will be computed
    on predicting the next token).
    """

    def __init__(self, data, llm_tokenizer, args):
        self.data = data
        self.llm_tokenizer = llm_tokenizer
        self.seq_len = args.seq_len
        self.args = args
        self.ecg_raw = getattr(args, 'ecg_raw', False)

        # Get special token IDs
        self.pad_token_id = llm_tokenizer.pad_token_id
        self.eos_token_id = llm_tokenizer.eos_token_id

        # File manager for loading ECG signals
        if self.ecg_raw:
            self.fm = FileManager()

        if is_main() and args.dev:
            print(f"PretrainDataset initialized with {len(data)} samples")
            print(f"  seq_len: {self.seq_len}")
            print(f"  pad_token_id: {self.pad_token_id}")
            print(f"  eos_token_id: {self.eos_token_id}")
            print(f"  ecg_raw: {self.ecg_raw}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]

        # If ecg_raw is enabled and this item has an ECG path, prepend ECG tokens
        if self.ecg_raw and "ecg_path" in item and item["ecg_path"]:
            ecg_path = item["ecg_path"].replace("./data", "./ecg_bench/data")
            try:
                ecg_signal = self.fm.open_npy(ecg_path)["ecg"]
                ecg_token_str = ecg_signal_to_token_string(ecg_signal)
                text = f"ECG Signal:\n{ecg_token_str}\n\n{text}"
            except Exception as e:
                if self.args.dev and is_main():
                    print(f"Warning: Could not load ECG from {ecg_path}: {e}")

        # Tokenize the text
        tokens = self.llm_tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.seq_len,
        )

        # Pad if necessary
        if len(tokens) < self.seq_len:
            padding_len = self.seq_len - len(tokens)
            # Left padding (consistent with other dataloaders in the codebase)
            tokens = [self.pad_token_id] * padding_len + tokens

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [0 if t == self.pad_token_id else 1 for t in tokens]

        # Labels: same as input_ids, but -100 for padding
        labels = [t if t != self.pad_token_id else -100 for t in tokens]

        if self.args.dev and is_main() and index == 0:
            self._debug_print(tokens, attention_mask, labels, text)

        return {
            "elm_input_ids": torch.tensor(tokens, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
        }

    def _debug_print(self, tokens, attention_mask, labels, text):
        """Print debug information for the first sample."""
        print("\n=== PretrainDataset Debug ===")
        print(f"Text (first 200 chars): {text[:200]}...")
        print(f"Token count: {len(tokens)}")
        print(f"Non-padding tokens: {sum(attention_mask)}")
        print(f"First 10 tokens: {tokens[:10]}")
        print(f"Last 10 tokens: {tokens[-10:]}")
        print(f"First 10 labels: {labels[:10]}")
        decoded = self.llm_tokenizer.decode(tokens, skip_special_tokens=False)
        print(f"Decoded (first 200 chars): {decoded[:200]}...")
        print("=" * 50)


def load_pretrain_corpus(corpus_path: str):
    """Load pretrain corpus from parquet file(s)."""
    corpus_path = Path(corpus_path)

    if corpus_path.is_file() and corpus_path.suffix == ".parquet":
        # Single parquet file
        dataset = load_dataset("parquet", data_files=str(corpus_path), split="train")
    elif corpus_path.is_dir():
        # Directory with parquet files
        parquet_files = list(corpus_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {corpus_path}")
        dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
    else:
        raise ValueError(f"Invalid corpus path: {corpus_path}")

    return dataset


def load_ecg_data(ecg_data_name: str, num_examples: int, seed: int = 42, ecg_raw: bool = False):
    """
    Load ECG QA dataset and convert to plain text format for pretraining.

    Args:
        ecg_data_name: Name of ECG dataset (e.g., "ecg-qa-ptbxl-250-1250")
        num_examples: Number of examples to sample
        seed: Random seed for sampling
        ecg_raw: If True, keep ecg_path column for signal loading

    Returns:
        HuggingFace Dataset with "text" field containing Q&A as plain text
        (and optionally "ecg_path" if ecg_raw=True)
    """
    # Load the ECG dataset
    dataset = load_dataset(
        f"willxxy/{ecg_data_name}",
        split="fold1_train",
        cache_dir=HF_CACHE_DIR,
    )

    # Sample if needed
    if num_examples is not None and num_examples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_examples))

    # Columns to keep
    keep_columns = {"text"}
    if ecg_raw and "ecg_path" in dataset.column_names:
        keep_columns.add("ecg_path")

    # Convert Q&A format to plain text
    def format_ecg_text(example):
        text_data = example["text"]

        # Parse JSON if needed
        if isinstance(text_data, str):
            try:
                text_data = json.loads(text_data)
            except json.JSONDecodeError:
                # Already plain text
                result = {"text": text_data, "source": "ecg"}
                if ecg_raw and "ecg_path" in example:
                    result["ecg_path"] = example["ecg_path"]
                return result

        # Format as Q&A text
        if isinstance(text_data, list):
            # Check if it's ECG-QA format: ["task_type", "question", ["answer"]]
            if len(text_data) == 3 and isinstance(text_data[0], str) and isinstance(text_data[1], str):
                task_type = text_data[0]
                question = text_data[1]
                answer = text_data[2]
                if isinstance(answer, list):
                    answer = ", ".join(str(a) for a in answer)
                text = f"Question: {question}\nAnswer: {answer}"
            # Check if it's chat format: [{"from": "human", "value": "..."}, ...]
            elif len(text_data) > 0 and isinstance(text_data[0], dict):
                formatted_parts = []
                for turn in text_data:
                    role = turn.get("from", "unknown")
                    value = turn.get("value", "")
                    if role.lower() in ["human", "user"]:
                        formatted_parts.append(f"Question: {value}")
                    else:
                        formatted_parts.append(f"Answer: {value}")
                text = "\n".join(formatted_parts)
            else:
                # Unknown list format
                text = str(text_data)
        elif isinstance(text_data, dict):
            # Handle other dict formats
            text = str(text_data)
        else:
            text = str(text_data)

        result = {"text": text, "source": "ecg"}
        if ecg_raw and "ecg_path" in example:
            result["ecg_path"] = example["ecg_path"]
        return result

    # Apply formatting, keeping necessary columns
    remove_cols = [c for c in dataset.column_names if c not in keep_columns]
    dataset = dataset.map(format_ecg_text, remove_columns=remove_cols)

    return dataset


def mix_datasets(pretrain_corpus, ecg_data, seed: int = 42):
    """
    Mix pretrain corpus with ECG data.

    Args:
        pretrain_corpus: Main pretrain corpus dataset
        ecg_data: ECG QA dataset (already formatted)
        seed: Random seed for shuffling

    Returns:
        Combined and shuffled dataset
    """
    # Ensure both have "text" column
    if "source" not in pretrain_corpus.column_names:
        pretrain_corpus = pretrain_corpus.map(lambda x: {"source": "pretrain", **x})

    # Concatenate and shuffle
    combined = concatenate_datasets([pretrain_corpus, ecg_data])
    combined = combined.shuffle(seed=seed)

    return combined
