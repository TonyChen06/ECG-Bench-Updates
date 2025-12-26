"""
Dataloader for pretraining tasks.

Loads pretraining data from JSONL files and prepares it for training.
This follows the same pattern as the ECG raw dataloader but for
synthetic pretraining tasks.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from pathlib import Path

from ecg_bench.configs.constants import (
    ECG_RAW_TOKEN_PREFIX,
    ECG_RAW_NUM_BINS,
    HF_LLMS,
    SIGNAL_TOKEN_PLACEHOLDER,
)
from ecg_bench.utils.chat_template import get_conv_template
from ecg_bench.utils.gpu_setup import is_main


class PretrainDataset(Dataset):
    """
    Dataset for pretraining tasks.

    Loads tasks from JSONL files and formats them for the LLM.
    """

    def __init__(
        self,
        data_path: str,
        mode: str,
        llm_tokenizer_components: Dict,
        args,
    ):
        self.data_path = Path(data_path)
        self.mode = mode
        self.args = args
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"]

        # Load data
        self.data = self._load_data()

        # Add ECG tokens to vocabulary
        self._add_ecg_tokens()

        # Set up chat template
        if self.args.llm:
            self.chat_template = self._make_chat_template()

    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        file_path = self.data_path / f"{self.mode.replace('eval', 'test')}.jsonl"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))

        if is_main():
            print(f"Loaded {len(data)} samples from {file_path}")

        return data

    def _add_ecg_tokens(self):
        """Add ECG raw tokens to the tokenizer vocabulary."""
        new_vocab = [f"{ECG_RAW_TOKEN_PREFIX}{i}" for i in range(ECG_RAW_NUM_BINS)]
        if self.args.dev and is_main():
            print(f"Adding {len(new_vocab)} ECG raw tokens to vocabulary")
        self.llm_tokenizer.add_tokens(new_vocab)

    def _make_chat_template(self):
        """Create chat template for the LLM."""
        chat_template = get_conv_template(HF_LLMS[self.args.llm]["chat_template"])
        if HF_LLMS[self.args.llm]["system_prompt"] and self.args.system_prompt:
            with open(self.args.system_prompt, encoding="utf-8") as f:
                system_prompt = f.read()
            chat_template.set_system_message(system_prompt)
        return chat_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        instance = self.data[index]
        text = instance["text"]

        prompt = self._make_prompt(text)

        if self.args.dev and is_main() and index == 0:
            print("Sample prompt:\n", prompt[:500], "...")

        if self.mode == "train":
            return self._prepare_training_set(prompt)
        else:
            return self._prepare_eval_set(prompt)

    def _make_prompt(self, text: List[Dict]) -> str:
        """Format text into a prompt using the chat template."""
        prompt = self.chat_template.copy()

        for turn in text:
            is_human = turn["from"].lower() in ["human", "user"]
            role = prompt.roles[0] if is_human else prompt.roles[1]
            prompt.append_message(role, turn["value"])

        return prompt.get_prompt()

    def _prepare_training_set(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepare a training sample."""
        truncated_padded_input = self._trunc_pad_input(prompt)
        attention_mask = self._create_attention_mask(truncated_padded_input)
        labels = self._create_labels(truncated_padded_input)

        assert len(truncated_padded_input) == len(attention_mask) == len(labels) == self.args.llm_input_len, (
            f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)} != {len(labels)} != {self.args.llm_input_len}"
        )

        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "signal_id_indices": torch.tensor([-1], dtype=torch.int64),  # No signal placeholder
        }

    def _prepare_eval_set(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepare an evaluation sample."""
        truncated_padded_input = self._trunc_pad_input(prompt)
        attention_mask = self._create_attention_mask(truncated_padded_input)

        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "signal_id_indices": torch.tensor([-1], dtype=torch.int64),
        }

    def _trunc_pad_input(self, prompt: str) -> List[int]:
        """Tokenize and truncate/pad the input."""
        prompt_tokens = self.llm_tokenizer.encode(prompt, add_special_tokens=False)

        if self.mode in ["eval", "inference"]:
            return prompt_tokens

        prompt_len = len(prompt_tokens)
        if prompt_len == self.args.llm_input_len:
            return prompt_tokens
        elif prompt_len < self.args.llm_input_len:
            return self._pad_input(prompt_tokens)

        # Truncate from the left (keep the end with assistant response)
        return prompt_tokens[-self.args.llm_input_len:]

    def _pad_input(self, tokens: List[int]) -> List[int]:
        """Left-pad tokens to llm_input_len."""
        padding_len = self.args.llm_input_len - len(tokens)
        return [self.llm_tokenizer.pad_token_id] * padding_len + tokens

    def _create_attention_mask(self, input_ids: List[int]) -> List[int]:
        """Create attention mask (1 for real tokens, 0 for padding)."""
        bos_token = next(iter(HF_LLMS[self.args.llm]["watch_tokens"]["bos_token"]))
        try:
            start_idx = input_ids.index(bos_token)
        except ValueError:
            start_idx = 0

        attention_mask = [0] * len(input_ids)
        attention_mask[start_idx:] = [1] * (len(input_ids) - start_idx)
        return attention_mask

    def _create_labels(self, input_ids: List[int]) -> List[int]:
        """Create labels for causal LM training (mask non-response tokens)."""
        wt = HF_LLMS[self.args.llm]["watch_tokens"]
        BOS = set(wt["bos_token"].keys() if isinstance(wt["bos_token"], dict) else wt["bos_token"])
        EOS = set(wt["eos_token"].keys() if isinstance(wt["eos_token"], dict) else wt["eos_token"])
        fe = wt.get("final_eos_token", ())
        FINAL_EOS = set(fe.keys() if isinstance(fe, dict) else fe)

        labels = [-100] * len(input_ids)
        i, L = 0, len(input_ids)
        seen_bos = False
        in_resp = False
        START = wt["response_start"]["order"]
        k = len(START)

        while i < L:
            tok = input_ids[i]
            if not seen_bos and tok in BOS:
                seen_bos = True
            if seen_bos and (not in_resp) and k > 0:
                if i + k <= L and input_ids[i:i + k] == START:
                    i += k
                    in_resp = True
                    continue
            if in_resp:
                labels[i] = tok
                if tok in EOS:
                    in_resp = False
            i += 1

        if L and input_ids[-1] in FINAL_EOS:
            labels[-1] = input_ids[-1]

        return labels


def build_pretrain_dataloader(
    data_path: str,
    mode: str,
    llm_tokenizer_components: Dict,
    args,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Build a dataloader for pretraining.

    Args:
        data_path: Path to the pretraining data directory
        mode: "train" or "eval"
        llm_tokenizer_components: Dict with "llm_tokenizer" key
        args: Training arguments
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        DataLoader
    """
    dataset = PretrainDataset(data_path, mode, llm_tokenizer_components, args)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
