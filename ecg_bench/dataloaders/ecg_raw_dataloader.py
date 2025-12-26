import numpy as np
import torch

from ecg_bench.dataloaders.base_dataloader import BaseECGDataset
from ecg_bench.utils.gpu_setup import is_main
from ecg_bench.configs.constants import (
    ECG_RAW_TOKEN_PREFIX,
    ECG_RAW_NUM_BINS,
    ECG_RAW_MIN_VALUE,
    ECG_RAW_MAX_VALUE,
    ECG_RAW_LEAD_INDICES,
)


class ECGRawDataset(BaseECGDataset):
    """
    ECG Raw Dataset that tokenizes ECG signals into discrete tokens.

    Takes 4 leads (II, aVR, V1, V4), clamps values to [-3, 3] mV,
    and bins into 600 equally spaced tokens. Leads are concatenated
    and tokens are directly inserted into the LLM input sequence.
    """

    def __init__(self, data, mode, llm_tokenizer_components, args):
        super().__init__(data, mode, args)
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"]
        self.build_ecg_raw_vocab()

    def build_ecg_raw_vocab(self):
        """Add ECG raw tokens to the tokenizer vocabulary."""
        new_vocab = [f"{ECG_RAW_TOKEN_PREFIX}{i}" for i in range(ECG_RAW_NUM_BINS)]
        if self.args.dev and is_main():
            print(f"Adding {len(new_vocab)} ECG raw tokens to vocabulary")
        self.llm_tokenizer.add_tokens(new_vocab)
        self.ecg_token_ids = set(self.llm_tokenizer.convert_tokens_to_ids(new_vocab))

    def ecg_to_tokens(self, ecg_signal: np.ndarray) -> list[int]:
        """
        Convert ECG signal to discrete token IDs.

        Args:
            ecg_signal: ECG signal array of shape (num_samples, num_leads) or (num_leads, num_samples)

        Returns:
            List of token IDs representing the binned ECG values
        """
        # Ensure shape is (num_samples, num_leads)
        if ecg_signal.shape[0] == 12:  # leads first
            ecg_signal = ecg_signal.T

        # Select only the 4 leads we care about: II, aVR, V1, V4
        selected_leads = ecg_signal[:, ECG_RAW_LEAD_INDICES]  # (num_samples, 4)

        # Clamp values to [-3, 3] mV
        clamped = np.clip(selected_leads, ECG_RAW_MIN_VALUE, ECG_RAW_MAX_VALUE)

        # Bin values into 600 discrete tokens (0-599)
        # Map [-3, 3] -> [0, 599]
        bin_width = (ECG_RAW_MAX_VALUE - ECG_RAW_MIN_VALUE) / ECG_RAW_NUM_BINS
        bin_indices = ((clamped - ECG_RAW_MIN_VALUE) / bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, ECG_RAW_NUM_BINS - 1)

        # Concatenate leads: flatten in lead-major order (all of lead1, then all of lead2, etc.)
        # This matches how other dataloaders concatenate leads
        token_indices = bin_indices.T.flatten()  # (4 * num_samples,)

        # Convert bin indices to token IDs
        token_strings = [f"{ECG_RAW_TOKEN_PREFIX}{idx}" for idx in token_indices]
        token_ids = self.llm_tokenizer.convert_tokens_to_ids(token_strings)

        return token_ids

    def __getitem__(self, index):
        instance = self.data[index]
        ecg_path = instance["ecg_path"].replace("./data", "./ecg_bench/data")
        ecg_signal = self.fm.open_npy(ecg_path)["ecg"]

        # Perturbations
        if self.args.noise_ecg:
            ecg_signal = self.noise_ecg(ecg_signal)
        if self.args.blackout_ecg:
            ecg_signal = self.blackout_ecg(ecg_signal)

        # Convert ECG to tokens
        ecg_tokens = self.ecg_to_tokens(ecg_signal)

        # Prepare text inputs
        text = instance["text"]
        prompt = self.make_prompt(text)
        if self.args.dev and is_main():
            print("prompt\n", prompt)

        if self.mode == "train":
            return self.prepare_training_set(ecg_tokens, prompt)
        elif self.mode in ["eval", "inference"]:
            return self.prepare_eval_inference_set(ecg_tokens, prompt)

    def prepare_training_set(self, ecg_tokens: list[int], prompt: str):
        truncated_padded_input = self.trunc_pad_input(ecg_tokens, prompt)
        attention_mask = self.create_attention_mask(truncated_padded_input)
        labels = self.create_labels(truncated_padded_input)
        signal_id_indices = self.find_ecg_token_indices(truncated_padded_input)

        if self.args.dev and is_main():
            self.decode_and_print_mapping(truncated_padded_input)
            self.check_labels(labels)
            self.check_attention_mask(truncated_padded_input, attention_mask)

        assert len(truncated_padded_input) == len(attention_mask) == len(labels) == self.args.llm_input_len, (
            f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)} != {len(labels)} != {self.args.llm_input_len}"
        )

        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "signal_id_indices": torch.tensor(signal_id_indices, dtype=torch.int64),
        }

    def prepare_eval_inference_set(self, ecg_tokens: list[int], prompt: str):
        truncated_padded_input = self.trunc_pad_input(ecg_tokens, prompt)
        attention_mask = self.create_attention_mask(truncated_padded_input)
        signal_id_indices = self.find_ecg_token_indices(truncated_padded_input)

        assert len(truncated_padded_input) == len(attention_mask), (
            f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)}"
        )

        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "signal_id_indices": torch.tensor(signal_id_indices, dtype=torch.int64),
        }

    def trunc_pad_input(self, ecg_tokens: list[int], prompt: str) -> list[int]:
        """Insert ECG tokens into prompt and handle truncation/padding."""
        before, after = self.split_prompt(prompt)

        if self.mode in ["eval", "inference"]:
            return before + ecg_tokens + after

        before_len, after_len, ecg_len = len(before), len(after), len(ecg_tokens)
        total_len = before_len + after_len + ecg_len

        if total_len == self.args.llm_input_len:
            return before + ecg_tokens + after
        elif total_len < self.args.llm_input_len:
            return self.pad_input(before + ecg_tokens + after)

        # Need to truncate - prioritize keeping ECG tokens
        min_ecg_len = getattr(self.args, 'min_ecg_tokens_len', 250)

        if before_len + min_ecg_len > self.args.llm_input_len:
            raise ValueError("before + min_ecg exceeds llm_input_len")

        # Calculate how many ECG tokens we can keep
        target_ecg = min(ecg_len, max(min_ecg_len, self.args.llm_input_len - (before_len + after_len)))
        ecg_tokens = ecg_tokens[:target_ecg]

        # Truncate after portion if needed
        remaining_after = self.args.llm_input_len - before_len - len(ecg_tokens)
        after = after[:max(remaining_after, 0)]

        return before + ecg_tokens + after

    def find_ecg_token_indices(self, input_ids: list[int]) -> list[int]:
        """Find indices of ECG tokens in the input sequence."""
        ecg_indices = [i for i, tid in enumerate(input_ids) if tid in self.ecg_token_ids]
        if not ecg_indices:
            if self.args.dev and is_main():
                print("No ECG tokens found in input_ids.")
            return [-1]
        return ecg_indices
