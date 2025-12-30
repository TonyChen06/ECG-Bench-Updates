import argparse
from ecg_bench.configs.constants import Mode


def get_args(mode: Mode) -> argparse.Namespace:
    if mode not in {"train", "eval", "inference", "post_train", "ecg_tokenizer", "preprocess", "rag", "signal2vec", "pretrain"}:
        raise ValueError(f"invalid mode: {mode}")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--dev", action="store_true", default=None, help="Development mode")
    if mode in {"train", "eval", "inference", "post_train", "ecg_tokenizer", "signal2vec"}:
        parser.add_argument("--ecg_tokenizer", type=str, default=None, help="Path to ECG Tokenizer")
        parser.add_argument("--signal2vec_embeddings", type=str, default=None, help="Path to Signal2Vec embeddings file")
    if mode in {"train", "eval", "inference", "post_train", "preprocess"}:
        parser.add_argument("--segment_len", type=int, default=1250, help="ECG Segment Length")
    if mode in {"train", "eval", "inference", "post_train"}:
        for flag, help_text in [
            ("--ecg_image", "Plot ECG Signal as Image"),
            ("--ecg_signal", "Raw ECG Signal"),
            ("--ecg_stacked_signal", "Stacked ECG Signal"),
            ("--ecg_token", "ECG Tokens"),
            ("--ecg_raw", "Raw ECG as discrete tokens (4 leads: II, aVR, V1, V4)"),
            ("--augment_ecg_image", "Augment ECG Image"),
            ("--noise_ecg", "Apply ECG Perturbation"),
            ("--blackout_ecg", "Apply ECG Blackout"),
            ("--no_signal", "No signal, text only"),
        ]:
            parser.add_argument(flag, action="store_true", default=None, help=help_text)

        parser.add_argument("--data", type=str, default=None, help="ID of the training/eval/inference/post-train data from huggingface datasets")
        parser.add_argument("--data_subset", type=float, default=None, help="Subset of data to use (between 0 and 1)")
        parser.add_argument("--encoder", type=str, default=None, help="Neural Network Encoder Model")
        parser.add_argument("--llm", type=str, default=None, help="Large Language Model")
        parser.add_argument("--peft", action="store_true", default=None, help="Use PEFT")
        parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
        parser.add_argument("--encoder_ckpt", type=str, default=None, help="Path to the encoder checkpoint")
        parser.add_argument("--elm_ckpt", type=str, default=None, help="Path to the LLM checkpoint")
        parser.add_argument("--attention_type", type=str, default="sdpa", help="Attention Type")
        parser.add_argument("--num_encoder_tokens", type=int, default=1, help="Number of encoder tokens")
        parser.add_argument("--update_encoder", action="store_true", default=False, help="Update encoder")
        parser.add_argument("--output_hidden_states", action="store_true", default=False, help="Output hidden states")
        parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
        parser.add_argument("--plat_rep_type", type=str, default="separate", choices=["separate", "combined"], help="Platonic Representation Type")

        parser.add_argument("--system_prompt", type=str, default=None, help="Path to System Prompt")
        parser.add_argument("--fold", type=str, default="1", help="Data Fold Number")

        parser.add_argument("--rag", action="store_true", default=None, help="Use RAG")
        parser.add_argument("--rag_k", type=int, default=1, help="RAG k")
        parser.add_argument("--rag_database", type=str, default=None, help="Path to RAG Database Containing Metadata and Indexes")
        parser.add_argument("--rag_query", type=str, default=None, choices=["ecg_signal", "ecg_feature"], help="Rag Query Type")
        parser.add_argument("--rag_location", type=str, default="system_prompt", choices=["system_prompt", "user_query"], help="RAG Location")
        parser.add_argument("--rag_content", type=str, default=None, choices=["ecg_feature", "diagnostic_report"], help="RAG Content")

        parser.add_argument("--wandb", action="store_true", default=None, help="Enable logging")

        parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
        parser.add_argument("--distributed", action="store_true", default=None, help="Enable distributed training")

    if mode == "train":
        parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer type")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
        parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
        parser.add_argument("--patience_delta", type=float, default=0.1, help="Delta for early stopping")
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
        parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for optimizer")
        parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer")
        parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
        parser.add_argument("--embed_skip_warmup", action="store_true", default=False, help="Skip warmup for embedding layers (use peak LR from start)")
        parser.add_argument("--ref_global_bs", type=int, default=None)
        parser.add_argument("--grad_accum_steps", type=int, default=1)
        parser.add_argument("--scale_wd", type=str, default="none", choices=["none", "inv_sqrt", "inv_linear"])
        parser.add_argument("--llm_input_len", type=int, default=1024, help="LLM Input Sequence Length")
        parser.add_argument("--min_ecg_tokens_len", type=int, default=250, help="Minimum ECG token length to consider")

    if mode == "post_train":
        parser.add_argument("--dpo_beta", type=float, default=0.5, help="DPO beta")

    if mode in {"ecg_tokenizer", "preprocess", "signal2vec"}:
        parser.add_argument("--num_cores", type=int, default=12, help="Number of cores for parallel processing")
        parser.add_argument("--sampled_file", type=str, default=None, help="Path to the sampled ECG files for tokenizer training")

    if mode == "signal2vec":
        parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for SkipGram")
        parser.add_argument("--window_size", type=int, default=12, help="Window size for SkipGram")
        parser.add_argument("--neg_alpha", type=float, default=0.75, help="Negative sampling alpha for SkipGram")
        parser.add_argument("--subsample_t", type=float, default=1e-5, help="Subsampling threshold for SkipGram")
        parser.add_argument("--min_count", type=int, default=10, help="Minimum count for SkipGram")
        parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip for SkipGram")
        parser.add_argument("--renorm_every", type=int, default=0, help="Renormalization every for SkipGram")
        parser.add_argument("--neg_k", type=int, default=5, help="Number of negative samples for SkipGram")
        parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
        parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    if mode == "ecg_tokenizer":
        parser.add_argument("--num_merges", type=int, default=3500, help="Number of merges for BPE")
        parser.add_argument("--num_samples", type=int, default=300000, help="Number of samples for training the tokenizer")
        parser.add_argument("--path_to_ecg_npy", type=str, default=None, help="Path to the ECG npy files")

    if mode == "preprocess":
        parser.add_argument("--preprocess", action="store_true", default=False, help="Preprocess files")
        parser.add_argument("--base_data", type=str, default=None, help="Base dataset to preprocess")
        parser.add_argument("--map_data", type=str, default=None, help="External dataset to map to base dataset")
        parser.add_argument("--toy", action="store_true", default=None, help="Create a toy dataset")
        parser.add_argument("--mix_data", type=str, default=None, help="Mix data: comma-separated list of JSON filenames")
        parser.add_argument("--target_sf", type=int, default=250, help="Target sampling frequency")

    if mode == "rag":
        parser.add_argument("--rag_data", type=str, default=None, help="Path to the data for RAG database creation")

    if mode == "pretrain":
        # LLM and training arguments
        parser.add_argument("--llm", type=str, required=True, help="Large Language Model")
        parser.add_argument("--peft", action="store_true", default=None, help="Use PEFT")
        parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
        parser.add_argument("--elm_ckpt", type=str, default=None, help="Path to the LLM checkpoint")
        parser.add_argument("--attention_type", type=str, default="sdpa", help="Attention Type")
        parser.add_argument("--output_hidden_states", action="store_true", default=False, help="Output hidden states")
        parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
        parser.add_argument("--system_prompt", type=str, default=None, help="Path to System Prompt")

        # Pretraining data
        parser.add_argument("--pretrain_data", type=str, required=True, help="Path to pretraining data directory")

        # Training hyperparameters
        parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
        parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
        parser.add_argument("--patience_delta", type=float, default=0.1, help="Delta for early stopping")
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
        parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for optimizer")
        parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer")
        parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
        parser.add_argument("--embed_skip_warmup", action="store_true", default=False, help="Skip warmup for embedding layers (use peak LR from start)")
        parser.add_argument("--ref_global_bs", type=int, default=None)
        parser.add_argument("--grad_accum_steps", type=int, default=1)
        parser.add_argument("--scale_wd", type=str, default="none", choices=["none", "inv_sqrt", "inv_linear"])
        parser.add_argument("--llm_input_len", type=int, default=2048, help="LLM Input Sequence Length")

        # Distributed and logging
        parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
        parser.add_argument("--distributed", action="store_true", default=None, help="Enable distributed training")
        parser.add_argument("--wandb", action="store_true", default=None, help="Enable logging")

        # Scheduled sampling parameters (horizon-based curriculum)
        parser.add_argument("--ss_horizon_start", type=int, default=1, help="Starting prediction horizon (1=next token prediction)")
        parser.add_argument("--ss_horizon_end", type=int, default=10, help="Ending prediction horizon (e.g., 10=predict 10 tokens ahead)")

        # Gradient clipping
        parser.add_argument("--max_grad_norm", type=float, default=0, help="Max gradient norm for clipping (0 to disable)")

        # Curriculum learning - preserve task order during training
        parser.add_argument("--curriculum", action="store_true", default=True, help="Preserve curriculum order (no shuffling)")
        parser.add_argument("--no_curriculum", action="store_true", help="Disable curriculum (enable shuffling)")

        # Stage-based training (for detachable stages)
        parser.add_argument("--stage", type=int, default=None, help="Train only on a specific stage (1-9). If not set, trains on all stages.")
        parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from (loads model and tokenizer)")

        # ECG embedding warmup: Stage 1 only trains new ECG token embeddings
        parser.add_argument("--no_ecg_warmup", action="store_true", default=False, help="Skip ECG-embedding-only warmup (train full model from start)")

    return parser.parse_args()
