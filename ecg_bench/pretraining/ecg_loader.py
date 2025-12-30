"""
ECG loader for pretraining.

Loads real ECG signals from MIMIC QA dataset for use in Task 6.
The preprocessed .npy files contain both ECG signals and diagnostic reports.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datasets import load_dataset
import random

from ecg_bench.configs.constants import HF_CACHE_DIR


class PretrainECGLoader:
    """
    Loads ECG signals and diagnoses from MIMIC QA dataset for pretraining tasks.

    Uses the MIMIC QA train split to get valid ECG paths, then loads the
    preprocessed .npy files which contain both the ECG signal and the
    machine-generated diagnostic report.

    Supports partitioning ECGs between tasks (e.g., Task 6 vs Task 8) to avoid
    data leakage.
    """

    def __init__(
        self,
        dataset_name: str = "ecg-qa-mimic-iv-ecg-250-1250",
        fold: str = "1",
        cache_ecgs: bool = True,
        max_cache_size: int = 10000,
    ):
        """
        Initialize the ECG loader.

        Args:
            dataset_name: Name of the dataset to load from HuggingFace
            fold: Which fold to use (default "1")
            cache_ecgs: Whether to cache loaded ECG data in memory
            max_cache_size: Maximum number of ECGs to cache
        """
        self.dataset_name = dataset_name
        self.fold = fold
        self.cache_ecgs = cache_ecgs
        self.max_cache_size = max_cache_size

        # Cache stores full data dict (ecg + report)
        self._ecg_cache: Dict[str, Dict[str, Any]] = {}
        # Use a set for unique paths, then convert to list
        self._unique_ecg_paths: Set[str] = set()
        self._ecg_paths: List[str] = []

        # Partition tracking - split ECGs between different tasks
        self._task6_paths: List[str] = []  # First half for Task 6
        self._task8_paths: List[str] = []  # Second half for Task 8

        self._load_dataset()

    def _load_dataset(self):
        """Load the dataset and extract unique ECG paths."""
        print(f"Loading MIMIC QA dataset: {self.dataset_name} (fold {self.fold})...")

        dataset = load_dataset(
            f"willxxy/{self.dataset_name}",
            split=f"fold{self.fold}_train",
            cache_dir=HF_CACHE_DIR,
        )

        # Extract unique ECG paths (many QA samples may share the same ECG)
        for item in dataset:
            ecg_path = item["ecg_path"].replace("./data", "./ecg_bench/data")
            self._unique_ecg_paths.add(ecg_path)

        self._ecg_paths = list(self._unique_ecg_paths)

        # Partition paths: first half for Task 6, second half for Task 8
        # Use deterministic split based on sorted paths
        sorted_paths = sorted(self._ecg_paths)
        midpoint = len(sorted_paths) // 2
        self._task6_paths = sorted_paths[:midpoint]
        self._task8_paths = sorted_paths[midpoint:]

        print(f"Loaded {len(self._ecg_paths)} unique ECG paths from dataset")
        print(f"  Task 6 partition: {len(self._task6_paths)} ECGs")
        print(f"  Task 8 partition: {len(self._task8_paths)} ECGs")

    def _load_ecg_data(self, ecg_path: str) -> Optional[Dict[str, Any]]:
        """
        Load full ECG data from disk including signal and report.

        Returns:
            Dict with 'ecg' (numpy array) and 'report' (str diagnosis), or None if failed
        """
        try:
            data = np.load(ecg_path, allow_pickle=True).item()
            return {
                "ecg": data["ecg"],
                "report": data.get("report", ""),
                "path": data.get("path", ""),
            }
        except Exception as e:
            print(f"Error loading ECG from {ecg_path}: {e}")
            return None

    def _get_ecg_from_path(self, ecg_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Helper to get ECG data from a specific path."""
        # Check cache first
        if self.cache_ecgs and ecg_path in self._ecg_cache:
            ecg_data = self._ecg_cache[ecg_path]
        else:
            ecg_data = self._load_ecg_data(ecg_path)

            if ecg_data is not None and self.cache_ecgs:
                # Add to cache if under limit
                if len(self._ecg_cache) < self.max_cache_size:
                    self._ecg_cache[ecg_path] = ecg_data

        if ecg_data is None:
            return None, None

        metadata = {
            "ecg_path": ecg_path,
            "source": "mimic-iv-ecg",
            "report": ecg_data.get("report", ""),
            "original_path": ecg_data.get("path", ""),
        }

        return ecg_data["ecg"], metadata

    def get_random_ecg(self, partition: str = "all") -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get a random ECG signal and its diagnosis from the dataset.

        Args:
            partition: Which partition to sample from:
                       "all" - any ECG (default, backward compatible)
                       "task6" - first half, for Task 6 generation
                       "task8" - second half, for Task 8 long-range prediction

        Returns:
            Tuple of (ecg_signal, metadata) or (None, None) if failed
            metadata includes 'report' with the machine-generated diagnosis
        """
        if partition == "task6":
            paths = self._task6_paths
        elif partition == "task8":
            paths = self._task8_paths
        else:
            paths = self._ecg_paths

        if not paths:
            return None, None

        idx = random.randint(0, len(paths) - 1)
        ecg_path = paths[idx]

        return self._get_ecg_from_path(ecg_path)

    def get_ecg_by_index(self, idx: int) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get a specific ECG signal by index.

        Args:
            idx: Index of the ECG to retrieve

        Returns:
            Tuple of (ecg_signal, metadata) or (None, None) if failed
        """
        if idx < 0 or idx >= len(self._ecg_paths):
            return None, None

        ecg_path = self._ecg_paths[idx]

        # Check cache first
        if self.cache_ecgs and ecg_path in self._ecg_cache:
            ecg_data = self._ecg_cache[ecg_path]
        else:
            ecg_data = self._load_ecg_data(ecg_path)

            if ecg_data is not None and self.cache_ecgs:
                if len(self._ecg_cache) < self.max_cache_size:
                    self._ecg_cache[ecg_path] = ecg_data

        if ecg_data is None:
            return None, None

        metadata = {
            "ecg_path": ecg_path,
            "source": "mimic-iv-ecg",
            "report": ecg_data.get("report", ""),
            "original_path": ecg_data.get("path", ""),
        }

        return ecg_data["ecg"], metadata

    def __len__(self) -> int:
        """Return the number of ECGs available."""
        return len(self._ecg_paths)

    def clear_cache(self):
        """Clear the ECG cache."""
        self._ecg_cache.clear()


# Global loader instance (initialized lazily)
_ecg_loader: Optional[PretrainECGLoader] = None


def get_ecg_loader(
    dataset_name: str = "ecg-qa-mimic-iv-ecg-250-1250",
    fold: str = "1",
    reinitialize: bool = False,
) -> PretrainECGLoader:
    """
    Get the global ECG loader instance.

    Args:
        dataset_name: Dataset to load from
        fold: Which fold to use
        reinitialize: If True, create a new loader even if one exists

    Returns:
        PretrainECGLoader instance
    """
    global _ecg_loader

    if _ecg_loader is None or reinitialize:
        _ecg_loader = PretrainECGLoader(dataset_name=dataset_name, fold=fold)

    return _ecg_loader
