"""
LLM Classification Tasks for SPSA Experiments

Implements the 6 benchmarks from MeZO paper:
- SST-2: Sentiment Analysis (binary)
- RTE: Recognizing Textual Entailment (binary)
- BoolQ: Boolean Question Answering (binary)
- COPA: Choice of Plausible Alternatives (binary)
- CB: CommitmentBank (3-way)
- WiC: Word-in-Context (binary)

Each task is formatted as a text-to-label classification problem.
Following MeZO paper: 1000/500/1000 train/val/test splits.
"""
import os
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset


# Task configuration constants
TASK_CONFIG = {
    "sst2": {
        "dataset_name": "glue",
        "dataset_config": "sst2",
        "text_columns": ["sentence"],
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["negative", "positive"],
        "prompt_template": "Review: {sentence}\nSentiment:",
        "train_size": 1000,
        "val_size": 500,
        "test_size": 1000,
    },
    "rte": {
        "dataset_name": "glue",
        "dataset_config": "rte",
        "text_columns": ["sentence1", "sentence2"],
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["yes", "no"],
        "prompt_template": "{sentence1}\nQuestion: Does this imply that \"{sentence2}\"? Answer:",
        "train_size": 1000,
        "val_size": 500,
        "test_size": 1000,
    },
    "boolq": {
        "dataset_name": "super_glue",
        "dataset_config": "boolq",
        "text_columns": ["question", "passage"],
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["no", "yes"],
        "prompt_template": "{passage}\nQuestion: {question}? Answer:",
        "train_size": 1000,
        "val_size": 500,
        "test_size": 1000,
    },
    "copa": {
        "dataset_name": "super_glue",
        "dataset_config": "copa",
        "text_columns": ["premise", "choice1", "choice2", "question"],
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["1", "2"],
        "prompt_template": "{premise}\nQuestion: What is the {question}?\nChoice 1: {choice1}\nChoice 2: {choice2}\nAnswer:",
        "train_size": 400,
        "val_size": 100,
        "test_size": 500,
    },
    "cb": {
        "dataset_name": "super_glue",
        "dataset_config": "cb",
        "text_columns": ["premise", "hypothesis"],
        "label_column": "label",
        "num_labels": 3,
        "label_names": ["yes", "no", "maybe"],
        "prompt_template": "{premise}\nQuestion: Does this imply that \"{hypothesis}\"? Answer:",
        "train_size": 250,
        "val_size": 57,
        "test_size": 250,
    },
    "wic": {
        "dataset_name": "super_glue",
        "dataset_config": "wic",
        "text_columns": ["sentence1", "sentence2", "word"],
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["no", "yes"],
        "prompt_template": "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nQuestion: Does the word \"{word}\" have the same meaning in both sentences? Answer:",
        "train_size": 1000,
        "val_size": 500,
        "test_size": 1000,
    },
}


class LLMTaskDataset:
    """
    Dataset wrapper for LLM classification tasks.
    Handles loading, formatting, and batching of examples.
    """

    def __init__(
        self,
        task_name: str,
        tokenizer: Any,
        max_length: int = 512,
        seed: int = 42,
        cache_dir: str = "./data_cache",
    ):
        self.task_name = task_name.lower()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.cache_dir = cache_dir

        if self.task_name not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(TASK_CONFIG.keys())}"
            )

        self.config = TASK_CONFIG[self.task_name]
        self.num_labels = self.config["num_labels"]
        self.label_names = self.config["label_names"]

        # Load and prepare datasets
        self._load_datasets()

        # Get label token ids for classification
        self._setup_label_tokens()

    def _load_datasets(self):
        """Load and split the dataset according to MeZO paper splits."""
        dataset_name = self.config["dataset_name"]
        dataset_config = self.config["dataset_config"]

        # Load from HuggingFace datasets
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            cache_dir=self.cache_dir,
        )

        # Get splits
        train_data = dataset.get("train", None)
        val_data = dataset.get("validation", None)
        test_data = dataset.get("test", None)

        # If no test split, use validation for test
        if test_data is None or len(test_data) == 0:
            test_data = val_data

        # Shuffle with seed
        random.seed(self.seed)

        # Sample according to MeZO paper sizes
        train_size = min(self.config["train_size"], len(train_data) if train_data else 0)
        val_size = min(self.config["val_size"], len(val_data) if val_data else 0)
        test_size = min(self.config["test_size"], len(test_data) if test_data else 0)

        # Shuffle and select
        if train_data:
            train_indices = list(range(len(train_data)))
            random.shuffle(train_indices)
            self.train_data = [train_data[i] for i in train_indices[:train_size]]
        else:
            self.train_data = []

        if val_data:
            val_indices = list(range(len(val_data)))
            random.shuffle(val_indices)
            self.val_data = [val_data[i] for i in val_indices[:val_size]]
        else:
            self.val_data = []

        if test_data:
            test_indices = list(range(len(test_data)))
            random.shuffle(test_indices)
            self.test_data = [test_data[i] for i in test_indices[:test_size]]
        else:
            self.test_data = []

        print(f"[{self.task_name}] Loaded: train={len(self.train_data)}, "
              f"val={len(self.val_data)}, test={len(self.test_data)}")

    def _setup_label_tokens(self):
        """Setup label token IDs for classification."""
        self.label_token_ids = []
        for label_name in self.label_names:
            # Get the first token of the label (with space prefix for OPT)
            tokens = self.tokenizer.encode(" " + label_name, add_special_tokens=False)
            if len(tokens) > 0:
                self.label_token_ids.append(tokens[0])
            else:
                # Fallback: use label name without space
                tokens = self.tokenizer.encode(label_name, add_special_tokens=False)
                self.label_token_ids.append(tokens[0] if tokens else 0)

        print(f"[{self.task_name}] Label tokens: {list(zip(self.label_names, self.label_token_ids))}")

    def format_example(self, example: Dict) -> str:
        """Format a single example using the task template."""
        template = self.config["prompt_template"]
        text_columns = self.config["text_columns"]

        # Build format dict
        format_dict = {}
        for col in text_columns:
            if col in example:
                format_dict[col] = example[col]
            else:
                format_dict[col] = ""

        return template.format(**format_dict)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Get a batch of examples.

        Returns:
            input_ids: [batch_size, seq_len] tokenized inputs
            attention_mask: [batch_size, seq_len] attention mask
            labels: [batch_size] integer labels
        """
        if split == "train":
            data = self.train_data
        elif split == "val" or split == "validation":
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")

        if len(data) == 0:
            raise ValueError(f"No data available for split: {split}")

        # Sample batch
        indices = random.sample(range(len(data)), min(batch_size, len(data)))
        batch_examples = [data[i] for i in indices]

        # Format examples
        texts = [self.format_example(ex) for ex in batch_examples]
        labels = [ex[self.config["label_column"]] for ex in batch_examples]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        return input_ids, attention_mask, labels_tensor

    def get_full_split(
        self,
        split: str,
        device: str = "cuda",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Get all examples from a split.
        Returns lists of tensors (one per example) to avoid OOM.
        """
        if split == "train":
            data = self.train_data
        elif split == "val" or split == "validation":
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for ex in data:
            text = self.format_example(ex)
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            all_input_ids.append(encoding["input_ids"].to(device))
            all_attention_masks.append(encoding["attention_mask"].to(device))
            all_labels.append(ex[self.config["label_column"]])

        return all_input_ids, all_attention_masks, all_labels


def compute_llm_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_token_ids: List[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute classification loss for LLM.

    Uses the logits at the LAST non-padding token position to predict the label.
    This is memory efficient - we only look at the final token's logits.

    Args:
        logits: [batch_size, seq_len, vocab_size] model output logits
        labels: [batch_size] integer labels (0, 1, or 2 for 3-way)
        label_token_ids: List of token IDs corresponding to each label
        attention_mask: [batch_size, seq_len] mask (1 for real tokens, 0 for padding)

    Returns:
        Cross-entropy loss scalar
    """
    batch_size = logits.size(0)
    device = logits.device

    # Find the last non-padding position for each example
    if attention_mask is not None:
        # Sum attention mask to get length, then subtract 1 for index
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = seq_lengths.clamp(min=0)
    else:
        # Assume no padding - use last position
        seq_lengths = torch.full((batch_size,), logits.size(1) - 1, device=device)

    # Extract logits at the last position for each example
    # Shape: [batch_size, vocab_size]
    last_logits = logits[torch.arange(batch_size, device=device), seq_lengths.long()]

    # Extract only the logits for our label tokens
    # Shape: [batch_size, num_labels]
    label_token_ids_tensor = torch.tensor(label_token_ids, device=device, dtype=torch.long)
    label_logits = last_logits[:, label_token_ids_tensor]

    # Cast to float32 for numerical stability in cross-entropy
    label_logits = label_logits.float()

    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(label_logits, labels)

    return loss


def compute_llm_classification_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_token_ids: List[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute classification accuracy for LLM.

    Args:
        logits: [batch_size, seq_len, vocab_size] model output logits
        labels: [batch_size] integer labels
        label_token_ids: List of token IDs corresponding to each label
        attention_mask: [batch_size, seq_len] mask

    Returns:
        Accuracy as a float between 0 and 1
    """
    batch_size = logits.size(0)
    device = logits.device

    # Find the last non-padding position
    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = seq_lengths.clamp(min=0)
    else:
        seq_lengths = torch.full((batch_size,), logits.size(1) - 1, device=device)

    # Extract logits at the last position
    last_logits = logits[torch.arange(batch_size, device=device), seq_lengths.long()]

    # Extract only the logits for our label tokens
    label_token_ids_tensor = torch.tensor(label_token_ids, device=device, dtype=torch.long)
    label_logits = last_logits[:, label_token_ids_tensor]

    # Get predictions
    predictions = label_logits.argmax(dim=-1)

    # Compute accuracy
    correct = (predictions == labels).float().sum().item()
    total = labels.size(0)

    return correct / total if total > 0 else 0.0


def get_task_dataset(
    task_name: str,
    tokenizer: Any,
    max_length: int = 512,
    seed: int = 42,
    cache_dir: str = "./data_cache",
) -> LLMTaskDataset:
    """
    Factory function to create a task dataset.

    Args:
        task_name: One of 'sst2', 'rte', 'boolq', 'copa', 'cb', 'wic'
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        seed: Random seed for reproducibility
        cache_dir: Directory to cache datasets

    Returns:
        LLMTaskDataset instance
    """
    return LLMTaskDataset(
        task_name=task_name,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
        cache_dir=cache_dir,
    )


def list_available_tasks() -> List[str]:
    """Return list of available task names."""
    return list(TASK_CONFIG.keys())


def get_task_info(task_name: str) -> Dict:
    """Return configuration info for a task."""
    if task_name.lower() not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_CONFIG[task_name.lower()].copy()


# Convenience function for quick testing
def test_task_loading():
    """Quick test of task loading functionality."""
    from transformers import AutoTokenizer

    print("Testing task loading...")

    # Load a small tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    for task_name in ["sst2", "rte", "boolq"]:
        print(f"\nTesting {task_name}...")
        dataset = get_task_dataset(task_name, tokenizer, max_length=128)

        # Get a batch
        input_ids, attention_mask, labels = dataset.get_batch("train", batch_size=2)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Label token IDs: {dataset.label_token_ids}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_task_loading()
