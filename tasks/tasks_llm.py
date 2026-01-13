"""
LLM Classification Tasks for SPSA Experiments

Implements the 6 benchmarks from MeZO paper:
- SST-2: Sentiment Analysis (binary)
- RTE: Recognizing Textual Entailment (binary)
- BoolQ: Boolean Question Answering (binary)
- WSC: Winograd Schema Challenge (binary)
- WiC: Word-in-Context (binary)
- SQuAD: Stanford Question Answering Dataset (extractive QA)

Each task is formatted as a text-to-label classification problem (except SQuAD which is generative).
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
        "task_type": "classification",
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
        "task_type": "classification",
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
        "task_type": "classification",
    },
    "wsc": {
        "dataset_name": "super_glue",
        "dataset_config": "wsc",
        "text_columns": ["text", "span1_text", "span2_text"],
        "label_column": "label",
        "num_labels": 2,
        "label_names": ["no", "yes"],
        "prompt_template": "{text}\nQuestion: In the above sentence, does \"{span2_text}\" refer to \"{span1_text}\"? Answer:",
        "train_size": 554,  # WSC has limited data
        "val_size": 104,
        "test_size": 146,
        "task_type": "classification",
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
        "task_type": "classification",
    },
    "squad": {
        "dataset_name": "squad",
        "dataset_config": None,
        "text_columns": ["context", "question"],
        "label_column": "answers",  # Special handling for SQuAD answers
        "num_labels": None,  # Generative task
        "label_names": None,  # Generative task
        "prompt_template": "Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        "train_size": 1000,
        "val_size": 500,
        "test_size": 1000,
        "task_type": "generative",
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
        self.task_type = self.config.get("task_type", "classification")

        # Load and prepare datasets
        self._load_datasets()

        # Get label token ids for classification (skip for generative tasks)
        if self.task_type == "classification":
            self._setup_label_tokens()
        else:
            self.label_token_ids = None

    def _load_datasets(self):
        """Load and split the dataset according to MeZO paper splits."""
        dataset_name = self.config["dataset_name"]
        dataset_config = self.config["dataset_config"]

        # Load from HuggingFace datasets
        if dataset_config is not None:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                cache_dir=self.cache_dir,
            )
        else:
            # For datasets without a config (like squad)
            dataset = load_dataset(
                dataset_name,
                cache_dir=self.cache_dir,
            )

        # Get splits
        train_data = dataset.get("train", None)
        val_data = dataset.get("validation", None)
        test_data = dataset.get("test", None)

        # Check if test labels are hidden (-1) - common in GLUE for leaderboard
        use_val_as_test = False
        if test_data is None or len(test_data) == 0:
            use_val_as_test = True
            print(f"[{self.task_name}] No test split available")
        elif test_data and len(test_data) > 0:
            label_col = self.config["label_column"]
            sample_labels = [test_data[i][label_col] for i in range(min(10, len(test_data)))]
            if all(l == -1 for l in sample_labels):
                use_val_as_test = True
                print(f"[{self.task_name}] Test labels hidden (-1)")

        # If test is unusable, split validation into val (first half) and test (second half)
        if use_val_as_test and val_data:
            val_list = list(val_data)
            random.seed(self.seed)
            random.shuffle(val_list)
            mid = len(val_list) // 2
            # Use first half for val, second half for test
            test_data = val_list[mid:]
            val_data = val_list[:mid]
            print(f"[{self.task_name}] Split validation into val={len(val_data)}, test={len(test_data)}")

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

    def get_answer(self, example: Dict) -> str:
        """Get the answer for an example (for generative tasks like SQuAD)."""
        if self.task_type == "generative" and self.task_name == "squad":
            answers = example.get("answers", {})
            answer_texts = answers.get("text", [])
            if answer_texts:
                return answer_texts[0]  # Return first answer
            return ""
        else:
            # For classification, return the label name
            label_idx = example[self.config["label_column"]]
            if self.label_names and 0 <= label_idx < len(self.label_names):
                return self.label_names[label_idx]
            return str(label_idx)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Get a batch of examples.

        Returns:
            input_ids: [batch_size, seq_len] tokenized inputs
            attention_mask: [batch_size, seq_len] attention mask
            labels: [batch_size] integer labels for classification, or List[str] answers for generative
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

        if self.task_type == "classification":
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
        else:
            # Generative task (e.g., SQuAD)
            answers = [self.get_answer(ex) for ex in batch_examples]

            # For training, we concatenate prompt + answer and use LM loss
            texts_with_answers = [f"{t} {a}" for t, a in zip(texts, answers)]

            # Tokenize prompt only (for generation evaluation)
            prompt_encodings = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Tokenize full sequence (prompt + answer) for training loss
            full_encodings = self.tokenizer(
                texts_with_answers,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = full_encodings["input_ids"].to(device)
            attention_mask = full_encodings["attention_mask"].to(device)

            # For generative tasks, return answers as strings for evaluation
            # and prompt lengths for loss masking
            prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1).tolist()

            return input_ids, attention_mask, {"answers": answers, "prompt_lengths": prompt_lengths}

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


def normalize_answer(s: str) -> str:
    """Normalize answer for SQuAD evaluation."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_squad_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score for a single SQuAD prediction."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = set(pred_tokens) & set(truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_squad_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score for a single SQuAD prediction."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_generative_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: List[int],
) -> torch.Tensor:
    """
    Compute language modeling loss only on the answer portion (after prompt).

    Args:
        logits: [batch_size, seq_len, vocab_size] model output logits
        input_ids: [batch_size, seq_len] input token ids (prompt + answer)
        attention_mask: [batch_size, seq_len] attention mask
        prompt_lengths: List of prompt lengths (answer starts after this)

    Returns:
        Cross-entropy loss on answer tokens only
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Shift for causal LM: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Create mask for answer tokens only (after prompt)
    answer_mask = torch.zeros(batch_size, seq_len - 1, device=device)
    for i, prompt_len in enumerate(prompt_lengths):
        # Answer tokens start at position prompt_len (0-indexed)
        # After shift, we want positions from prompt_len-1 onwards
        start_pos = max(0, prompt_len - 1)
        end_pos = attention_mask[i].sum().item() - 1  # Last non-padding position
        if start_pos < end_pos:
            answer_mask[i, start_pos:end_pos] = 1.0

    # Compute cross-entropy loss
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    answer_mask = answer_mask.view(-1)

    loss = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, reduction='none'
    )

    # Apply mask and average
    masked_loss = loss * answer_mask
    if answer_mask.sum() > 0:
        return masked_loss.sum() / answer_mask.sum()
    else:
        return masked_loss.sum()  # Will be 0


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
        task_name: One of 'sst2', 'rte', 'boolq', 'wsc', 'wic', 'squad'
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test classification tasks
    for task_name in ["sst2", "rte", "boolq", "wsc", "wic"]:
        print(f"\nTesting {task_name}...")
        dataset = get_task_dataset(task_name, tokenizer, max_length=128)

        # Get a batch
        input_ids, attention_mask, labels = dataset.get_batch("train", batch_size=2)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Label token IDs: {dataset.label_token_ids}")
        print(f"  Task type: {dataset.task_type}")

    # Test generative task (SQuAD)
    print(f"\nTesting squad...")
    dataset = get_task_dataset("squad", tokenizer, max_length=256)
    input_ids, attention_mask, labels_info = dataset.get_batch("train", batch_size=2)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Answers: {labels_info['answers']}")
    print(f"  Prompt lengths: {labels_info['prompt_lengths']}")
    print(f"  Task type: {dataset.task_type}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_task_loading()
