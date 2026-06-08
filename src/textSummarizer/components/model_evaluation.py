import os
from typing import Dict, Iterable, List

import pandas as pd
import torch
from datasets import load_from_disk
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.entity import ModelEvaluationConfig
from textSummarizer.logging import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def _batch(items: List[str], batch_size: int) -> Iterable[List[str]]:
        for index in range(0, len(items), batch_size):
            yield items[index:index + batch_size]

    @staticmethod
    def _average_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=True,
        )
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        for prediction, reference in zip(predictions, references):
            scores = scorer.score(reference, prediction)
            for rouge_name in totals:
                totals[rouge_name] += scores[rouge_name].fmeasure

        count = max(len(predictions), 1)
        return {rouge_name: value / count for rouge_name, value in totals.items()}

    def _load_model_and_tokenizer(self):
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(
                f"Trained model not found at {self.config.model_path}. "
                "Run the model trainer stage before evaluation."
            )
        if not os.path.exists(self.config.tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {self.config.tokenizer_path}. "
                "Run the model trainer stage before evaluation."
            )

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        return model, tokenizer

    def evaluate(self, sample_size: int = 10, batch_size: int = 2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        model_pegasus, tokenizer = self._load_model_and_tokenizer()
        model_pegasus = model_pegasus.to(device)

        dataset_samsum_pt = load_from_disk(str(self.config.data_path))
        test_dataset = dataset_samsum_pt["test"]
        sample_size = min(sample_size, len(test_dataset))
        sample = test_dataset.select(range(sample_size))

        dialogues = sample["dialogue"]
        references = sample["summary"]
        predictions = []
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        for dialogue_batch in tqdm(list(self._batch(dialogues, batch_size)), desc="Evaluating"):
            inputs = tokenizer(
                dialogue_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            summaries = model_pegasus.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                **gen_kwargs,
            )
            decoded = tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions.extend(decoded)

        rouge_dict = self._average_rouge(predictions, references)
        os.makedirs(self.config.root_dir, exist_ok=True)
        pd.DataFrame([rouge_dict]).to_csv(self.config.metric_file_name, index=False)
        logger.info("Saved evaluation metrics to %s", self.config.metric_file_name)
        return rouge_dict
