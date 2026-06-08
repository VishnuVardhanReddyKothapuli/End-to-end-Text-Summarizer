import inspect
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        os.environ["WANDB_DISABLED"] = "true"

    def _training_arguments(self) -> TrainingArguments:
        kwargs = {
            "output_dir": self.config.root_dir,
            "num_train_epochs": self.config.num_train_epochs,
            "warmup_steps": self.config.warmup_steps,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_train_batch_size,
            "weight_decay": self.config.weight_decay,
            "logging_steps": self.config.logging_steps,
            "eval_steps": self.config.eval_steps,
            "save_steps": self.config.save_steps,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "report_to": "none",
        }

        argument_names = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in argument_names:
            kwargs["eval_strategy"] = self.config.eval_strategy
        else:
            kwargs["evaluation_strategy"] = self.config.eval_strategy

        return TrainingArguments(**kwargs)

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        dataset_samsum_pt = load_from_disk(str(self.config.data_path))
        train_split = "test" if "test" in dataset_samsum_pt else "train"
        eval_split = "validation" if "validation" in dataset_samsum_pt else train_split

        trainer = Trainer(
            model=model_pegasus,
            args=self._training_arguments(),
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt[train_split],
            eval_dataset=dataset_samsum_pt[eval_split],
        )

        trainer.train()

        model_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
        tokenizer_path = os.path.join(self.config.root_dir, "tokenizer")
        model_pegasus.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)
        logger.info("Saved model to %s and tokenizer to %s", model_path, tokenizer_path)
