import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = None

    def _get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        return self.tokenizer

    def convert_examples_to_features(self, example_batch):
        tokenizer = self._get_tokenizer()
        input_encodings = tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True
        )

        try:
            target_encodings = tokenizer(
                text_target=example_batch['summary'],
                max_length=128,
                truncation=True
            )
        except TypeError:
            with tokenizer.as_target_tokenizer():
                target_encodings = tokenizer(
                    example_batch['summary'],
                    max_length=128,
                    truncation=True
                )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        output_path = os.path.join(self.config.root_dir, 'samsum_dataset')
        if os.path.exists(os.path.join(output_path, "dataset_dict.json")):
            logger.info("Transformed dataset already exists at %s", output_path)
            return

        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(output_path)
