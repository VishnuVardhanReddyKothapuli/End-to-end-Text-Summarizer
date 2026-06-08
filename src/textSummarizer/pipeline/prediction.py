import os

from transformers import AutoTokenizer, pipeline

from textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.prediction_config = config.get_model_evaluation_config()
        self.trainer_config = config.get_model_trainer_config()

    def _model_source(self):
        model_path = str(self.prediction_config.model_path)
        tokenizer_path = str(self.prediction_config.tokenizer_path)
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            return model_path, tokenizer_path
        return self.trainer_config.model_ckpt, self.trainer_config.model_ckpt

    def predict(self, text: str) -> str:
        model_source, tokenizer_source = self._model_source()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        summarizer = pipeline("summarization", model=model_source, tokenizer=tokenizer)
        return summarizer(text, **gen_kwargs)[0]["summary_text"]
