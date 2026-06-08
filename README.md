# End-to-end Text Summarizer

This project implements an end-to-end text summarization pipeline using the SAMSum dataset and Google's PEGASUS model. It includes data ingestion, validation, transformation, model training, model evaluation, and a FastAPI prediction service.

## Implementation Workflow

Follow this order when building or updating the project:

1. Update `config/config.yaml`
   - Add artifact paths for every stage.
   - Configure the dataset source URL.
   - Configure model, tokenizer, and evaluation output paths.

2. Update `params.yaml`
   - Set training hyperparameters such as epochs, batch size, warmup steps, evaluation strategy, and gradient accumulation.

3. Update entity classes
   - Define dataclass configs in `src/textSummarizer/entity/__init__.py`.
   - Each pipeline stage should have its own config class.

4. Update the configuration manager
   - Add config builder methods in `src/textSummarizer/config/configuration.py`.
   - Each method should read YAML values and return the matching entity config.

5. Implement components
   - Add stage logic inside `src/textSummarizer/components/`.
   - Current components:
     - `data_ingestion.py`
     - `data_validation.py`
     - `data_transformation.py`
     - `model_trainer.py`
     - `model_evaluation.py`

6. Implement pipelines
   - Add orchestration files inside `src/textSummarizer/pipeline/`.
   - Current stages:
     - `stage_01_data_ingestion.py`
     - `stage_02_data_validation.py`
     - `stage_03_data_transformation.py`
     - `stage_04_model_trainer.py`
     - `stage_05_model_evaluation.py`
     - `prediction.py`

7. Update `main.py`
   - Run all training pipeline stages in order.

8. Update `app.py`
   - Expose the FastAPI routes for health checks, training, and prediction.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Full Training Pipeline

```bash
python main.py
```

This runs:

1. Data ingestion
2. Data validation
3. Data transformation
4. Model training
5. Model evaluation

The trained model is saved to:

```text
artifacts/model_trainer/pegasus-samsum-model
```

The tokenizer is saved to:

```text
artifacts/model_trainer/tokenizer
```

Evaluation metrics are saved to:

```text
artifacts/model_evaluation/metrics.csv
```

## Run the API

Start the FastAPI app:

```bash
python app.py
```

Open the API docs:

```text
http://127.0.0.1:8080/docs
```

Health check:

```text
GET /health
```

Train from the API:

```text
GET /train
```

Predict from the API:

```text
POST /predict
```

Example request body:

```json
{
  "text": "Your long dialogue or article goes here."
}
```

## Docker

Build the image:

```bash
docker build -t text-summarizer .
```

Run the container:

```bash
docker run -p 8080:8080 text-summarizer
```

## Notes

- Full PEGASUS fine-tuning can take a long time on CPU.
- Use a GPU runtime for faster model training.
- If local trained artifacts do not exist, prediction falls back to the configured base checkpoint.
