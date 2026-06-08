from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from textSummarizer.logging import logger


def run_stage(stage_name, pipeline_cls):
    try:
        logger.info(f'>>>>> stage {stage_name} started <<<<<')
        pipeline_cls().main()
        logger.info(f'>>>>>> stage {stage_name} completed <<<<<\n\nx==============x')
    except Exception as e:
        logger.exception(e)
        raise e


def run_training_pipeline():
    stages = [
        ("Data Ingestion stage", DataIngestionTrainingPipeline),
        ("Data Validation stage", DataValidationTrainingPipeline),
        ("Data Transformation stage", DataTransformationTrainingPipeline),
        ("Model Trainer stage", ModelTrainerTrainingPipeline),
        ("Model Evaluation stage", ModelEvaluationTrainingPipeline),
    ]
    for stage_name, pipeline_cls in stages:
        run_stage(stage_name, pipeline_cls)

if __name__ == "__main__":
    run_training_pipeline()
