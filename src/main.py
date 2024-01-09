from sentimentClassifier import logger
from sentimentClassifier.pipeline import  DataPreprocessTrainingPipeline,DataIngestionTrainingPipeline, PrepareBaseModelTrainingPipeline, TrainingPipeline, EvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data PreProcessing stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_preprocess = DataPreprocessTrainingPipeline()
   data_preprocess.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME= "Prepare base model"


try:
    logger.info(f"************************************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME= "Training model"


try:
    logger.info(f"************************************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation"


try:
    logger.info(f"************************************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e