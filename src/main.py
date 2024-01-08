from sentimentClassifier import logger
from sentimentClassifier.pipeline import PrepareBaseModelTrainingPipeline, TrainingPipeline


# STAGE_NAME= "Prepare base model"


# if __name__ == "__main__":
#     try:
#         logger.info(f"************************************")
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
#         obj = PrepareBaseModelTrainingPipeline()
#         obj.main()
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
#     except Exception as e:
#         logger.exception(e)
#         raise e
    

STAGE_NAME= "Training model"


if __name__ == "__main__":
    try:
        logger.info(f"************************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e