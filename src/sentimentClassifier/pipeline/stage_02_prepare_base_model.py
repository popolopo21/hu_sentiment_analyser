from sentimentClassifier.components.model import SentimentAnalyserModel
from sentimentClassifier.config import ConfigurationManager
import torch
from sentimentClassifier import logger

STAGE_NAME= "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        sentiment_analyser_model_config= config.get_sentiment_analyser_model_config()
        sentiment_analyser_model = SentimentAnalyserModel(config=sentiment_analyser_model_config)
        sentiment_analyser_model.save_model()

if __name__ == "__main__":
    try:
        logger.info(f"************************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e