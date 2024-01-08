from sentimentClassifier.components.model import SentimentAnalyserModel
from sentimentClassifier.config.configuration import ConfigurationManager
from sentimentClassifier import logger
from sentimentClassifier.components.train import Trainer

STAGE_NAME= "Train model"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        sentiment_analyser_model_config= config.get_sentiment_analyser_model_config()
        trainer_config = config.get_training_config()
        sentiment_analyser_model = SentimentAnalyserModel(config=sentiment_analyser_model_config)
        trainer = Trainer(model=sentiment_analyser_model, training_config=trainer_config)
        trainer.train()
        trainer.test()

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