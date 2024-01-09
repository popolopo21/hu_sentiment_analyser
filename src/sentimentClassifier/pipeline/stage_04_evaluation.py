from sentimentClassifier.components.model.sentiment_analyser import SentimentAnalyserModel
from sentimentClassifier.components.evaluation.evaluation import Evaluation
from sentimentClassifier.config.configuration import ConfigurationManager
from sentimentClassifier import logger

STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        sentiment_analyser_model_config = config.get_sentiment_analyser_model_config()
        base_model = SentimentAnalyserModel(sentiment_analyser_model_config)
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(evaluation_config, base_model)
        evaluation.test()
        evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f"************************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
