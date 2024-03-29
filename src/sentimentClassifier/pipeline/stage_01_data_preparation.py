from sentimentClassifier.config.configuration import ConfigurationManager
from sentimentClassifier.components.data_preprocess import DataPreprocess
from sentimentClassifier import logger

STAGE_NAME = "Data Preprocess stage"


class DataPreprocessTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocess_config = config.get_data_preprocess_config()
        data_preprocess = DataPreprocess(data_preprocess_config)
        data_preprocess.process_and_save()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocess()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e