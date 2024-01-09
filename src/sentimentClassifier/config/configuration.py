from sentimentClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from sentimentClassifier.utils.common import read_yaml, create_directories
from sentimentClassifier.entity import (
    DataIngestionConfig,
    DataPreprocessConfig, 
    SentimentAnalyserConfig,
    TrainingConfig,
    EvaluationConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocess_config(self) -> DataPreprocessConfig:
        config = self.config.data_preprocess
        params = self.params.data_preprocess

        create_directories([config.root_dir])

        data_preprocess_config = DataPreprocessConfig(
            root_dir=config.root_dir,
            reviews=config.reviews_path,
            train_path=config.train_path,
            val_path=config.val_path,
            test_path=config.test_path,
            emojis_path=config.emojis_path,
            stopwords_path =config.stopwords_path,
            accepted_punctuations_path= config.accepted_punctuations_path,
            test_size= params.test_size,
            val_size=params.val_size,
            random_seed=params.random_seed
        )

        return data_preprocess_config

    
    def get_sentiment_analyser_model_config(self) -> SentimentAnalyserConfig:
        config = self.config.model
        params = self.params.model

        create_directories([config.root_dir])

        sentiment_analyser_model_config = SentimentAnalyserConfig(
            root_dir = config.root_dir,
            bert_tokenizer_path = config.bert_tokenizer_path,
            bert_model_path = config.bert_model_path,
            sentiment_model_path = config.sentiment_model_path,
            bert_model_uri = config.bert_model_uri,
            params_cnn_kernel_size = params.cnn_kernel_size,
            params_cnn_dilations = params.cnn_dilations,
            params_lstm_n_filters = params.lstm_n_filters,
            params_lstm_hidden_dim = params.lstm_hidden_dim,
            params_lstm_n_layers = params.lstm_n_layers,
            params_lstm_bidirectional = params.lstm_bidirectional,
            params_linear_output_dim = params.linear_output_dim,
            params_dropout = params.dropout,
        )

        return sentiment_analyser_model_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params.training
        model_config = self.config.model

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir = Path(config.root_dir),
            base_model_path= Path(model_config.sentiment_model_path),
            trained_model_path= Path(config.trained_model_path),
            bert_tokenizer_uri= config.bert_tokenizer_uri,
            bert_tokenizer_path= Path(config.bert_tokenizer_path),
            training_data= Path(config.training_data_path),
            validation_data= Path(config.validation_data_path),
            history_path = Path(config.history_path),
            test_result_path = Path(config.test_result_path),
            params_epoch= params.epoch,
            params_batch_size= params.batch_size,
            params_learning_rate = params.learning_rate,
            params_learning_rate_scheduler_gamma = params.learning_rate_gamma_value,
            params_early_stopping_patience = params.early_stopping_patience,
            params_random_seed= params.random_seed
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        params = self.params.evaluation
        training_config = self.config.training
        model_params = self.params.model
        training_params = self.params.training

        create_directories([config.root_dir])

        evaluation_config = EvaluationConfig(
            root_dir = Path(config.root_dir),
            trained_model_path= Path(config.trained_model_path),
            test_data_path= Path(config.test_data_path),
            bert_tokenizer_path=Path(config.bert_tokenizer_path),
            bert_tokenizer = training_config.bert_tokenizer_uri,
            model_params=model_params,
            training_params= training_params,
            params_batch_size= params.batch_size,
            params_random_seed= params.random_seed,
        )

        return evaluation_config