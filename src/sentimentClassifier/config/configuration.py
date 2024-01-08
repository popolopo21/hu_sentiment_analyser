from sentimentClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
import os
from sentimentClassifier.utils.common import read_yaml, create_directories
from sentimentClassifier.entity import DataPreparationConfig, SentimentAnalyserConfig, TrainingConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    
    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation

        data_preparation_config = DataPreparationConfig(
            port_reviews_file = config.port_reviews_file,
            amazon_reviews_file = config.amazon_reviews_file,
            final_reviews_file = config.final_reviews_file,
        )

        return data_preparation_config
    
    def get_sentiment_analyser_model_config(self) -> SentimentAnalyserConfig:
        config = self.config.prepare_base_model
        params = self.params

        create_directories([config.root_dir])

        sentiment_analyser_model_config = SentimentAnalyserConfig(
            root_dir = config.root_dir,
            bert_tokenizer_path = config.bert_tokenizer_path,
            bert_model_path = config.bert_model_path,
            sentiment_model_path = config.sentiment_model_path,
            params_bert_model = params.BERT_MODEL,
            params_bert_model_output_hidden_states = params.BERT_MODEL_OUTPUT_HIDDEN_STATES,
            params_cnn_kernel_size = params.CNN_KERNEL_SIZE,
            params_cnn_dilations = params.CNN_DILATIONS,
            params_lstm_n_filters = params.LSTM_N_FILTERS,
            params_lstm_hidden_dim = params.LSTM_HIDDEN_DIM,
            params_lstm_n_layers = params.LSTM_N_LAYERS,
            params_lstm_batch_first = params.LSTM_BATCH_FIRST,
            params_linear_output_dim = params.LINEAR_OUTPUT_DIM,
            params_lstm_bidirectional = params.LSTM_BIDIRECTIONAL,
            params_dropout = params.DROPOUT,
            params_learning_rate = params.LEARNING_RATE
        )

        return sentiment_analyser_model_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params
        prepare_base_model = self.config.prepare_base_model

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir = Path(config.root_dir),
            base_model_path= Path(prepare_base_model.sentiment_model_path),
            trained_model_path= Path(config.trained_model_path),
            bert_tokenizer_path= Path(config.bert_tokenizer_path),
            training_data= Path(config.training_data_path), #Needs to refactor
            history_path = Path(config.history_path),
            test_result_path = Path(config.test_result_path),
            params_epoch= params.EPOCH,
            params_batch_size= params.BATCH_SIZE,
            params_learning_rate = params.LEARNING_RATE,
            params_bert_tokenizer= params.BERT_TOKENIZER,
            params_bert_tokenizer_max_length = params.BERT_TOKENIZER_MAX_LENGTH,
            params_bert_tokenizer_do_lowercase = params.BERT_TOKENIZER_DO_LOWERCASE,
            params_learning_rate_scheduler_gamma = params.LEARNING_RATE_SCHEDULER_GAMMA
        )

        return training_config