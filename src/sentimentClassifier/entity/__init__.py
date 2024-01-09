from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    

@dataclass(frozen=True)
class DataPreprocessConfig:
    root_dir:Path
    reviews: Path
    train_path: Path
    val_path: Path
    test_path: Path
    emojis_path: Path
    stopwords_path: Path
    accepted_punctuations_path: Path
    test_size: float
    val_size: float
    random_seed: int

@dataclass(frozen=True)
class SentimentAnalyserConfig:
    root_dir: Path
    bert_tokenizer_path: Path
    bert_model_path: Path
    sentiment_model_path: Path
    bert_model_uri: str
    params_cnn_kernel_size: int
    params_cnn_dilations: List[int]
    params_lstm_n_filters: int
    params_lstm_hidden_dim: int
    params_lstm_n_layers: int
    params_linear_output_dim: int
    params_lstm_bidirectional: bool
    params_dropout: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    base_model_path: Path
    trained_model_path: Path
    bert_tokenizer_path: Path
    training_data: Path
    validation_data: Path
    history_path: Path
    test_result_path: Path
    bert_tokenizer_uri: str
    params_epoch: int
    params_batch_size: int
    params_learning_rate: float
    params_learning_rate_scheduler_gamma: float
    params_early_stopping_patience: int
    params_random_seed: float

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    test_data_path: Path
    bert_tokenizer: str
    bert_tokenizer_path: Path
    model_params: dict
    training_params:dict
    params_batch_size: int
    params_random_seed: int
