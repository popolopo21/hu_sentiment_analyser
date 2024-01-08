from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataPreparationConfig:
    port_reviews_file: Path
    amazon_reviews_file: Path
    final_reviews_file: Path

@dataclass(frozen=True)
class SentimentAnalyserConfig:
    root_dir: Path
    bert_tokenizer_path: Path
    bert_model_path: Path
    sentiment_model_path: Path
    params_bert_model: str
    params_bert_model_output_hidden_states: bool
    params_cnn_kernel_size: int
    params_cnn_dilations: List[int]
    params_lstm_n_filters: int
    params_lstm_hidden_dim: int
    params_lstm_n_layers: int
    params_lstm_batch_first: bool
    params_linear_output_dim: int
    params_lstm_bidirectional: bool
    params_dropout: float
    params_learning_rate: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    base_model_path: Path
    trained_model_path: Path
    bert_tokenizer_path: Path
    training_data: Path
    history_path: Path
    test_result_path: Path
    params_bert_tokenizer: str
    params_bert_tokenizer_max_length: int
    params_bert_tokenizer_do_lowercase:bool
    params_epoch: int
    params_batch_size: int
    params_learning_rate: float
    params_learning_rate_scheduler_gamma: float