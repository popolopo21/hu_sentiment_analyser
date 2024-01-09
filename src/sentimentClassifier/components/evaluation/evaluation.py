from urllib.parse import urlparse
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
import mlflow
from transformers import AutoTokenizer
import os
from sentimentClassifier.components.model.sentiment_analyser import SentimentAnalyserModel
from sentimentClassifier.entity import EvaluationConfig
from sentimentClassifier.utils.common import check_file_exists
from sentimentClassifier.components.common import DataModule
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


load_dotenv(find_dotenv())

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

class Evaluation:
    def __init__(self, evaluation_config: EvaluationConfig, model: SentimentAnalyserModel):
        super().__init__()
        self.config = evaluation_config
        self.model = model
        self.model.load_state_dict(torch.load(evaluation_config.trained_model_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = self.get_hubert_tokenizer()
        self.test_dataloader = self.get_dataloader()

    def get_hubert_tokenizer(self):
        if check_file_exists(self.config.bert_tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(self.config.bert_tokenizer_path,do_lower_case=True)    
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config.bert_tokenizer,do_lower_case=True)
            tokenizer.save_pretrained(self.config.bert_tokenizer_path)
        return tokenizer

    def get_dataloader(self):
        test_data_module = DataModule(self.config.test_data_path, self.tokenizer, self.config.params_batch_size)
        return test_data_module.loader
    
    def test(self):
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        with torch.no_grad():
            with tqdm(self.test_dataloader, unit="batch") as tepoch:
                tepoch.set_description("Evaluating model")
                for batch in tepoch:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    targets = batch["targets"].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item()
                    preds = F.log_softmax(outputs, dim=1)
                    preds = preds.argmax(dim=1, keepdim=True).squeeze()
                    predictions.extend(preds.cpu().tolist())
                    true_labels.extend(targets.cpu().tolist())

        self.accuracy = accuracy_score(true_labels, predictions)
        self.loss = total_loss / len(self.test_dataloader)
        self.cr= classification_report(true_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot()
        self.conf_matrix_figure = plt.gcf()  # Get the current figure

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.log_params(self.config.model_params)
            mlflow.log_params(self.config.training_params)
            mlflow.log_metrics({
                "loss": self.loss
            })
            mlflow.log_metric("accuracy", self.cr.pop("accuracy"))
            for class_or_avg, metrics_dict in self.cr.items():
                for metric, value in metrics_dict.items():
                    mlflow.log_metric(class_or_avg + '_' + metric,value)
            mlflow.log_figure(self.conf_matrix_figure, 'confusion_matrix.png')
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="SentimentAnalyserModel")
            else:
                mlflow.pytorch.log_model(self.model, "model")
