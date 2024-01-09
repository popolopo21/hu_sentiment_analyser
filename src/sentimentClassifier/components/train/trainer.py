
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd
from transformers import AutoTokenizer

from sentimentClassifier.entity import TrainingConfig
from sentimentClassifier.components.model import SentimentAnalyserModel
from sentimentClassifier.utils.common import check_file_exists
from ..common import DataModule

class Trainer:

    def __init__(self, training_config: TrainingConfig, model: SentimentAnalyserModel):
        super().__init__()
        self.config = training_config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.params_learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.config.params_learning_rate_scheduler_gamma)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = self.get_hubert_tokenizer()
        self.train_dataloader, self.validation_dataloader = self.get_dataloaders()

    def get_hubert_tokenizer(self):
        if check_file_exists(self.config.bert_tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(self.config.bert_tokenizer_path,do_lower_case= False)    
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config.bert_tokenizer_uri,do_lower_case= False)
            tokenizer.save_pretrained(self.config.bert_tokenizer_path)
        return tokenizer

    def get_dataloaders(self):
        train_data_module = DataModule(self.config.training_data, self.tokenizer, self.config.params_batch_size)
        val_data_module = DataModule(self.config.validation_data, self.tokenizer, self.config.params_batch_size)

        return train_data_module.loader, val_data_module.loader
    
    def train_epoch(self, i_epoch: int):
        self.model.train()
        total_loss = 0.
        predictions, true_labels = [], []
        with tqdm(self.train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {i_epoch}")
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                preds = F.log_softmax(outputs, dim=1)
                preds = preds.argmax(dim=1, keepdim=True).squeeze()
                predictions.extend(preds.cpu())
                true_labels.extend(targets.tolist())
                correct = (preds == targets).sum().item()
                accuracy = correct / len(targets)
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                total_loss += loss.item()

        t_accuracy = accuracy_score(true_labels, predictions)
        return total_loss / len(self.train_dataloader),t_accuracy 

    def validate_epoch(self, i_epoch: int):
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        with torch.no_grad():
            with tqdm(self.validation_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {i_epoch}")
                for batch in tepoch:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    targets = batch["targets"].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, targets)

                    total_loss += loss.item()
                    preds = F.log_softmax(outputs, dim=1)
                    preds = preds.argmax(dim=1, keepdim=True).squeeze()
                    predictions.extend([pred.item() for pred in preds])
                    true_labels.extend(targets.cpu().tolist())
        accuracy = accuracy_score(true_labels, predictions)

        return total_loss / len(self.validation_dataloader), accuracy

    def train(self):
        best_accuracy = 0
        trigger_times = 0
        for i, epoch in enumerate(range(self.config.params_epoch)):
            train_loss, train_acc = self.train_epoch(i)
            val_loss, val_acc = self.validate_epoch(i)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(self.model.state_dict(), self.config.trained_model_path)
            else:
                trigger_times +=1
            if trigger_times>= self.config.params_early_stopping_patience:
                break
            self.scheduler.step()
