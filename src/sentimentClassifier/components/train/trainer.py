import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import json
from sentimentClassifier import logger
import pandas as pd
from transformers import AutoTokenizer

from sentimentClassifier.entity import TrainingConfig
from sentimentClassifier.components.model import SentimentAnalyserModel
from sentimentClassifier.utils.common import check_file_exists
from .data_module import DataModule

    



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
        self.data_module = self.init_datamodule()

    def get_hubert_tokenizer(self):
        if check_file_exists(self.config.bert_tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(self.config.params_bert_tokenizer,do_lower_case= self.config.params_bert_tokenizer_do_lowercase)    
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config.params_bert_tokenizer,do_lower_case= self.config.params_bert_tokenizer_do_lowercase)
            tokenizer.save_pretrained(self.config.bert_tokenizer_path)
        return tokenizer

    def init_datamodule(self):
        df = pd.read_csv(self.config.training_data)
        reviews = df['Rating'].tolist()
        texts = df['Text'].tolist()
        data_module = DataModule(texts, reviews, self.tokenizer, self.config.params_bert_tokenizer_max_length, self.config.params_batch_size)
        return data_module
    
    def train_epoch(self, i_epoch: int):
        self.model.train()
        total_loss = 0.
        predictions, true_labels = [], []
        with tqdm(self.data_module.train_loader, unit="batch") as tepoch:
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
        return total_loss / len(self.data_module.train_loader),t_accuracy 

    def validate_epoch(self, i_epoch: int):
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []

        with torch.no_grad():
            with tqdm(self.data_module.val_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {i_epoch}")
                for batch in tepoch:
                    for batch in tepoch:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        targets = batch["targets"].to(self.device)

                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, targets)

                        total_loss += loss.item()
                        preds = F.log_softmax(outputs, dim=1)
                        preds = preds.argmax(dim=1, keepdim=True).squeeze()
                        predictions.extend(preds.cpu())
                        true_labels.extend(targets.cpu().tolist())
        print('1')
        accuracy = accuracy_score(true_labels, predictions)
        print('1')

        return total_loss / len(self.data_module.val_loader), accuracy

    def train(self):
        history = {"epoch_n": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_accuracy = 0
        for i, epoch in enumerate(range(self.config.params_epoch)):
            train_loss, train_acc = self.train_epoch(i)
            val_loss, val_acc = self.validate_epoch(i)
            print('2')

            history[f"epoch_n"].append(epoch)
            history[f"train_loss"].append(train_loss)
            history[f"train_acc"].append(train_acc)
            history[f"val_loss"].append(val_loss)
            history[f"val_acc"].append(val_acc)

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(self.model.state_dict(), self.config.trained_model_path)
            self.scheduler.step()
        print('2')
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.config.history_path, index=False)
    def test(self):
        print('3')

        test = {"review": [], "targets": [], "outputs": []}
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.data_module.test_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    for batch in tepoch:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)

                        outputs = self.model(input_ids, attention_mask)
                        test["review"].extend(batch["review"])
                        test["targets"].extend(batch["targets"].tolist())
                        test["outputs"].extend(outputs.detach().cpu().tolist())
        test_df = pd.DataFrame(test)
        test_df.to_csv(self.config.test_result_path, index=False)
