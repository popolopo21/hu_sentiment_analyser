import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sentimentClassifier.entity import SentimentAnalyserConfig
from sentimentClassifier.utils.common import check_file_exists

class SentimentAnalyserModel(nn.Module):
    def __init__(self, config: SentimentAnalyserConfig):
        super().__init__()

        self.config = config
        self.bert = self._get_hubert_model()

        self.convs = self._get_cnns()
        self.lstm = self._get_lstm()
        self.out = self._get_linear()
        self.dropout = self._get_dropout()

    def _get_cnns(self):
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        return nn.ModuleList([
                                nn.Conv1d(in_channels = 4*embedding_dim, 
                                            out_channels = self.config.params_lstm_n_filters, 
                                            kernel_size = self.config.params_cnn_kernel_size,
                                            dilation = dilation
                                            )
                                for dilation in self.config.params_cnn_dilations
                            ])
    
    def _get_lstm(self):
        return nn.LSTM(
                        self.config.params_lstm_n_filters,
                        self.config.params_lstm_hidden_dim,
                        num_layers = self.config.params_lstm_n_layers,
                        bidirectional = self.config.params_lstm_bidirectional,
                        batch_first = True,
                        dropout = 0 if self.config.params_lstm_n_layers < 2 else self.config.params_dropout
                        )
    
    def _get_linear(self):
        return nn.Linear(self.config.params_lstm_hidden_dim * 2 if self.config.params_lstm_bidirectional else self.config.params_lstm_hidden_dim, self.config.params_linear_output_dim)

    def _get_dropout(self):
        return nn.Dropout(self.config.params_dropout)
    
    def _get_hubert_model(self):
        if check_file_exists(self.config.bert_model_path):
            model = AutoModel.from_pretrained(self.config.bert_model_path, output_hidden_states=True)    
        else:
            model = AutoModel.from_pretrained(self.config.bert_model_uri, output_hidden_states=True)
            model.save_pretrained(self.config.bert_model_path)
        return model
    

    def forward(self, input_ids,attention_mask):
        for param in self.bert.parameters():
            param.requires_grad = False
        embedded = self.bert(input_ids = input_ids,
                            attention_mask = attention_mask)
        
        hidden_states = embedded[2]
        embedded = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)

        embedded = embedded.permute(0, 2, 1)

        conved = [F.max_pool1d(F.relu(conv(embedded)),1).permute(0, 2, 1) for conv in self.convs]
        cat = torch.cat(conved, dim=1)
        #embedded = [batch size, sent len, emb dim]
        _, hidden = self.lstm(cat)
        #hidden = [n layers * n directions, batch size, emb dim]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        output = self.out(hidden)
        #output = [batch size, out dim]        
        return output
    
    def save_model(self):
        torch.save(self.state_dict(), self.config.sentiment_model_path)