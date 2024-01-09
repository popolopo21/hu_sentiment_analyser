import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

class ReviewDataset(Dataset):

    def __init__(self, review, target, tokenizer):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.review)
    def __getitem__(self,item):
        review =str(self.review[item])

        encoding = self.tokenizer.encode_plus(
            review,
            max_length=512,
            add_special_tokens = True,
            padding = 'max_length',
            truncation=True,
            return_attention_mask = True,
            return_token_type_ids = False,
            return_tensors='pt'
        )

        return{
            'review' : review,
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.target[item], dtype = torch.long)
        }