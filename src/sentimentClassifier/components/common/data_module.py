from torch.utils.data import DataLoader, random_split
from .review_dataset import ReviewDataset
import pandas as pd

class DataModule:
    def __init__(self, data_path, tokenizer, batch_size):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_path = data_path

        self.loader = self._create_loader()

    def _create_loader(self):
        df = pd.read_csv(self.data_path)
        dataset = ReviewDataset(df.Text.tolist(), df.Score.tolist(), self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return loader