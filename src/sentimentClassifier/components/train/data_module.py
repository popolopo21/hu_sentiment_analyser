from torch.utils.data import DataLoader, random_split
from .review_dataset import ReviewDataset

class DataModule:
    def __init__(self, reviews, targets, tokenizer, max_len, batch_size, val_split=0.1, test_split=0.1):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

        self.train_loader, self.val_loader, self.test_loader = self.create_loaders()

    def create_loaders(self):
        # Splitting the data into train, validation, and test sets
        total_size = len(self.reviews)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size

        full_dataset = ReviewDataset(self.reviews, self.targets, self.tokenizer, self.max_len)
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        # Creating the DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader