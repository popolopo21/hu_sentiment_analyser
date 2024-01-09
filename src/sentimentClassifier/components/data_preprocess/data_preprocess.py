import pandas as pd
from sklearn.model_selection import train_test_split

from sentimentClassifier.entity import DataPreprocessConfig


class DataPreprocess:
    def __init__(self, preprocess_config: DataPreprocessConfig):
        self.config = preprocess_config
        self.df = pd.read_csv(self.config.reviews)
    
    def split_data(self):
        # Split data into train and test
        train_df, test_df = train_test_split(self.df, test_size=self.config.test_size, random_state=self.config.random_seed)

        # Further split train data into train and validation
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)  # Adjust val size based on remaining data
        train_df, val_df = train_test_split(train_df, test_size=val_size_adjusted, random_state=self.config.random_seed)

        return train_df, val_df, test_df

    def save_data(self, train_df, val_df, test_df):

        # Save datasets to csv
        train_df.to_csv(self.config.train_path, index=False)
        val_df.to_csv(self.config.val_path, index=False)
        test_df.to_csv(self.config.test_path, index=False)

    def process_and_save(self):
        # Process and split the data
        train_df, val_df, test_df = self.split_data()

        # Save the data
        self.save_data(train_df, val_df, test_df)