import pandas as pd
from typing import List
from loguru import logger

from readability_transformers.dataset import Dataset

class CommonLitDataset(Dataset):
    def __init__(self, label: str, cache: bool = True):
        """Loads the CommonLit dataset.

        Args:
            label (str): CommonLit dataset consists of the "train" dataset and the 
                         "test" dataset used for Kaggle evaluation. 
            cache (bool): if set to True, caches the train-valid-test split when called. Usually we train the
                SentenceTransformer first then train the ReadabilityPrediction model. We usually want to use
                the same splitted train-valid-test throughout the whole process (unless doing some sort of ablation component study).
        Returns:
            data (pd.DataFrame): .csv -> pd.DataFrame instance of the dataset.
        """
        super().__init__()
        
        self.cache = cache

        if label == "train":
            self.data = pd.read_csv("readability_transformers/dataset/data/train.csv")
        elif label == "test":
            self.data = pd.read_csv("readability_transformers/dataset/data/test.csv")
        else:
            raise Exception("Choose label = {train, test}.")

        for idx, row in self.data.iterrows():
            self.data.loc[idx, "excerpt"] = row["excerpt"].replace("\n", " ").replace("\t", " ").replace("  ", " ")
    