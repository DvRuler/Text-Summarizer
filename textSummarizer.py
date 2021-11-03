#%%
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

# %%

#Creating the dataframe
df = pd.read_csv("", encoding="latin-1") #enter filepath to data here
df = df[["text", "ctext"]]
df.columns = ["summary", "full-text"]
df = df.dropna()

# %%

#Split data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.1)

# %%
#Create dataset
class SummaryData(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def _len_(self):
        return len(self.data)

    def _getitem_(self, index: int):
        data_row = self.data.iloc[index]

        fullText = data_row["full-text"]

        fullText_encoding = tokenizer(
            fullText,
            max_length = self.text_max_token_len,
            padding = "max_length",
            truncation=True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = "pt"
            )

        summary = data_row["summary"]

        summary_encoding = tokenizer(
            summary,
            max_length = self.summary_max_token_len,
            padding = "max_length",
            truncation=True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = "pt"
        )

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            fullText = fullText,
            summary = summary,
            fullText_input_ids = fullText_encoding["input_ids"].flatten(),
            fullText_attention_mask = fullText_encoding["attention_mask"].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = summary_encoding["attention_mask"].flatten()
        )

# %%

class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        fullText_max_token_len: int = 512,
        summary_max_token_len: int = 128
        ):

        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.fulltext_max_token_len = fullText_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage = None):
        self.train_dataset = SummaryData(
            self.train_df,
            self.tokenizer,
            self.fulltext_max_token_len,
            self.summary_max_token_len
            )

        self.test_dataset = SummaryData(
            self.test_df,
            self.tokenizer,
            self.fulltext_max_token_len,
            self.summary_max_token_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 4
        )
# %%

#Select model and initialise tokenizer - base T5
MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# %%

#Model Parameters
N_EPOCHS = 3
BATCH_SIZE = 8

data_module = SummaryDataModule(
    train_df,
    test_df,
    tokenizer,
    batch_size = BATCH_SIZE
    )

#Building the model

class SummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict =True)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_attention_mask,
        labels = None
        ):
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(
        self,
        batch,
        batch_idx
        ):
        input_ids = batch["fullText_input_ids"]
        attention_mask = batch["fullText_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
        )

        self.log(
            "train_loss",
            loss,
            prog_bar = True,
            logger = True
            )
        return loss

    def validation_step(
        self,
        batch,
        batch_idx
        ):
        input_ids = batch["fullText_input_ids"]
        attention_mask = batch["fullText_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
        )

        self.log(
            "val_loss",
            loss,
            prog_bar = True,
            logger = True
            )
        return loss

    def test_step(
        self,
        batch,
        batch_idx
        ):
        input_ids = batch["fullText_input_ids"]
        attention_mask = batch["fullText_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
        )

        self.log(
            "test_loss",
            loss,
            prog_bar = True,
            logger = True
            )
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = 0.0001)

# %%

#Create instance of model
model = SummaryModel()

# %%

#Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath = "checkpoints",
    filename = "best-checkpoint",
    save_top_k = 1,
    verbose = True,
    monitor = "val_loss",
    mode = "min"
)

logger = TensorBoardLogger("lightning_logs", name = "news-article-summary")

#Create trainer
trainer = pl.Trainer(
    logger = logger,
    checkpoint_callback = checkpoint_callback,
    max_epochs = N_EPOCHS,
    progress_bar_refresh_rate = 30
)

#Train model
trainer.fit(model, data_module)