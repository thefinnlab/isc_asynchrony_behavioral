from argparse import ArgumentError
from typing import Optional, Tuple

import os, sys

from pathlib import Path
from lightning import LightningDataModule
from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader, Subset

from src.data.components.audio_text_dataset import AudioTextDataset
# from src.data.components.datasets import TokenTaggingDataset
from src.data.components.collators import audio_text_collator


class AudioTextDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        cache_dir: str,
        audio_model_name: str = 'wav2vec2',
        text_model_name: str = 'gpt2',
        min_words: int = 4,
        max_words: int = 60,
        preload_audio: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        # train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        subset_percentage: float = None, 
        debug: bool = False,
    ):
        super().__init__()

        if dataset_name == 'libritts-r':
            self.hparams.train_split = 'train-clean-360'
            self.hparams.val_split = 'dev-clean'
            self.hparams.test_split = 'test-clean'
        else:
            self.hparams.train_split = 'train'
            self.hparams.val_split = 'validation'
            self.hparams.test_split = 'test'

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Ensure subsets are within the range we expect
        print (self.hparams.subset_percentage)
        print (type(self.hparams.subset_percentage))
        if self.hparams.subset_percentage:
            assert (self.hparams.subset_percentage > 0. and self.hparams.subset_percentage <= 1.0)

        self.dataset = None
        self.collator_fn = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.text_model_name, add_prefix_space=False, use_fast=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        self.dataset_path = Path(self.hparams.data_dir)

        print(f"Loading data from {self.dataset_path}")
        if not os.path.exists(self.dataset_path):
            raise ValueError("The provided folder does not exist.")

        if stage == "fit":
            

            # # creating a null version of the training labels that are yoked to the text
            # # this will maintain the correspondence across batches
            # if self.hparams.shuffle_labels_yoked:
            #     for prominence in self.train_prominences:
            #         random.shuffle(prominence)

            ####################################
            ######## EXTRACT TRAIN DATA ########
            ####################################

            # create datasets
            self.train_dataset = AudioTextDataset(
                dataset_dir=self.hparams.data_dir,
                cache_dir=self.hparams.cache_dir,
                audio_model_name=self.hparams.audio_model_name, 
                text_model_name=self.hparams.text_model_name, 
                split=self.hparams.train_split, 
                min_words=self.hparams.min_words,
                max_words=self.hparams.max_words,
                preload_audio=self.hparams.preload_audio,
            )

            self.train_dataset.preprocess_data()

            # If specified sample a subset of the data
            if self.hparams.subset_percentage:

                # Calculate size of the subset
                subset_size = int(self.hparams.subset_percentage * len(self.train_dataset))

                # Randomly split the dataset
                indices = torch.randperm(len(self.train_dataset)).tolist()
                subset_indices = indices[:subset_size]

                # Copy over for reference and create a subset
                self.full_train_dataset = self.train_dataset
                self.train_dataset = Subset(self.train_dataset, subset_indices)

            print (f"Train samples: {len(self.train_dataset)}", flush=True)

            ####################################
            ######### EXTRACT VAL DATA #########
            ####################################

            # create datasets
            self.val_dataset = AudioTextDataset(
                dataset_dir=self.hparams.data_dir,
                cache_dir=self.hparams.cache_dir,
                audio_model_name=self.hparams.audio_model_name, 
                text_model_name=self.hparams.text_model_name, 
                split=self.hparams.val_split, 
                min_words=self.hparams.min_words,
                max_words=self.hparams.max_words,
                preload_audio=self.hparams.preload_audio,
            )

            self.val_dataset.preprocess_data()

            print (f"Validation samples: {len(self.val_dataset)}", flush=True)

        if stage == "test":

            # create datasets
            self.test_dataset = AudioTextDataset(
                dataset_dir=self.hparams.data_dir,
                cache_dir=self.hparams.cache_dir,
                audio_model_name=self.hparams.audio_model_name, 
                text_model_name=self.hparams.text_model_name, 
                split=self.hparams.test_split, 
                min_words=self.hparams.min_words,
                max_words=self.hparams.max_words,
                preload_audio=self.hparams.preload_audio,
            )

            self.test_dataset.preprocess_data()

            print (f"Test samples: {len(self.test_dataset)}", flush=True)

    def collate(self, batch):
        return audio_text_collator(batch, pad_token=self.tokenizer.pad_token_id)

    # def collate(self, batch):
    #     return collate_fn(batch, self.tokenizer.pad_token_id)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )
