from argparse import ArgumentError
from typing import Optional, Tuple

import os, sys
import json
import random

from sklearn.model_selection import train_test_split
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers.models.auto.tokenization_auto import AutoTokenizer

from transformers import AutoTokenizer
from transformers import LlamaTokenizer

from src.data.components.wavelet_prosody import WaveletProminenceExtractor
from src.data.components.datasets import TokenTaggingDataset
from src.data.components.collators import collate_fn, encode_and_pad_batch

class WaveletDataModule(LightningDataModule):
    """
    LightningDataModule for Wavelet prominence data.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        shuffle_labels_yoked: bool = False,
        use_fast_tokenizer: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        model_name: str = None,
        score_first_token: bool = False,
        score_last_token: bool = False,
        relative_to_prev: bool = False,
        n_prev: int = 1,
        relative_to_mean: bool = False,
        word_stats_path: str = None,
        debug: bool = False,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "teacher_probs",
        ]

    def prepare_data(self):
        """
        Data preparation that should happen only once.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Data setup for each stage (fit/test).
        """
        if not self.hparams.relative_to_mean:
            self.hparams.word_stats = None
        elif self.hparams.relative_to_mean and not self.hparams.word_stats_path:
            raise ValueError(
                "If relative_to_mean is True, you must provide a word_stats_path."
            )
        else:
            self.hparams.word_stats = json.load(open(self.hparams.word_stats_path, "r"))

        if not self.tokenizer:
            if "gpt2" in self.hparams.model_name:
                print("Using GPT2 tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hparams.model_name, add_prefix_space=False
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif "bert" in self.hparams.model_name.lower():
                print(f"Using {self.hparams.model_name} tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
                self.tokenizer.pad_token_id = self.tokenizer.sep_token_id
            else:
                raise ValueError(
                    f"Model name {self.hparams.model_name} not recognized."
                )
        self.pad_token_id = self.tokenizer.pad_token_id
        print(f"Dataloader: padding with token id: {self.pad_token_id}")

        self.dataset_path = Path(self.hparams.data_dir)
        print(f"Loading data from {self.dataset_path}")
        if not os.path.exists(self.dataset_path):
            raise ValueError("The provided folder does not exist.")

        if stage == "fit":
            # Extract train and validation data
            train_extractor = WaveletProminenceExtractor(
                self.dataset_path, "train"
            )
            val_extractor = WaveletProminenceExtractor(
                self.dataset_path, "validation"
            )

            all_texts = train_extractor.get_all_texts() + val_extractor.get_all_texts()
            all_prominences = (
                train_extractor.get_all_real_prominence()
                + val_extractor.get_all_real_prominence()
            )

            # Split data into training and validation sets
            (
                self.train_texts,
                self.val_texts,
                self.train_prominences,
                self.val_prominences,
            ) = train_test_split(
                all_texts,
                all_prominences,
                test_size=0.1,
            )

            # creating a null version of the training labels that are yoked to the text
            # this will maintain the correspondence across batches
            if self.hparams.shuffle_labels_yoked:
                for prominence in self.train_prominences:
                    random.shuffle(prominence)

            # Create datasets
            self.train_dataset = TokenTaggingDataset(
                input_texts=self.train_texts,
                targets=self.train_prominences,
                tokenizer=self.tokenizer,
                model_name=self.hparams.model_name,
                score_first_token=self.hparams.score_first_token,
                score_last_token=self.hparams.score_last_token,
                relative_to_prev=self.hparams.relative_to_prev,
                n_prev=self.hparams.n_prev,
                relative_to_mean=self.hparams.relative_to_mean,
                word_stats=self.hparams.word_stats,
                debug=self.hparams.debug,
            )
            self.val_dataset = TokenTaggingDataset(
                input_texts=self.val_texts,
                targets=self.val_prominences,
                tokenizer=self.tokenizer,
                model_name=self.hparams.model_name,
                score_first_token=self.hparams.score_first_token,
                score_last_token=self.hparams.score_last_token,
                relative_to_prev=self.hparams.relative_to_prev,
                n_prev=self.hparams.n_prev,
                relative_to_mean=self.hparams.relative_to_mean,
                word_stats=self.hparams.word_stats,
                debug=self.hparams.debug,
            )

        if stage == "test":
            test_extractor = WaveletProminenceExtractor(
                self.dataset_path, "test"
            )
            self.test_texts = test_extractor.get_all_texts()
            self.test_prominences = test_extractor.get_all_real_prominence()
            self.test_dataset = TokenTaggingDataset(
                input_texts=self.test_texts,
                targets=self.test_prominences,
                tokenizer=self.tokenizer,
                model_name=self.hparams.model_name,
                score_first_token=self.hparams.score_first_token,
                score_last_token=self.hparams.score_last_token,
                relative_to_prev=self.hparams.relative_to_prev,
                n_prev=self.hparams.n_prev,
                relative_to_mean=self.hparams.relative_to_mean,
                word_stats=self.hparams.word_stats,
                debug=self.hparams.debug,
            )

    def encode_collate(self, batch):
        return encode_and_pad_batch(batch, self.tokenizer, self.hparams.model_name)

    def collate(self, batch):
        return collate_fn(batch, self.tokenizer.pad_token_id)

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