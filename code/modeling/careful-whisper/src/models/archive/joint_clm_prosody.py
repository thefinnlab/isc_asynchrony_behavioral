import os
import inspect
from typing import Any, Dict, List, Tuple
import pickle

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MinMetric, MaxMetric
from torch.distributions.gamma import Gamma
from torch.nn import L1Loss
import numpy as np

from torch import Tensor
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelWithLMHead
from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup

from src.utils import utils
from src.utils.torch_utils import (
	masked_loss, 
	print_num_trainable_params
)

from src.utils.torch_metrics import (
	MaskedMeanAbsoluteError,
	MaskedR2Score,
	MaskedPearsonCorrCoef,
	MeanMetric,
	MaskedAccuracy
)

class ProsodyCausalLM(LightningModule):
	def __init__(
			self,
			model_name: str,
			optimizer: torch.optim.Optimizer = AdamW,
			scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
			loss_fn: nn.Module = None,
			output_activation: nn.Module = torch.nn.Identity(),
			num_labels: int = 1,
			p_dropout: float = 0.1,
			# use_peft: bool = True,
			# lora_kwargs: Dict = {}
		):
			super().__init__()

			self.save_hyperparameters(logger=False, ignore=["loss_fn", "output_activation"])
	
			# initialize the model
			print("Loading Huggingface model.")
			self.model = AutoModelForCausalLM.from_pretrained(model_name)

			# create embedding layer for prosody
			self.prosody_embed = nn.Linear(1, self.model.config.hidden_size, bias=False)

			# projection from hidden state to prosody value
			self.dropout = nn.Dropout(p_dropout)
			self.regressor = nn.Linear(self.model.config.hidden_size, num_labels*2) # mu and variance for each label
			self.output_activation = output_activation

			# initialize weights according to initialization rules of model
			self.model._init_weights(self.prosody_embed)

			# if use_peft:
			#     lora_config = LoraConfig(**lora_kwargs)
			#     get_peft_model(self.model, lora_config)
			#     # lora_config = LoraConfig(
			#     #     r=16,
			#     #     lora_alpha=16,
			#     #     lora_dropout=0.1,
			#     #     task_type="CAUSAL_LM"
			#     # )
						
			######################
			### Setup metrics ####
			######################
	
			# use separate metric instance for train, val and test step
			# to ensure a proper reduction over the epoch
			self.train_loss = MeanMetric()  # already masked in loss function in step
			self.val_loss = MeanMetric()
			self.test_loss = MeanMetric()

			# clm loss
			self.train_clm_loss = MeanMetric()
			self.val_clm_loss = MeanMetric()
			self.test_clm_loss = MeanMetric()

			# prosody loss
			self.train_prosody_loss = MeanMetric()
			self.val_prosody_loss = MeanMetric()
			self.test_prosody_loss = MeanMetric()

			# mae
			self.train_mae = MaskedMeanAbsoluteError()
			self.val_mae = MaskedMeanAbsoluteError()
			self.test_mae = MaskedMeanAbsoluteError()

			# perplexity
			self.train_perplexity = MeanMetric()
			self.val_perplexity = MeanMetric()
			self.test_perplexity = MeanMetric()

			# accuracy
			self.train_accuracy = MeanMetric()
			self.val_accuracy = MeanMetric()
			self.test_accuracy = MeanMetric()
	
			# self.train_r2 = MaskedR2Score()
			self.val_r2 = MaskedR2Score()
			self.test_r2 = MaskedR2Score()
	
			self.val_pearson = MaskedPearsonCorrCoef()
			self.test_pearson = MaskedPearsonCorrCoef()
	
			# for logging best so far validation accuracy
			self.val_loss_best = MinMetric()
			self.val_clm_loss_best = MinMetric()
			self.val_prosody_loss_best = MinMetric()

			self.val_accuracy_best = MaxMetric()
			self.val_perplexity_best = MinMetric()

			self.val_mae_best = MinMetric()
			self.val_r2_best = MaxMetric()
			self.val_pearson_best = MaxMetric()

			# print number of trainable parameters
			print_num_trainable_params(
				self, model_name=f"ProsodyCausalLM {model_name}"
			)

			print (self.hparams)
	
	def get_input_embeddings(self, input_ids, prosody_values):
		
		# get token embedding and project prominence to embedding space
		input_embeds = self.model.transformer.wte(input_ids)
		prosody_embed = self.prosody_embed(prosody_values.unsqueeze(-1))

		# add the two together
		input_embeds = input_embeds + prosody_embed
		
		return input_embeds

	def get_shifted_labels(self, logits, labels, mask):
		'''
		Get shifted logits/labels for CLM modeling --> first logit corresponds to second label
		'''
		# shift the logits and labels to be paired
		shifted_logits = logits[..., :-1, :].contiguous()
		shifted_labels = labels[..., 1:].contiguous()

		# flatten labels and logits 
		shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))
		shifted_labels = shifted_labels.view(-1)

		# shift the attention mask to be in line with the labels 
		shifted_mask = mask[..., 1:].contiguous().view(-1)		

		return shifted_logits, shifted_labels, shifted_mask

	def calculate_accuracy(self, preds: Tensor, target: Tensor, mask: Tensor):

		preds = torch.argmax(preds, dim=-1)
		correct = (preds == target) * mask.bool()

		correct = torch.sum(correct)
		total = torch.sum(mask)

		return correct.float() / total

	def clm_loss(self, batch, outputs):
		'''
		Causal language modeling loss --> cross-entropy over each predicted token and 
		the ground truth token
		'''

		logits, labels, mask = self.get_shifted_labels(
			logits=outputs['logits'],
			labels=batch['input_ids'],
			mask=batch['attention_mask']
		)

		# Flatten the tokens and compute loss only over non-masked tokens
		loss = masked_loss(
			labels=labels,
			predictions=logits,
			mask=mask,
			loss_fn=nn.CrossEntropyLoss(reduction="none"),
		)
		
		return loss

	def prosody_loss(self, batch, outputs):
		'''
		Prosody NLL loss --> models the mean and variance of data distribution
		attempts to predict ground-truth prosody value
		'''

		# prosody labels
		labels = batch['tokenized_labels']
		dist = outputs['dist']
		loss_mask = batch["loss_mask"]  # ignore padded sequence in loss
		
		# log likelihood of labels given the distribution
		labels = labels * loss_mask + 1e-4  # add small constant for numerical stability
		nll = -dist.log_prob(labels)

		# mask loss
		masked_nll = nll * loss_mask
		masked_nll_mean = masked_nll.sum() / loss_mask.sum()

		return masked_nll_mean
		
	def forward(self, batch: Dict[str, torch.tensor], eps=1e-4, verbose=False):
		
		# tokenized labels = prosody values
		input_embeds = self.get_input_embeddings(
			input_ids = batch['input_ids'], 
			prosody_values = batch['tokenized_labels']
		)

		# get outputs from causal LM
		outputs = self.model.transformer(
			inputs_embeds=input_embeds, 
			attention_mask=batch["attention_mask"], 
		)

		# get the logits for predicting each item in the sequence
		logits = self.model.lm_head(outputs.last_hidden_state)

		# get prosody predictions
		preds = self.regressor(outputs.last_hidden_state)

		# split last dimension into mu and var
		mu, var = torch.chunk(preds, chunks=2, dim=-1)
		
		# ensure positivity of var + add a small constant for numerical stability
		var = F.softplus(var)
		var = (var + eps).squeeze(-1)
		
		# have to squeeze the last dimension due to chunking
		if self.output_activation is not None:
			mu = self.output_activation(mu.squeeze(-1))

		# Gamma distribution with concentration mu and rate var
		mu = F.softplus(mu)
		dist = Gamma(mu, var)
		preds = dist.mean
		
		# return predictions 
		outputs = {
			'logits': logits,
			'dist': dist,
			'preds': preds,
			'mu': mu,
			'var': var,
		}
		
		return outputs

	def step(self, batch: Dict[str, torch.tensor], verbose: bool = False):
		if verbose:
			print(f"-- Step --")

		# make forward pass
		outputs = self(batch)

		# calculate loss for each
		clm_loss = self.clm_loss(batch, outputs)
		prosody_loss = self.prosody_loss(batch, outputs)

		# combine the two losses
		loss = clm_loss + prosody_loss

		outputs.update({'clm_loss': clm_loss, 'prosody_loss': prosody_loss})

		# if verbose:
		#     print(f"text: {batch['input_text']}")
		#     print(f"labels {labels}, \nmask {loss_mask}")

		return loss, outputs

	def on_train_start(self):
		"""
		Reset validation metrics 
		"""
		pass
		# # reset loss
		# self.val_loss.reset()
		# self.val_clm_loss.reset()
		# self.val_prosody_loss.reset()

		
		# self.val_loss_best.reset()
		# self.val_clm_loss_best.reset()
		# self.val_prosody_loss_best.reset()

		# # reset r2
		# self.val_r2.reset()
		# self.val_r2_best.reset()

	def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
		
		# loss is batch loss, train_loss tracks it over the epoch
		loss, outputs = self.step(batch)
		
		# perplexity is the exponentiated NLL of a token given a sequence
		perplexity = torch.exp(outputs['clm_loss'].mean())

		# get shifted logits and labels for accuracy
		logits, labels, mask = self.get_shifted_labels(outputs["logits"], batch["input_ids"], batch["attention_mask"])
		accuracy = self.calculate_accuracy(logits, labels, mask)

		# add loss to entire epoch, also calculate MAE over the step
		self.train_loss(loss)
		
		# log individual losses
		self.train_clm_loss(outputs['clm_loss'])
		self.train_prosody_loss(outputs['prosody_loss'])

		# calculate metrics
		self.train_perplexity(perplexity)
		self.train_accuracy(accuracy)

		self.train_mae(outputs["preds"], batch["tokenized_labels"], batch["loss_mask"])
		
		self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log("train/clm_loss", self.train_clm_loss, on_step=True, on_epoch=True, prog_bar=True)
		self.log("train/prosody_loss", self.train_prosody_loss, on_step=True, on_epoch=True, prog_bar=True)

		self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
		self.log("train/perplexity", self.train_perplexity, on_step=True, on_epoch=True, prog_bar=True)
		self.log("train/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)
		
		return {
			"loss": loss,
			"preds": outputs["preds"],
			"targets": batch["tokenized_labels"],
			"attention_mask": batch["attention_mask"],
		}
	
	def on_train_epoch_end(self):
		pass

	def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
		"""
		Compute metrics for the step
		"""
		
		# loss is batch loss, val_loss tracks it over the epoch
		loss, outputs = self.step(batch)

		# perplexity is the exponentiated NLL of a token given a sequence
		perplexity = torch.exp(outputs['clm_loss'].mean())

		# get shifted logits and labels for accuracy
		logits, labels, mask = self.get_shifted_labels(outputs["logits"], batch["input_ids"], batch["attention_mask"])
		accuracy = self.calculate_accuracy(logits, labels, mask)

		# calculate metrics (loss/accuracy)
		self.val_loss(loss)
		self.val_clm_loss(outputs['clm_loss'])
		self.val_prosody_loss(outputs['prosody_loss'])

		# log metrics
		self.val_perplexity(perplexity)
		self.val_accuracy(accuracy)

		self.val_mae(outputs["preds"], batch["tokenized_labels"], batch["loss_mask"])
		self.val_r2(outputs["preds"], batch["tokenized_labels"], batch["loss_mask"])
		self.val_pearson(outputs["preds"], batch["tokenized_labels"], batch["loss_mask"])

		# log metrics per epoch
		self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("val/clm_loss", self.val_clm_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("val/prosody_loss", self.val_prosody_loss, on_step=False, on_epoch=True, prog_bar=True)

		# clm metrics
		self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
		self.log("val/perplexity", self.val_perplexity, on_step=False, on_epoch=True, prog_bar=True)

		# prosody metrics
		self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
		self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
		self.log("val/pearson", self.val_pearson, on_step=False, on_epoch=True, prog_bar=True)

		return {
			"loss": loss,
			"preds": outputs["preds"],
			"targets": batch["tokenized_labels"],
			"attention_mask": batch["attention_mask"],
		}

	def on_validation_epoch_end(self):
		"""
		Aggregate metrics over the epoch and log
		"""
		
		# calculate metrics over the epoch
		loss = self.val_loss.compute()
		clm_loss = self.val_clm_loss.compute()
		prosody_loss = self.val_prosody_loss.compute()
		
		mae = self.val_mae.compute()
		r2 = self.val_r2.compute()
		pearson = self.val_pearson.compute()
		perplexity = self.val_perplexity.compute()
		accuracy = self.val_accuracy.compute()

		# add to min/max metric and determine best over the epoch
		self.val_loss_best(loss)
		self.val_clm_loss_best(clm_loss)
		self.val_prosody_loss_best(prosody_loss)

		self.val_perplexity_best(perplexity)
		self.val_accuracy_best(accuracy)
		
		self.val_mae_best(mae)
		self.val_r2_best(r2)
		self.val_pearson_best(pearson)

		# add to the logger
		self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
		self.log("val/clm_loss_best", self.val_clm_loss_best.compute(), prog_bar=True)
		self.log("val/prosody_loss_best", self.val_prosody_loss_best.compute(), prog_bar=True)

		self.log("val/perplexity_best", self.val_perplexity_best.compute(), prog_bar=True)
		self.log("val/accuracy_best", self.val_accuracy_best.compute(), prog_bar=True)

		self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)
		self.log("val/r2_best", self.val_r2_best.compute(), prog_bar=True)
		self.log("val/pearson_best", self.val_pearson_best.compute(), prog_bar=True)
	
	# def save_outputs(self, batch, items)

	# 	for item in items:
	# 		np.save(

	# 			f"{self.save_path}/predictions/"

	# 		)
	# 		batch[item]

	def test_step(
		self, batch: Dict[str, torch.tensor], batch_idx: int, verbose: bool = True
	):
		# loss is batch loss, test_loss tracks it over the epoch
		loss, outputs = self.step(batch)

		# perplexity is the exponentiated NLL of a token given a sequence
		perplexity = torch.exp(outputs['clm_loss'].mean())

		# get shifted logits and labels for accuracy
		logits, labels, mask = self.get_shifted_labels(outputs["logits"], batch["input_ids"], batch["attention_mask"])
		accuracy = self.calculate_accuracy(logits, labels, mask)

		# calculate metrics (loss/accuracy)
		self.test_loss(loss)
		self.test_clm_loss(outputs['clm_loss'])
		self.test_prosody_loss(outputs['prosody_loss'])

		# log metrics per epoch
		self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test/clm_loss", self.test_clm_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test/prosody_loss", self.test_prosody_loss, on_step=False, on_epoch=True, prog_bar=True)
		
		# log metrics
		self.test_perplexity(perplexity)
		self.test_accuracy(accuracy)

		self.test_mae(outputs["preds"], batch["tokenized_labels"], batch["loss_mask"])
		self.test_r2(outputs["preds"], batch["tokenized_labels"], batch["loss_mask"])
		self.test_pearson(outputs["preds"], batch["tokenized_labels"], batch["loss_mask"])

		self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test/pearson", self.test_pearson, on_step=False, on_epoch=True, prog_bar=True)
		self.log("test/perplexity", self.test_perplexity, on_step=False, on_epoch=True, prog_bar=True)

		##########################
		##### Save predictions ###
		##########################
	
		return {
			"loss": loss,
			"preds": outputs["preds"],
			"targets": batch["tokenized_labels"],
			"attention_mask": batch["attention_mask"],
			"loss_mask": batch["loss_mask"],
			"input_ids": batch["input_ids"],
			"input_text": batch["input_text"],
			"original_labels": batch["original_labels"],
		}

	def on_test_epoch_end(self):
		pass
	
	def on_epoch_end(self):
		# reset metrics at the end of every epoch!
		self.train_loss.reset()
		self.val_loss.reset()
		self.test_loss.reset()

		# clm loss
		self.train_clm_loss.reset()
		self.val_clm_loss.reset()
		self.test_clm_loss.reset()

		# prosody loss
		self.train_prosody_loss.reset()
		self.val_prosody_loss.reset()
		self.test_prosody_loss.reset()

		# mae
		self.train_mae.reset()
		self.test_mae.reset()
		self.val_mae.reset()

		# perplexity
		self.train_perplexity.reset()
		self.val_perplexity.reset()
		self.test_perplexity.reset()

		# accuracy
		self.train_accuracy.reset()
		self.val_accuracy.reset()
		self.test_accuracy.reset()

		# r2
		self.test_r2.reset()
		self.val_r2.reset()

		# pearson r
		self.val_pearson.reset()
		self.test_pearson.reset()

	@property
	def total_training_steps(self) -> int:
		"""Total training steps inferred from datamodule and devices."""
		if (
			isinstance(self.trainer.limit_train_batches, int)
			and self.trainer.limit_train_batches != 0
		):
			dataset_size = self.trainer.limit_train_batches
		elif isinstance(self.trainer.limit_train_batches, float):
			# limit_train_batches is a percentage of batches
			dataset_size = len(self.trainer.datamodule.train_dataloader())
			dataset_size = int(dataset_size * self.trainer.limit_train_batches)
		else:
			dataset_size = len(self.trainer.datamodule.train_dataloader())
	
		num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
		if self.trainer.tpu_cores:
			num_devices = max(num_devices, self.trainer.tpu_cores)
	
		effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
		max_estimated_steps = (
			dataset_size // effective_batch_size
		) * self.trainer.max_epochs
	
		if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
			return self.trainer.max_steps
		return max_estimated_steps
	
	def configure_optimizers(self):
		"""Choose what optimizers and learning-rate schedulers to use in your optimization.
		Normally you'd need one. But in the case of GANs or similar you might have multiple.
	
		Examples:
			https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
		"""
		optimizer = self.hparams.optimizer(params=self.parameters())
		if self.hparams.scheduler is not None:
			scheduler = self.hparams.scheduler(optimizer=optimizer)
			return {
				"optimizer": optimizer,
				"lr_scheduler": {
					"scheduler": scheduler,
					"monitor": "val/loss",
					"interval": "epoch",
					"frequency": 1,
				},
			}
		return {"optimizer": optimizer}