import os
import inspect
from typing import Any, Dict, List, Tuple, DefaultDict
import pickle

import torch
from lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import MinMetric, MaxMetric
from torch.distributions.gamma import Gamma
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup

from peft import get_peft_model, LoraConfig, TaskType

from src.utils.torch_utils import (
	masked_loss, masked_GNLLL, masked_gamma_loss,
	print_num_trainable_params, freeze_model, unfreeze_model
)
from src.utils.torch_metrics import (
	MaskedMeanAbsoluteError, MaskedR2Score, MaskedPearsonCorrCoef,
	MeanMetric, MaskedAccuracy
)

class ProsodyCausalLM(LightningModule):
	def __init__(
			self,
			model_name: str,
			optimizer: torch.optim.Optimizer = AdamW,
			scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
			output_activation: nn.Module = nn.Identity(),
			num_labels: int = 1,
			p_dropout: float = 0.1,
			loss_mode: str = 'joint',
			lora_config: Dict = None,
			pretrained: bool = True,
			use_prosody_embeddings: bool = True,
			tie_prosody_embeddings: bool = True,
			freeze_kwargs: DefaultDict = {
				'freeze_lm': False,
				'unfreeze_after': -1
			},
			loss_kwargs: DefaultDict = {
				'w_prosody': 1,
				'w_clm': 1
			}
		):
			super().__init__()

			self.save_hyperparameters(logger=False)

			# initialize the model
			print("Loading Huggingface model.")

			if pretrained:
				print (f'Loading pretrained model')
				self.model = AutoModelForCausalLM.from_pretrained(model_name)
			else:
				print (f'Initializing new model')
				self.config = AutoConfig.from_pretrained(model_name)
				self.model = AutoModelForCausalLM.from_config(self.config)

			self.freeze_kwargs = freeze_kwargs

			if self.freeze_kwargs['freeze_lm']:
				self.model = freeze_model(self.model)

			# create embedding layer for prosody
			self.use_prosody_embeddings = use_prosody_embeddings
			self.prosody_embed = nn.Linear(num_labels, self.model.config.hidden_size, bias=False)

			# projection from hidden state to prosody value
			self.dropout = nn.Dropout(p_dropout)
			self.regressor = nn.Linear(self.model.config.hidden_size, num_labels*2, bias=False) # mu and variance for each label
			self.output_activation = output_activation

			# initialize weights according to initialization rules of model
			self.model._init_weights(self.prosody_embed)
			self.model._init_weights(self.regressor)

			# # ties the weights of the prosody embeddings to that of the regressor
			if tie_prosody_embeddings:
				# tie decoder weights to the encoder
				self.regressor.weight.data[num_labels-1:, :] = self.prosody_embed.weight.data.transpose(0, 1)

			if lora_config:
				config = LoraConfig(**lora_config)
				self.model = get_peft_model(self.model, config)
				print (self.model.print_trainable_parameters())

			self._init_metrics()

			# Validate and set the loss mode
			valid_modes = ['clm', 'prosody', 'joint']

			if loss_mode not in valid_modes:
				raise ValueError(f"Invalid loss_mode. Expected one of: {valid_modes}")
			else:
				print (f'Using {loss_mode} loss')

			self.loss_mode = loss_mode
			self.loss_kwargs = loss_kwargs

			# # Set requires_grad based on loss mode
			# self._set_requires_grad()

	def _init_metrics(self):
		"""Initialize all metrics used in the model."""
		# use separate metric instance for train, val and test step
		# to ensure a proper reduction over the epoch
		metric_classes = {
			'loss': MeanMetric,
			'clm_loss': MeanMetric,
			'prosody_loss': MeanMetric,
			'mae': MaskedMeanAbsoluteError,
			'perplexity': MeanMetric,
			'accuracy': MeanMetric,
			'r2': MaskedR2Score,
			'pearson': MaskedPearsonCorrCoef
		}

		for name, metric_class in metric_classes.items():
			for prefix in ['train', 'val', 'test']:
				setattr(self, f'{prefix}_{name}', metric_class())

		# for logging best so far validation metrics
		best_metrics = ['val_loss', 'val_clm_loss', 'val_prosody_loss', 'val_accuracy', 
						'val_perplexity', 'val_mae', 'val_r2', 'val_pearson']

		for metric in best_metrics:
			setattr(self, f'{metric}_best', MinMetric() if 'loss' in metric or 'perplexity' in metric else MaxMetric())

	# def _set_requires_grad(self):
	# 	# Turn off gradients for prosody-related parameters if not using prosody loss
	# 	prosody_params = [self.prosody_embed, self.regressor]
	# 	for param in prosody_params:
	# 		param.requires_grad = (self.loss_mode in ['prosody', 'joint'])

	#################################
	####### Helper methods ##########
	#################################

	def get_input_embeddings(self, input_ids, prosody_values, mask):

		# get token embedding and project prominence to embedding space
		input_embeds = self.model.transformer.wte(input_ids)

		# if self.loss_mode in ['prosody', 'joint']:
		if self.use_prosody_embeddings:
			prosody_embed = self.prosody_embed(prosody_values.unsqueeze(-1))
			prosody_embed = prosody_embed * mask.unsqueeze(-1)

			# add prosody embeddings to imput embeddings
			input_embeds = input_embeds + prosody_embed

		return input_embeds

	def get_shifted_clm_labels(self, logits, labels, mask):
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

	def get_shifted_prosody_labels(self, mu, var, labels, mask):
		'''
		Each first mu corresponds to second label --> we want to predict
		prosody at timestep N+1 from embedding N
		'''

		# shift the mu/var to be in line with the labels
		shifted_mu = mu[..., :-1].contiguous().view(-1)
		shifted_var = var[..., :-1].contiguous().view(-1)

		# shift the loss mask to be in line
		shifted_labels = labels[..., 1:].contiguous().view(-1)
		shifted_mask = mask[..., 1:].contiguous().view(-1)

		return shifted_mu, shifted_var, shifted_labels, shifted_mask

	def calculate_accuracy(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
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

		logits, labels, mask = self.get_shifted_clm_labels(
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
		# # prosody labels
		# labels = batch['tokenized_labels']
		# # dist = outputs['dist']
		# mu = outputs['mu']
		# var = outputs['var'].squeeze(-1)

		# loss_mask = batch["loss_mask"]  # ignore padded sequence in loss

		mu, var, labels, mask = self.get_shifted_prosody_labels(
			mu=outputs['mu'],
			var=outputs['var'],
			labels=batch['tokenized_labels'],
			mask=batch['loss_mask'],
		)

		loss = masked_gamma_loss(
			mu=mu,
			var=var,
			target=labels,
			mask=mask
		)
		
		return loss

	#################################
	####### Lightning methods #######
	#################################

	def forward(self, batch: Dict[str, torch.tensor], eps=1e-4):

		# get token embedding and project prominence to embedding space
		input_embeds = self.get_input_embeddings(
			input_ids=batch['input_ids'], 
			prosody_values=batch['tokenized_labels'],
			mask=batch['loss_mask']
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
		mu, var = torch.chunk(preds, chunks=2, dim=-1) # split last dimension into mu and var
		var = F.softplus(var) + eps # ensure positivity of var + add a small constant for numerical stability

		# have to squeeze the last dimension due to chunking
		if self.output_activation is not None:
			mu = self.output_activation(mu.squeeze(-1))

		# Gamma distribution with concentration mu and rate var
		mu = F.softplus(mu).squeeze(-1)
		var = var.squeeze(-1)

		dist = Gamma(mu, var.squeeze(-1))
		preds = dist.mean

		# return predictions 
		return {
			'logits': logits,
			# 'dist': dist,
			# 'preds': preds,
			'preds': preds,
			# 'preds': mu, 
			'mu': mu,
			'var': var,
		}

	def step(self, batch: Dict[str, torch.tensor]):

		# make forward pass
		outputs = self(batch)

		clm_loss = self.clm_loss(batch, outputs)
		prosody_loss = self.prosody_loss(batch, outputs)

		# calculate loss based on the selected mode
		if self.loss_mode == 'clm':
			loss = clm_loss
		elif self.loss_mode == 'prosody':
			loss = prosody_loss
		else:  # 'both'
			loss = self.loss_kwargs['w_clm'] * clm_loss + self.loss_kwargs['w_prosody'] * prosody_loss

		outputs.update({
			'loss': loss,
			'clm_loss': clm_loss,
			'prosody_loss': prosody_loss
		})

		return loss, outputs

	def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
		return self._shared_step(batch, 'train')

	def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
		return self._shared_step(batch, 'val')

	def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
		return self._shared_step(batch, 'test')

	def _shared_step(self, batch: Dict[str, torch.tensor], stage: str):
		loss, outputs = self.step(batch)

		metrics = self._calculate_metrics(outputs, batch, stage)
		self._log_metrics(metrics, stage)

		return {
			"loss": loss,
			"preds": outputs["preds"],
			"targets": batch["tokenized_labels"],
			"attention_mask": batch["attention_mask"],
		}

	def _calculate_metrics(self, outputs, batch, stage):

		# perplexity is the exponentiated NLL of a token given a sequence
		perplexity = torch.exp(outputs['clm_loss'].mean())

		# get shifted logits and labels for accuracy
		logits, labels, mask = self.get_shifted_clm_labels(outputs["logits"], batch["input_ids"], batch["attention_mask"])

		# shift mu/var of prosody predictions
		mu, var, prosody_labels, prosody_mask = self.get_shifted_prosody_labels(
			mu=outputs['mu'],
			var=outputs['var'],
			labels=batch['tokenized_labels'],
			mask=batch['loss_mask'],
		)

		dist = Gamma(mu, var.squeeze(-1))
		mu = dist.mean

		accuracy = self.calculate_accuracy(logits, labels, mask)

		metrics = {
			'loss': outputs['clm_loss'] + outputs['prosody_loss'],
			'clm_loss': outputs['clm_loss'],
			'prosody_loss': outputs['prosody_loss'],
			'perplexity': perplexity,
			'accuracy': accuracy,
			'mae': (mu, prosody_labels, prosody_mask),
			'pearson': (mu, prosody_labels, prosody_mask),
			'r2': (mu, prosody_labels, prosody_mask),
		}

		return metrics

	def _log_metrics(self, metrics, stage):
		for name, value in metrics.items():
			metric = getattr(self, f'{stage}_{name}')
			if isinstance(value, tuple):
				metric(*value)
			else:
				metric(value)
			self.log(f"{stage}/{name}", metric, on_step=stage=='train', on_epoch=True, prog_bar=True)

	def on_train_epoch_end(self):
		self._shared_epoch_end('train')

	def on_validation_epoch_end(self):
		self._shared_epoch_end('val')

	def on_test_epoch_end(self):
		self._shared_epoch_end('test')

	def _shared_epoch_end(self, stage):

		metrics = ['loss', 'clm_loss', 'prosody_loss', 'perplexity', 'accuracy', 'mae', 'r2', 'pearson']

		if stage == 'val':
			for metric in metrics:
				value = getattr(self, f'val_{metric}').compute()
				best_metric = getattr(self, f'val_{metric}_best')
				best_metric(value)
				self.log(f"val/{metric}_best", best_metric.compute(), prog_bar=True)


		# for metric in metrics:
		# 	_metric = getattr(self, f'{stage}_{metric}')
		# 	_metric.reset()

		print (f'Finished epoch {self.current_epoch}')

		if stage =='val' and (self.freeze_kwargs['unfreeze_after'] == self.current_epoch):
			print (f'Unfreezing model at epoch {self.current_epoch}')
			# unfreeze the network, switch to other form of loss, use the prosody embedding
			self.model = unfreeze_model(self.model)
			self.use_prosody_embeddings = True
			self.loss_mode = 'joint'

	# # def on_epoch_end(self):
	# def on_validation_epoch_end(self):
	# 	"""Reset all metrics at the end of every epoch."""
	# 	for metric in self.metrics():
	# 		metric.reset()

	# 	print (f'Finished epoch {self.current_epoch}')

	# 	if self.freeze_kwargs['unfreeze_after'] == self.current_epoch:
	# 		print (f'Unfreezing model at epoch {self.current_epoch}')
	# 		# unfreeze the network, switch to other form of loss, use the prosody embedding
	# 		self.model = unfreeze_model(self.model)
	# 		self.use_prosody_embeddings = True
	# 		self.loss_mode = 'joint'

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