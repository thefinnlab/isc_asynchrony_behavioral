import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
import numpy as np
import numbers
from typing import List, Tuple
from omegaconf import DictConfig, OmegaConf

def audio_text_collator(batch, pad_token=-999):
    """
    Collates and pads a batch of audio-text data.
    
    Args:
        batch: List of dictionaries containing audio and text features
        pad_token: Padding token value for text tokens
        
    Returns:
        A dictionary of padded batch elements
    """
    # Extract texts and get all keys
    texts = [item["text"] for item in batch]
    keys = list(batch[0].keys())
    
    # Unpack all items from batch into separate lists
    store = {k: [] for k in keys}
    for item in batch:
        for k in keys:
            # Remove batch dimension from padding if needed
            value = item[k].squeeze(0) if hasattr(item[k], 'size') and item[k].size(0) > 1 else item[k]
            store[k].append(value)
    
    # Create result dictionary
    result = {"text": texts}
    
    # Pad text tokens and attention masks
    result["text_tokens"] = pad_sequence(store['text_tokens'], batch_first=True, padding_value=pad_token)
    result["attention_mask"] = pad_sequence(store['attention_mask'], batch_first=True, padding_value=0)
    
    # Pad prominence and boundary, adding an extra dimension
    result["prominence"] = pad_sequence(store['prominence'], batch_first=True, padding_value=0).unsqueeze(-1)
    result["boundary"] = pad_sequence(store['boundary'], batch_first=True, padding_value=0).unsqueeze(-1)
    
    # Convert attention mask to same dtype as other tensors
    result["attention_mask"] = result["attention_mask"].to(store['audio_features'][0].dtype)
    
    # Pad feature tensors (audio and optionally video)
    result["audio_features"] = pad_feature_tensor(store['audio_features'])
    
    if 'video_features' in keys:
        result["video_features"] = pad_feature_tensor(store['video_features'])

    if 'audiovisual_features' in keys:
        result["audiovisual_features"] = pad_feature_tensor(store['audiovisual_features'])
    
    return result

def pad_feature_tensor(features):
    """
    Pads a list of feature tensors to the maximum length in the batch.
    
    Args:
        features: List of feature tensors
        
    Returns:
        Padded tensor of shape (batch_size, max_length, embed_dim)
    """
    max_length = max(feat.shape[0] for feat in features)
    embed_dim = features[0].shape[1]
    padded_tensor = torch.zeros(len(features), max_length, embed_dim)
    
    for i, feat in enumerate(features):
        padded_tensor[i, :feat.shape[0], :] = feat
    
    return padded_tensor


def encode_and_pad_batch(
    batch: List[Tuple[str, List[numbers.Number]]],
    tokenizer,
    model_name: str,
    label_pad_value=-999,
):
    """
    Encode a batch of sentences and their corresponding token labels, and pad both.
    :param batch: list of tuples (sentence, labels)
    :param tokenizer: tokenizer object
    :param max_length: maximum length for padding (optional)
    :return: input_ids, attention_masks, padded_labels
    """
    sentences, labels_batch = zip(*batch)

    if model_name == "gpt2":
        encoded_batch = tokenizer.batch_encode_plus(
            sentences,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
    elif model_name == "bert-base-uncased":
        encoded_batch = tokenizer.batch_encode_plus(
            sentences,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
    else:
        raise ValueError("Model not supported")

    input_ids = encoded_batch["input_ids"]
    attention_masks = encoded_batch["attention_mask"]

    # print(f"Batch Encoding:\n", encoded_batch)

    # print(f"Batch Encoding:")
    # print(
    #     f"Shapes of input_ids, attention_masks: {input_ids.shape}, {attention_masks.shape}"
    # )
    # Pad the labels
    max_length = input_ids.size(1)
    padded_labels_batch = []
    for label_sequence in labels_batch:
        padded_label_sequence = label_sequence + [label_pad_value] * (
            max_length - len(label_sequence)
        )
        padded_labels_batch.append(padded_label_sequence)
        # print(len(padded_label_sequence))

    padded_labels_tensor = torch.tensor(padded_labels_batch)
    # print(
    #     f"Shapes of input_ids, attention_masks, padded_labels_tensor: {input_ids.shape}, {attention_masks.shape}, {padded_labels_tensor.shape}"
    # )
    # return dict with keys "input_ids", "attention_mask", "labels"
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": padded_labels_tensor,
    }


def vector_collate_fn(batch, eos_token_id, invalid_label=-1):
    """
    Same as the collate_fn below, but for vectors instead of scalars as labels
    """
    max_len = max([len(item["input_ids"]) for item in batch])
    vector_size = len(batch[0]["tokenized_labels"][0])
    padding_tensor = torch.ones(vector_size) * invalid_label
    # print(f"vector size: {vector_size}")

    padded_token_ids = torch.full((len(batch), max_len), eos_token_id, dtype=torch.long)
    padded_tokenized_labels = torch.full(
        (len(batch), max_len, vector_size), invalid_label, dtype=torch.float32
    )
    padded_loss_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    # print(f"shapes of padded_tokenized_labels: {padded_tokenized_labels.shape}")

    for i, item in enumerate(batch):
        # print(f"input_ids: {item['input_ids']}")
        # print(f"tokenized labels: {item['tokenized_labels']}")
        # print(f"loss mask: {item['loss_mask']}")
        token_ids = torch.tensor(item["input_ids"])
        tokenized_labels = torch.tensor(np.array(item["tokenized_labels"]))
        loss_mask = torch.tensor(item["loss_mask"])

        padded_token_ids[i, : len(token_ids)] = token_ids
        padded_tokenized_labels[i, : len(tokenized_labels), :] = tokenized_labels
        padded_loss_mask[i, : len(loss_mask)] = loss_mask
        attention_mask[i, : len(token_ids)] = 1

    return {
        "input_text": [item["input_text"] for item in batch],
        "tokenized_text": [item["tokenized_text"] for item in batch],
        "original_labels": [item["original_labels"] for item in batch],
        "tokenized_labels": padded_tokenized_labels,
        "input_ids": padded_token_ids,
        "loss_mask": padded_loss_mask,
        "attention_mask": attention_mask,
        "word_to_tokens": [item["word_to_tokens"] for item in batch],
    }


def collate_fn(batch, eos_token_id, invalid_label=-1):
    max_len = max([len(item["input_ids"]) for item in batch])

    padded_token_ids = torch.full((len(batch), max_len), eos_token_id, dtype=torch.long)
    padded_tokenized_labels = torch.full(
        (len(batch), max_len), invalid_label, dtype=torch.float32
    )
    padded_loss_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        # print(f"input_ids: {item['input_ids']}")
        # print(f"tokenized labels: {item['tokenized_labels']}")
        # print(f"loss mask: {item['loss_mask']}")
        token_ids = torch.tensor(item["input_ids"])
        tokenized_labels = torch.tensor(item["tokenized_labels"])
        loss_mask = torch.tensor(item["loss_mask"])

        padded_token_ids[i, : len(token_ids)] = token_ids
        padded_tokenized_labels[i, : len(tokenized_labels)] = tokenized_labels
        padded_loss_mask[i, : len(loss_mask)] = loss_mask
        attention_mask[i, : len(token_ids)] = 1

    return {
        "input_text": [item["input_text"] for item in batch],
        "tokenized_text": [item["tokenized_text"] for item in batch],
        "original_labels": [item["original_labels"] for item in batch],
        "tokenized_labels": padded_tokenized_labels,
        "input_ids": padded_token_ids,
        "loss_mask": padded_loss_mask,
        "attention_mask": attention_mask,
        "word_to_tokens": [item["word_to_tokens"] for item in batch],
    }


def rnn_collate_fn(batch, pad_value=0):
    sequences = [item for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=pad_value
    )
    mask = (padded_sequences != pad_value).float()  # Assuming 0 is your padding value
    # unsqueeze(2) adds a dimension for the features (1 feature for F0)
    return padded_sequences.unsqueeze(2), lengths, mask.unsqueeze(2)
    # return padded_sequences, lengths, mask
