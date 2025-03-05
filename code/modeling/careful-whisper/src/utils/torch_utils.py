import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gamma import Gamma

########################################
###### Causal modeling functions #######
########################################

# def hf_to_pt(mask):
#     return torch.where(mask == 1, 0.0, float('-inf'))

# def pt_to_hf(mask):
#     return (mask != float('-inf')).int()

# def convert_attn_mask_type(mask, mask_type='hf'):

#     assert (mask_type in ['hf', 'pt'])

#     if mask_type == 'hf':
#         return hf_to_pt(mask)
#     elif mask_type == 'pt':
#         return pt_to_hf(mask)
#     else:
#         raise ValueError(f'mask_type must specify either hf or pt (huggingface or pytorch)')

def get_shifted_labels(labels, logits, mask):
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

    return shifted_labels, shifted_logits, shifted_mask

def clm_loss(labels, logits, mask, loss_fn=nn.CrossEntropyLoss(reduction='none')):
    '''
    Causal language modeling loss --> cross-entropy over each predicted token and 
    the ground truth token
    '''

    # # Mask has to be one of huggingface or pytorch
    # assert (mask_type in ['hf', 'pt'])

    labels, logits, mask = get_shifted_labels(
        labels=labels,
        logits=logits,
        mask=mask
    )

    # Flatten the tokens and compute loss only over non-masked tokens
    loss = masked_loss(
        labels=labels,
        predictions=logits,
        mask=mask,
        loss_fn=loss_fn,
    )

    return loss

def calculate_accuracy(labels: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels) * mask.bool()

    correct = torch.sum(correct)
    total = torch.sum(mask)

    return correct.float() / total

########################################
###### Specific loss functions  ########
########################################

class MLPGaussianRegressor(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        input_size: int = 768,
        hidden_size: int = 128,
        num_labels: int = 1,
        dropout_probability: int = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.Dropout(dropout_probability))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(
            num_layers - 2
        ):  # -2 because input and output layers are separate
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.Dropout(dropout_probability))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_size, 2 * num_labels))

        print(f"Initialized MLP Gaussian Regressor")
        print_num_trainable_params(self, model_name="MLP Gaussian Regressor")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLPRegressor(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        input_size: int = 768,
        hidden_size: int = 128,
        num_labels: int = 1,
        dropout_probability: int = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.Dropout(dropout_probability))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(
            num_layers - 2
        ):  # -2 because input and output layers are separate
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.Dropout(dropout_probability))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_labels))

        print(f"Initialized MLP Regressor")
        print_num_trainable_params(self, model_name="MLP Regressor")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def freeze_model(model):
    for i, param in model.named_parameters():
        param.requires_grad = False
    return model

def unfreeze_model(model):
    for i, param in model.named_parameters():
        param.requires_grad = True
    return model

def print_num_trainable_params(model, model_name="model"):
    """
    Print the number of trainable parameters in a PyTorch Lightning module.

    Parameters:
    - model: A PyTorch Lightning module.

    Returns: None
    """
    # use .parameters() function to get all model parameters
    # use .requires_grad attribute to check if the parameter is trainable
    # use .nelement() function to get the number of elements in a parameter tensor
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The {model_name} has {trainable_params} trainable parameters.")
    return trainable_params


def build_regressor(regressor, hidden_size, num_labels):
    if regressor == "MLP":
        model = MLPRegressor(hidden_size, num_labels)
    else:
        raise ValueError(f"Unsupported regressor type {regressor}")
    return model


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: tensor_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(value) for value in obj]
    else:
        return obj


def masked_loss(labels, predictions, mask, loss_fn=nn.MSELoss(reduction="none")):
    """
    Compute the masked loss for given labels, predictions and mask.

    :param labels: Tensor containing the ground truth labels
    :param predictions: Tensor containing the predicted labels
    :param mask: Tensor containing the mask to apply on the loss
    :param loss_function: PyTorch loss function to compute the loss (default: nn.MSELoss(reduction="none"))

    :return: Masked loss
    """
    # Compute the element-wise loss
    # print(f"shapes {labels.shape}, {predictions.shape}")
    # print(predictions)
    loss = loss_fn(predictions, labels)

    # Apply the mask to the loss
    masked_loss = loss * mask

    # Compute the mean of the masked loss
    masked_loss_mean = torch.sum(masked_loss) / torch.sum(mask)

    return masked_loss_mean


def masked_GNLLL(
    input,
    target,
    var,
    mask,
    loss_fn=nn.GaussianNLLLoss(full=True, reduction="none"),
):
    """
    Args:
        input: expectation of the Gaussian distribution. (mu)
        target: sample from the Gaussian distribution.
        var: (sigma**2) tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
    :return: Mean Reduced Masked loss
    """
    # Compute the element-wise loss
    loss = loss_fn(input, target, var)
    masked_loss = loss * mask
    # Compute the mean of the masked loss
    masked_loss_mean = torch.sum(masked_loss) / torch.sum(mask)

    return masked_loss_mean

def masked_gamma_loss(
    mu,
    var,
    target,
    mask,
):
    dist = Gamma(mu, var)

    target = target * mask + 1e-4  # add small constant for numerical stability
    nll = -dist.log_prob(target)

    masked_nll = nll * mask
    masked_loss_mean = masked_nll.sum() / mask.sum()
    
    return masked_loss_mean


class SELU_Range(nn.Module):
    def __init__(self, alpha=1.67326, scale=1.0507):
        """
        SELU activation function with a default range of [0, 10].
        """
        super(SELU_Range, self).__init__()
        self.alpha = alpha
        self.scale = scale

    def forward(self, x):
        return self.scale * F.selu(x, self.alpha) + 5.0


class SELU_Learnable(nn.Module):
    """
    SELU activation function with a learnable range.
    """

    def __init__(self):
        super(SELU_Learnable, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * F.selu(x, self.alpha) + 5.0


class Custom_Range_Activation(nn.Module):
    """
    Custom activation function with a range of [0, 10].
    """

    def __init__(self):
        super(Custom_Range_Activation, self).__init__()

    def forward(self, x):
        return 10.0 * (1.0 / (1.0 + torch.exp(-x)))


class ScaledSigmoid(nn.Module):
    """
    Sigmoid activation function with a fixed range output.
    """

    def __init__(self, lower=0, upper=10):
        super(ScaledSigmoid, self).__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return self.lower + (self.upper - self.lower) * torch.sigmoid(x)
