#!/usr/bin/env python
# coding: utf-8

# In[51]:


import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import math
import numpy as np

from pathlib import Path
import shutil
import os
import tempfile
import json

from typing import Tuple, List, Callable, Dict

BASE_CHECKPOINT_DIR = "checkpoints"
BASE_CHECKPOINT_DATA_DIR = "games"
SAVE_DIR_SYNT = "./.data_alpha_tensor/synthetic_data"
SAVE_COB_DIR = "./.data_alpha_tensor/cob_matrices"


# # Neural net
# ## Torso a.k.a. the transformer
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html  

# In[52]:


class PositionEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return x


# In[53]:


class AttentionDenseBlock(torch.nn.Module):
    def __init__(self, inner_size: int, multiplier: int = 4):
        super().__init__()
        self.norm = torch.nn.LayerNorm(inner_size)
        self.linear = torch.nn.Linear(inner_size, inner_size * multiplier)
        self.activation = torch.nn.GELU()
        self.linear_final = torch.nn.Linear(
            inner_size * multiplier, inner_size
        )

    def forward(self, x: torch.Tensor):
        x_temp = self.activation(self.linear(self.norm(x)))
        return x + self.linear_final(x_temp)


# In[54]:


class AttentionHead(torch.nn.Module):
    def __init__(self, x_size: int, y_size: int, proj_dim: int):
        # x_size = N_x
        # y_size = N_y
        super(AttentionHead, self).__init__()
        self.proj_dim_isqrt = 1 / torch.sqrt(torch.tensor(proj_dim))
        self.queries_proj_layer = torch.nn.Linear(x_size, proj_dim)
        self.keys_proj_layer = torch.nn.Linear(y_size, proj_dim)
        self.values_proj_layer = torch.nn.Linear(y_size, proj_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: bool = False):
        queries = self.queries_proj_layer(x)
        keys = self.keys_proj_layer(y)
        values = self.values_proj_layer(y)
        attention = F.softmax(
            torch.matmul(queries, keys.transpose(-2, -1))
            * self.proj_dim_isqrt,
            dim=-1,
        )
        if mask:
            attention = torch.triu(attention, diagonal=1)
        output = torch.matmul(attention, values)
        return output


# In[55]:


class AlphaMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        proj_dim: int = 32,
        n_heads: int = 16,
        multiplier: int = 4,
    ):
        # x_dim = size of the last dimension of x
        # y_dim = size of the last dimension of y
        super().__init__()
        self.norm_layer_x = torch.nn.LayerNorm(x_dim)
        self.norm_layer_y = torch.nn.LayerNorm(y_dim)
        self.module_list = torch.nn.ModuleList([AttentionHead(x_dim, y_dim, proj_dim) for _ in range(n_heads)])
        self.linear = torch.nn.Linear(n_heads * proj_dim, x_dim)

        self.dense = AttentionDenseBlock(x_dim, multiplier)

    def forward(self, x: torch.nn.Module, y: torch.nn.Module, mask: bool = False):
        # x.size = (Nx, c1), y.size = (Ny, c2)
        x_norm = self.norm_layer_x(x)
        y_norm = self.norm_layer_y(y)
        temp = torch.cat([layer(x_norm, y_norm, mask) for layer in self.module_list], dim=-1)
        x = x + self.linear(temp)
        return self.dense(x)


# ## Policy

# In[56]:


class PolicyHeadDoubleAttention(torch.nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_heads: int,
        n_feat: int,
        emb_size: int,
        emb_dim: int,
    ):
        super().__init__()
        d_model = n_feat * n_heads
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.attention1 = AlphaMultiHeadAttention(d_model, d_model)
        self.drop1 = torch.nn.Dropout()
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.attention2 = AlphaMultiHeadAttention(d_model, emb_dim)
        self.drop2 = torch.nn.Dropout()

    def forward(self, x: torch.Tensor, e: torch.Tensor):
        x = self.layer_norm1(x)
        c = self.attention1(x, x, mask=True)
        c = self.drop1(c)
        x = x + c
        x = self.layer_norm2(x)
        c = self.attention2(x, e, mask=False)
        c = self.drop2(c)
        x = x + c
        return x


# In[57]:


class PolicyHeadCore(torch.nn.Module):
    def __init__(
        self,
        emb_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_feat: int = 32,
        n_heads: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_logits, n_feat * n_heads)
        self.position_encoding = PositionEncoding(n_feat * n_heads)
        self.decoders = torch.nn.ModuleList(
            [
                PolicyHeadDoubleAttention(
                    n_steps, n_heads, n_feat, emb_size, emb_dim
                )
                for _ in range(n_layers)
            ]
        )
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_feat * n_heads, n_logits)

    def forward(self, a: torch.Tensor, e: torch.Tensor):
        x = self.position_encoding(self.embedding(a))
        for layer in self.decoders:
            x = layer(x, e)
        o = self.linear2(self.relu(x))
        return o, x


# In[58]:


def sample_from_logits(a):
    # returns a sampled element and the associated probability
    # since cross entropy is run during training we expect logits
    # to be probabilities yet.
    probs = torch.cumsum(F.softmax(a, dim=-1), dim=-1)
    random_vals = torch.rand(probs.shape[0]).unsqueeze(-1).to(a.device)
    n_classes = a.shape[-1]
    new_a_idx = torch.argmax(1.0 * (probs > random_vals), dim=-1)
    index_bias = torch.arange(0, len(new_a_idx)).to(a.device) * n_classes
    probs = torch.take(probs, new_a_idx + index_bias)
    # new_a = F.one_hot(new_a_idx, n_classes)
    return new_a_idx, probs

class PolicyHead(torch.nn.Module):
    def __init__(
        self,
        emb_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
    ):
        super().__init__()
        self.n_logits = n_logits
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.core = PolicyHeadCore(emb_size, emb_dim, n_steps, n_logits)

    def _train_forward(self, e: torch.Tensor, g: torch.Tensor):
        # e is the embedding, shape = (N, m, c)
        # g represents the previous actions, when training it represents the
        # list of correct actions, thus we need to shift them (since we do not
        # want to consider also the latest, correct action when predicting).
        # g has shape (N, N_steps) and it is a one-hot encoding of N_logits
        g = torch.roll(g, shifts=-1, dims=1)
        # the first raw will have attention zero during training
        # g = F.one_hot(g, self.n_logits).float()
        o, z = self.core(g, e)
        return o, z[:, 0]

    def _eval_forward(self, e: torch.Tensor):
        bs = e.shape[0]
        future_g = (
            torch.zeros((bs, self.n_samples, self.n_steps)).long().to(e.device)
        )
        ps = torch.ones((bs, self.n_samples)).to(e.device)
        e = e.unsqueeze(1).repeat(1, self.n_samples, 1, 1)

        future_g = future_g.view(-1, self.n_steps)
        ps = ps.view(-1)
        e = e.view(-1, e.shape[-2], e.shape[-1])
        for i in range(self.n_steps):
            o_s, z_s = self.core(future_g[:, : i + 1], e)
            future_g[:, i], p_i = sample_from_logits(o_s[:, i])
            ps *= p_i
        future_g = future_g.view(bs, self.n_samples, self.n_steps)
        ps = ps.view(bs, self.n_samples)
        return (
            future_g,
            ps,
            z_s[:, 0].view(bs, self.n_samples, *z_s.shape[2:]).mean(1),
        )

    def forward(self, e: torch.Tensor, g: torch.Tensor = None):
        if g is None:
            return self._eval_forward(e)
        return self._train_forward(e, g)


# ## Value
# Value head is a multilayer perceptron

# In[59]:


class ValueHeadCore(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.linear(x))


# In[60]:


class ValueHead(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int = 512, output_size: int = 8
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *(
                [ValueHeadCore(input_size, hidden_size)]
                + [ValueHeadCore(hidden_size, hidden_size)] * 2
            )
        )
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        return self.linear(self.layers(x))


# In[61]:


class AttentionDenseBlock(torch.nn.Module):
    def __init__(self, inner_size: int, multiplier: int = 4):
        super().__init__()
        self.norm = torch.nn.LayerNorm(inner_size)
        self.linear = torch.nn.Linear(inner_size, inner_size * multiplier)
        self.activation = torch.nn.GELU()
        self.linear_final = torch.nn.Linear(
            inner_size * multiplier, inner_size
        )

    def forward(self, x: torch.Tensor):
        x_temp = self.activation(self.linear(self.norm(x)))
        return x + self.linear_final(x_temp)


# In[62]:


class TorsoAttentiveModes(torch.nn.Module):
    def __init__(self, input_dim: int):
        # input_dim = c
        super().__init__()
        self.attention = AlphaMultiHeadAttention(
            input_dim,
            input_dim,
        )

    def forward(self, x1, x2, x3):
        # x1.size = x2.size = x3.size = (N, S, S, c)
        # where N is the batch size
        size = x1.shape[-2]
        input_list = [x1, x2, x3]
        for m1, m2 in [(0, 1), (2, 0), (1, 2)]:
            matrix = torch.cat([input_list[m1], input_list[m2]], dim=-2)
            # matrix_size = (N, S, 2S, c)
            out = self.attention(matrix, matrix)
            input_list[m1] = out[:, :, :size]
            input_list[m2] = out[:, :, size:]
        return input_list


# In[63]:


class TorsoModel(torch.nn.Module):
    """Torso model of OpenAlphaTensor.

    It maps an input tensor of shape (N, T, S, S, S) to (N, 3S*S, c), where:

        N is the batch size;
        T is the context size (size of the history + 1);
        S is the number of elements in each matrix to be multiplied;
        c is the output dimensionality.
    """

    def __init__(
        self,
        scalars_size: int,
        input_size: int,
        tensor_length: int,
        out_size: int,
    ):
        # scalar_size = s
        # input_size = S
        # tensor_length = T
        # out_size = c
        super(TorsoModel, self).__init__()
        self.linears_1 = torch.nn.ModuleList(
            [
                torch.nn.Linear(scalars_size, input_size * input_size)
                for _ in range(3)
            ]
        )
        self.linears_2 = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_size * tensor_length + 1, out_size)
                for _ in range(3)
            ]
        )
        self.attentive_modes = torch.nn.ModuleList(
            [TorsoAttentiveModes(out_size) for _ in range(4)] # the paper has 8 attentive modes, but it takes a long time to train
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor):
        # x.size = (N, T, S, S, S)
        # scalars.size = (N, s)
        batch_size = x.shape[0]
        S = x.shape[-1]
        T = x.shape[1]
        x1 = x.permute(0, 2, 3, 4, 1).reshape(batch_size, S, S, S * T)
        x2 = x.permute(0, 4, 2, 3, 1).reshape(batch_size, S, S, S * T)
        x3 = x.permute(0, 3, 4, 2, 1).reshape(batch_size, S, S, S * T)
        input_list = [x1, x2, x3]
        for i in range(3):
            temp = self.linears_1[i](scalars).reshape(batch_size, S, S, 1)
            input_list[i] = torch.cat([input_list[i], temp], dim=-1)
            input_list[i] = self.linears_2[i](input_list[i])
        x1, x2, x3 = input_list
        for layer in self.attentive_modes:
            x1, x2, x3 = layer(x1, x2, x3)
        return torch.stack([x1, x2, x3], dim=2).reshape(
            batch_size, 3 * S * S, -1
        )


# In[64]:


class QuantileLoss(torch.nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber_loss = torch.nn.HuberLoss(reduction="none", delta=delta)

    def forward(self, q: torch.Tensor, g: torch.Tensor):
        n = q.shape[-1]
        tau = torch.arange(0, n).unsqueeze(0).to(q.device) / n
        h = self.huber_loss(g, q)
        k = torch.abs(tau - (g - q > 0).float())
        return torch.mean(h * k)


class ValueRiskManagement(torch.nn.Module):
    def __init__(self, u_q: float = 0.75):
        super(ValueRiskManagement, self).__init__()
        self.u_q = u_q

    def forward(self, q: torch.Tensor):
        # q shape = (N, n)
        j = int(self.u_q * q.shape[-1])
        return torch.mean(q[:, j:], dim=-1)


# In[65]:


class AlphaTensorModel(torch.nn.Module):
    def __init__(
        self,
        tensor_length: int,
        input_size: int,
        scalars_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
    ):
        # scalar_size = s
        # input_size = S
        # tensor_length = T
        # emb_dim = c
        super().__init__()
        self.tensor_length = tensor_length
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.torso = TorsoModel(
            scalars_size, input_size, tensor_length, emb_dim
        )
        emb_size = 3 * input_size * input_size
        print("Build policy head")
        self.policy_head = PolicyHead(
            emb_size, emb_dim, n_steps, n_logits, n_samples
        )
        print("Build value head")
        self.value_head = ValueHead(
            256 
        )  # value dependent on num_head and proj_dim
        self.policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.quantile_loss_fn = QuantileLoss()
        self.risk_value_management = ValueRiskManagement()

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        g_action: torch.Tensor,
        g_value: torch.Tensor,
    ):
        # shapes
        # x = (N, T, S, S, S)
        # s = (N, s)
        # g_action = (N, N_steps)
        # g_value = (N, )
        e = self.torso(x, s)
        o, z1 = self.policy_head(e, g_action)
        l_policy = self.policy_loss_fn(
            o.reshape(-1, o.shape[-1]), g_action.reshape(-1)
        )
        q = self.value_head(z1)
        l_value = self.quantile_loss_fn(q, g_value.float())
        return l_policy, l_value

    def _eval_forward(self, x: torch.Tensor, s: torch.Tensor):
        e = self.torso(x, s)
        a, p, z1 = self.policy_head(e)
        q = self.value_head(z1)
        q = self.risk_value_management(q)
        return a, p, q

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        g_action: torch.Tensor = None,
        g_value: torch.Tensor = None,
    ):
        if g_action is None:
            return self._eval_forward(x, s)
        else:
            assert g_value is not None
            return self._train_forward(x, s, g_action, g_value)

    @property
    def n_logits(self):
        return self.policy_head.n_logits

    @property
    def n_steps(self):
        return self.policy_head.n_steps

    @property
    def n_samples(self):
        return self.policy_head.n_samples


# # Some utility funcs

# In[66]:


def get_scalars(input_tensor: torch.Tensor, t_step: int, with_bs: bool = True):
    """Adds the time step to the current state tensor.

    Args:
        input_tensor (torch.Tensor): Current state tensor.
        t_step (int): Current time step.
        with_bs (bool, optional): Whether the batch size is present in the
        input tensor.
    """
    # scalars containing the iteration time
    if with_bs:
        bs = input_tensor.shape[0]
        scalars = torch.zeros((bs, 1))
        scalars[:, 0] = t_step
    else:
        scalars = torch.tensor(t_step).unsqueeze(-1).float()
    return scalars


def map_triplet_to_action(
    triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    base: int,
    n_steps: int,
    add_bias: bool = True,
):
    """Maps a triplet of tensors to an action.

    Args:
        triplet (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Triplet of
        tensors u, v, and w.
        base (int): Base used for the conversion.
        n_steps (int): Number of steps in the action.
        add_bias (bool, optional): Whether to add a bias to the action.
    """
    # map the triplet to an action. First, we concatenate the three tensors and
    # then we convert it to an action using the given base representation. Each
    # element is converted using the formula:
    #   action += element * base^(element_index)
    u, v, w = triplet
    n_dim = u.ndim
    action = torch.cat((u, v, w), dim=-1)
    action = action.reshape(-1, n_steps, action.shape[-1] // n_steps)
    if n_dim == 1:
        action = action.squeeze(0)
    if add_bias:
        action = action + base // 2
    action = action * torch.tensor(
        [base**i for i in range(action.shape[-1])]
    )
    action = action.sum(dim=-1)
    return action


# In[67]:


# @torch.jit.script
def _single_action_to_triplet(
    action_val: int,
    basis: int,
    out_dim: int,
    bias: int,
    device: str,
):
    """Converts an action to the original triplet (u, v, w) that generated it.

    Args:
        action_val (int): Action to convert.
        basis (int): Basis used for the conversion.
        out_dim (int): Output dimension.
        bias (int): Bias to subtract from the action.
        device (str): Name of the torch device to use.
    """
    triplet = torch.zeros(out_dim).to(device)
    if action_val > 0:
        idx = int(
            torch.log(torch.tensor(action_val))
            // torch.log(torch.tensor(basis))
        )
    else:
        idx = 0
    while idx >= 0:
        temp = int(basis**idx)
        triplet[idx] = action_val // temp - bias
        action_val = action_val - temp
        idx -= 1
    return triplet


# In[68]:


def map_action_to_triplet(
    action_tensor: torch.Tensor,
    cardinality: int = 5,
    vector_size: int = 5,
    add_bias: bool = True,
):
    """Maps a batch of actions to the batch of triplets that generated them.

    Args:
        action_tensor (torch.Tensor): Batch of actions.
        cardinality (int, optional): Cardinality of the action space.
        vector_size (int, optional): Size of the vector.
        add_bias (bool, optional): Whether to use bias.
    """
    # map the action to a triplet. The action is converted to a base 5
    # representation and then the three elements are extracted from it.
    # The action has shape (bs, n_steps) and it contains the token for
    # recreating u, v and w. The token is a number between 0 and n_logits.
    action_shape = action_tensor.shape
    action_tensor = action_tensor.reshape(-1)
    if add_bias:
        bias = cardinality // 2
    else:
        bias = 0
    triplets = torch.stack(
        [
            _single_action_to_triplet(
                action_tensor[idx],
                cardinality,
                vector_size,
                bias,
                action_tensor.device,
            )
            for idx in range(len(action_tensor))
        ]
    )
    final_size = triplets.shape[-1]
    return triplets.reshape((*action_shape, final_size))


# In[69]:


def generate_synthetic_data(
    tensor_size: int,
    n_data: int,
    limit_rank: int,
    prob_distr: Callable = torch.randn,
    random_seed: int = None,
):
    """Generates synthetic demonstrations.

    Args:
        tensor_size (int): Size of the tensor.
        n_data (int): Number of demonstrations.
        limit_rank (int): Limit rank of each tensor.
        prob_distr (Callable, optional): Distribution of the entries of the
        tensor.
        random_seed (int, optional): Random seed for reproducibility.
    """
    if random_seed is not None:
        torch.random.manual_seed(random_seed)
    for _ in range(n_data):
        # rank = torch.randint(low=1, high=limit_rank + 1, size=(1,)).item()
        rank = limit_rank
        output_tensor = torch.zeros(tensor_size, tensor_size, tensor_size)
        list_of_triplets = []
        for i in range(rank):
            valid_triplet = False
            while not valid_triplet:
                u = prob_distr(tensor_size)
                v = prob_distr(tensor_size)
                w = prob_distr(tensor_size)
                generated_tensor = (
                    u.reshape(-1, 1, 1)
                    * v.reshape(1, -1, 1)
                    * w.reshape(1, 1, -1)
                )
                if not (generated_tensor == 0).all():
                    valid_triplet = True
                    list_of_triplets.append((u, v, w))
                    output_tensor += generated_tensor
        yield output_tensor, list_of_triplets


# # Datasets
# 

# In[70]:


import tqdm
from torch.utils.data import DataLoader


# In[71]:


def compute_move(triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Computes the outer product of the three tensors in the triplet that
    will be subtracted from the current state.

    Args:
        triplets (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tensors u,
        v, and w.
    """
    u, v, w = triplets
    return u.reshape(-1, 1, 1) * v.reshape(1, -1, 1) * w.reshape(1, 1, -1)


# In[72]:


class SyntheticDataBuffer(Dataset):
    """Dataset of synthetically generated demonstrations."""

    def __init__(
        self,
        tensor_size,
        n_data,
        limit_rank,
        prob_distr,
        n_prev_actions: int,
        device: str,
        n_steps: int,
        random_seed=None,
    ):
        """Builds a dataset of synthetic demonstrations.

        Args:
            tensor_size (int): Size of the tensor.
            n_data (int): Number of demonstrations to generate.
            limit_rank (int): Maximum rank of the generated tensors.
            prob_distr (Callable): Probability distribution to use to generate
            the tensors.
            n_prev_actions (int): Number of previous actions to use as input.
            device (str): Name of the torch device to use.
            n_steps (int): Number of steps to perform in the environment.
            random_seed (int, optional): Random seed to use.
        """
        self.device = device
        self.len_data = 0
        self.n_prev_actions = n_prev_actions
        self.limit_rank = limit_rank
        self.n_steps = n_steps
        self.save_dir = os.path.join(SAVE_DIR_SYNT, f"size_{tensor_size}")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        number_of_triplets = len(list(Path(self.save_dir).glob("*.pt"))) // 2
        if number_of_triplets < n_data:
            self.len_data = number_of_triplets
            for i, (output_tensor, list_of_triplets) in enumerate(
                generate_synthetic_data(
                    tensor_size,
                    n_data - number_of_triplets,
                    limit_rank,
                    prob_distr,
                    random_seed,
                )
            ):
                torch.save(
                    output_tensor,
                    os.path.join(
                        self.save_dir, f"output_tensor_{self.len_data}.pt"
                    ),
                )
                torch.save(
                    list_of_triplets,
                    os.path.join(
                        self.save_dir, f"list_of_triplets_{self.len_data}.pt"
                    ),
                )
                self.len_data += 1
        else:
            self.len_data = n_data

    def __len__(self):
        return self.len_data * self.limit_rank

    @torch.no_grad()
    def __getitem__(self, idx):
        i = idx // self.limit_rank
        j = idx % self.limit_rank
        output_tensor = torch.load(
            os.path.join(self.save_dir, f"output_tensor_{i}.pt")
        )
        list_of_triplets = torch.load(
            os.path.join(self.save_dir, f"list_of_triplets_{i}.pt")
        )
        if j != self.limit_rank - 1:
            moves = list_of_triplets[j + 1 :]  # noqa E203
            output_tensor = self._apply_moves(output_tensor, moves)
        triplet = list_of_triplets[j]
        output_tensor = torch.stack(
            [
                output_tensor,
                *(
                    compute_move(t)
                    for t in reversed(
                        list_of_triplets[
                            j + 1 : j + 1 + self.n_prev_actions  # noqa E203
                        ]
                    )
                ),
            ]
        )
        if len(output_tensor) < self.n_prev_actions + 1:
            output_tensor = torch.cat(
                [
                    output_tensor,
                    torch.zeros(
                        self.n_prev_actions + 1 - len(output_tensor),
                        *output_tensor.shape[1:],
                    ),
                ]
            )
        policy = map_triplet_to_action(triplet, base=5, n_steps=self.n_steps)
        reward = torch.tensor([-(j + 1)])
        scalar = get_scalars(output_tensor, self.limit_rank - j, with_bs=False)
        return (
            output_tensor.to(self.device),
            scalar.to(self.device),
            policy.to(self.device),
            reward.to(self.device),
        )

    @staticmethod
    def _apply_moves(
        tensor: torch.Tensor,
        moves: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ):
        """Given an initial state and a list of moves, applies the moves to
        the state.

        Args:
            tensor (torch.Tensor): Initial state.
            moves (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
            List of moves.
        """
        for u, v, w in moves:
            tensor = tensor - u.reshape(-1, 1, 1) * v.reshape(
                1, -1, 1
            ) * w.reshape(1, 1, -1)
        return tensor


# In[73]:


class GameDataBuffer(Dataset):
    """Buffer to store the data from the games played by the MCTS agent."""

    def __init__(self, device: str, max_buffer_size: int):
        """Initializes the buffer.

        Args:
            device (str): Name of the torch device to use.
            max_buffer_size (int): Maximum size of the buffer.
        """
        self.num_games = 0
        self.temp_dir = tempfile.mkdtemp("game_data_buffer")
        self.game_data = {}
        self.max_buffer_size = max_buffer_size
        self.device = device

    def __del__(self):
        shutil.rmtree(self.temp_dir)

    def add_game(
        self,
        states: List[torch.Tensor],
        policies: List[torch.Tensor],
        rewards: List[torch.Tensor],
    ):
        """Adds a played game to the buffer.

        Args:
            states (List[torch.Tensor]): Observed game states.
            policies (List[torch.Tensor]): List of policies.
            rewards (List[torch.Tensor]): Observed rewards.
        """
        self.game_data[self.num_games] = len(states)
        torch.save(
            states, os.path.join(self.temp_dir, f"states_{self.num_games}.pt")
        )
        torch.save(
            policies,
            os.path.join(self.temp_dir, f"policies_{self.num_games}.pt"),
        )
        torch.save(
            rewards,
            os.path.join(self.temp_dir, f"rewards_{self.num_games}.pt"),
        )
        self.num_games += 1
        if self.num_games >= self.max_buffer_size:
            # remove oldest game. Note that this line is not thread safe. Lock
            # should be added if multiple threads are used.
            self.num_games = 0

    def __len__(self):
        return sum(self.game_data.values())

    @torch.no_grad()
    def __getitem__(self, idx):
        i = 0
        while idx >= self.game_data[i]:
            idx -= self.game_data[i]
            i += 1
        states = torch.load(os.path.join(self.temp_dir, f"states_{i}.pt"))
        policies = torch.load(os.path.join(self.temp_dir, f"policies_{i}.pt"))
        rewards = torch.load(os.path.join(self.temp_dir, f"rewards_{i}.pt"))
        return (
            states[idx].to(self.device),
            get_scalars(states[idx], idx, with_bs=False).to(self.device),
            policies[idx].to(self.device).argmax(dim=-1),
            rewards[idx].to(self.device).reshape(1),
        )

    def save_game_data(self, path: str):
        """Copy save_dir content in path and save game_data
        in json format
        """
        shutil.copytree(self.temp_dir, path, dirs_exist_ok=True)
        with open(os.path.join(path, "game_data.json"), "w") as f:
            json.dump(self.game_data, f)

    def load_game_data(self, path: str):
        """Load game_data from json format and copy content
        in save_dir
        """
        with open(os.path.join(path, "game_data.json"), "r") as f:
            self.game_data = json.load(f)
        shutil.copytree(path, self.temp_dir)
        self.num_games = len(self.game_data)


# In[74]:


class TensorGameDataset(Dataset):
    """Dataset to be used for training the AlphaTensor algorithm using both
    actor generated and synthetic data. A basis change can be applied to both
    the data type with a probability specified in the constructor. The
    synthetic data and the actor generated one are stored in two data buffers.
    """

    def __init__(
        self,
        len_data,
        pct_synth,
        tensor_size,
        n_synth_data,
        limit_rank,
        prob_distr,
        action_memory_len: int,
        device: str,
        n_steps: int,
        random_seed=None,
    ):
        self.synthetic_data_buffer = SyntheticDataBuffer(
            tensor_size,
            n_synth_data,
            limit_rank,
            prob_distr,
            action_memory_len,
            n_steps=n_steps,
            device=device,
            random_seed=random_seed,
        )
        self.game_data_buffer = GameDataBuffer(
            device=device, max_buffer_size=100000
        )
        self.best_game_data_buffer = GameDataBuffer(
            device=device, max_buffer_size=1000
        )
        self.len_data = len_data
        self.pct_synth = pct_synth
        self.pct_best_game = 0
        self.synth_bool = torch.ones(len_data, dtype=torch.bool)
        self.synth_idx = torch.from_numpy(
            np.random.choice(
                len(self.synthetic_data_buffer), len_data, \
                replace=True
                # replace=False

            )
        )
        self.game_idx = None
        self.best_game_idx = None
        self.action_memory_len = action_memory_len
        self.tensor_size = tensor_size
        self.device = device

    def change_training_split(self, pct_synth, pct_best_game):
        self.pct_synth = pct_synth
        self.pct_best_game = pct_best_game

    def recompute_synthetic_indexes(self):
        if len(self.game_data_buffer) > 0:
            self.synth_bool = torch.rand(self.len_data) < self.pct_synth
            len_synth_data = self.synth_bool.sum().item()
            self.synth_idx = torch.from_numpy(
                np.random.choice(
                    len(self.synthetic_data_buffer),
                    len_synth_data,
                    # replace=False,
                    replace=True
                )
            )
            if len(self.best_game_data_buffer) > 0 and self.pct_best_game > 0:
                len_game_data = int(
                    (1 - self.pct_synth - self.pct_best_game) * self.len_data
                )
                replace_game = len_game_data > len(self.game_data_buffer)
                len_best_game_data = (
                    self.len_data - len_synth_data - len_game_data
                )
                replace_best_game = len_best_game_data > len(
                    self.best_game_data_buffer
                )
                self.game_idx = torch.from_numpy(
                    np.random.choice(
                        len(self.game_data_buffer),
                        len_game_data,
                        replace=replace_game,
                    )
                )
                self.best_game_idx = torch.from_numpy(
                    np.random.choice(
                        len(self.best_game_data_buffer),
                        len_best_game_data,
                        replace=replace_best_game,
                    )
                )
            else:
                len_game_data = self.len_data - len_synth_data
                replace_game = len_game_data > len(self.game_data_buffer)
                self.game_idx = torch.from_numpy(
                    np.random.choice(
                        len(self.game_data_buffer),
                        len_game_data,
                        replace=replace_game,
                    )
                )

    def __getitem__(self, idx):
        if self.synth_bool[idx]:
            return self.synthetic_data_buffer[
                self.synth_idx[self.synth_bool[:idx].sum()]
            ]
        else:
            if self.pct_best_game > 0 and self.best_game_idx is not None:
                if idx - self.synth_bool[:idx].sum() < len(self.best_game_idx):
                    return self.best_game_data_buffer[
                        self.best_game_idx[idx - self.synth_bool[:idx].sum()]
                    ]
                else:
                    return self.game_data_buffer[
                        self.game_idx[
                            idx
                            - self.synth_bool[:idx].sum()
                            - len(self.best_game_idx)
                        ]
                    ]
            else:
                return self.game_data_buffer[
                    self.game_idx[idx - self.synth_bool[:idx].sum()]
                ]

    def __len__(self):
        return self.len_data

    def add_game(
        self,
        states: List[torch.Tensor],
        policies: List[torch.Tensor],
        rewards: List[torch.Tensor],
    ):
        self.game_data_buffer.add_game(states, policies, rewards)

    def add_best_game(
        self,
        states: List[torch.Tensor],
        policies: List[torch.Tensor],
        rewards: List[torch.Tensor],
    ):
        self.best_game_data_buffer.add_game(states, policies, rewards)

    def save_game_data(self, path):
        self.game_data_buffer.save_game_data(os.path.join(path, "game_data"))
        self.best_game_data_buffer.save_game_data(
            os.path.join(path, "best_game_data")
        )

    def load_game_data(self, path):
        self.game_data_buffer.load_game_data(os.path.join(path, "game_data"))
        self.best_game_data_buffer.load_game_data(
            os.path.join(path, "best_game_data")
        )

    @property
    def input_tensor(self) -> torch.Tensor:
        max_matrix_size = int(np.sqrt(self.tensor_size))
        input_tensor = torch.zeros(
            1,
            self.action_memory_len + 1,
            self.tensor_size,
            self.tensor_size,
            self.tensor_size,
        )
        matrix_dims = (
            torch.randint(1, max_matrix_size, (3,))
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        operation_tensor = self._build_tensor_game_input(
            *matrix_dims, action_memory_len=self.action_memory_len
        )

        input_tensor[
            0,
            :,
            : operation_tensor.shape[1],
            : operation_tensor.shape[2],
            : operation_tensor.shape[3],
        ] = operation_tensor
        return input_tensor.to(self.device)

    @staticmethod
    def _build_tensor_game_input(
        dim_1: int, dim_k: int, dim_2: int, action_memory_len: int
    ):
        """Build the input tensor for the game. The input tensor has shape
        (action_memory_len+1, matrix_size**2, matrix_size**2, matrix_size**2).
        The first slice represent the matrix multiplication tensor which will
        be reduced by the TensorGame algorithm. The other slices represent the
        action memory.
        """
        input_tensor = torch.zeros(
            action_memory_len + 1, dim_1 * dim_k, dim_k * dim_2, dim_1 * dim_2
        )
        for r in range(dim_1 * dim_2):
            for k in range(dim_k):
                input_tensor[
                    0, (r // dim_2) * dim_k + k, k * dim_2 + r % dim_2, r
                ] = 1
        return input_tensor


# In[75]:


def f_prob_distribution(size):
    """Samples a tensor of values from a distribution with a peak at 0 and a
    tail at -2 and 2.

    Args:
        size (int): Number of values to sample.
    """
    f_vals = torch.tensor([-2, -1, 0, 1, 2])
    f_probs = torch.tensor([0.001, 0.099, 0.8, 0.099, 0.001]).unsqueeze(0)
    f_cum_sum = torch.cumsum(f_probs, dim=-1)
    unif_prob = torch.rand((size, 1))
    tensor_idx = torch.argmax((unif_prob <= f_cum_sum).int(), dim=1)
    tensor = f_vals[tensor_idx]
    return tensor


# In[76]:


def get_change_basis_matrix(
    tensor_size: int,
    n_cob: int,
    entry_distribution: Callable = torch.randn,
    random_seed: int = None,
):
    """Generate a list of change of basis matrices.

    Args:
        tensor_size (int): Size of the tensor.
        n_cob (int): Number of change of basis matrices.
        entry_distribution (Callable, optional): Distribution of the entries
        of the change of basis matrices.
        random_seed (int, optional): Random seed for reproducibility.
    """
    if random_seed is not None:
        torch.random.manual_seed(random_seed)
    for _ in range(n_cob):
        diag_p = 2 * (torch.rand(tensor_size) > 0.5).float() - 1
        diag_l = 2 * (torch.rand(tensor_size) > 0.5).float() - 1
        random_matrix = entry_distribution((tensor_size, tensor_size))
        p_matrix = torch.diag(diag_p)
        l_matrix = torch.diag(diag_l)
        p_matrix = p_matrix + torch.triu(random_matrix, diagonal=1)
        l_matrix = l_matrix + torch.tril(random_matrix, diagonal=-1)
        yield torch.matmul(p_matrix, l_matrix)


def cob_entry_prob_distribution(size):
    full_size = int(np.prod(size))
    vals = torch.tensor([-1, 0, 1])
    probs = torch.tensor([0.0075, 0.985, 0.0075]).unsqueeze(0)
    cum_sum = torch.cumsum(probs, dim=-1)
    unif_prob = torch.rand((full_size, 1))
    tensor_idx = torch.argmax((unif_prob <= cum_sum).int(), dim=1)
    tensor = vals[tensor_idx]
    return tensor.reshape(size)


class ChangeOfBasis:
    def __init__(
        self,
        tensor_size: int,
        n_cob: int,
        cob_prob: float,
        device: str,
        random_seed: int = None,
    ):
        """
        Args:
            tensor_size (int): Size of the tensor.
            n_cob (int): Number of change of basis matrices.
            cob_prob (float): Probability of applying a change of basis.
            device (str): Name of the torch device to use.
            random_seed (int, optional): Random seed for reproducibility.
        """
        self.tmp_dir = Path(SAVE_COB_DIR)
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        for i, cob_matrix in enumerate(
            get_change_basis_matrix(
                tensor_size, n_cob, cob_entry_prob_distribution, random_seed
            )
        ):
            torch.save(cob_matrix, f"{self.tmp_dir}/cob_matrix_{i}.pt")
        self.tensor_size = tensor_size
        self.n_cob = n_cob
        self.cob_prob = cob_prob
        self.device = device

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor, return_basis: bool = False):
        """Apply a change of basis to a tensor.

        Args:
            tensor (torch.Tensor): Tensor to apply the change of basis to.
            return_basis (bool, optional): Whether to return the change of
            basis matrix as well.
        """
        cob_prob = torch.rand(1).item()
        if cob_prob > self.cob_prob:
            return tensor
        random_cob = torch.randint(low=0, high=self.n_cob, size=(1,))
        cob_matrix = torch.load(
            f"{self.tmp_dir}/cob_matrix_{int(random_cob)}.pt"
        ).to(self.device)

        # apply change of basis to each tensor dimension
        inner_tensor = tensor[0, 0]
        tensor_size = inner_tensor.shape[-1]
        original_shape = inner_tensor.shape
        cob_matrix = cob_matrix.transpose(0, 1)
        inner_tensor = torch.matmul(
            inner_tensor.reshape(-1, tensor_size), cob_matrix
        ).reshape(original_shape)
        inner_tensor = inner_tensor.permute(0, 2, 1)
        inner_tensor = torch.matmul(
            inner_tensor.reshape(-1, tensor_size), cob_matrix
        ).reshape(original_shape)
        inner_tensor = inner_tensor.permute(2, 1, 0)
        inner_tensor = torch.matmul(
            inner_tensor.reshape(-1, tensor_size), cob_matrix
        ).reshape(original_shape)
        inner_tensor = inner_tensor.permute(2, 0, 1)
        tensor[0, 0] = inner_tensor
        if return_basis:
            return tensor, cob_matrix.transpose(0, 1)
        return tensor


# # MCTS

# In[77]:


def extract_present_state(state: torch.Tensor) -> torch.Tensor:
    return state[:, 0]


# In[78]:


def to_hash(tensor: torch.Tensor) -> str:
    """Converts a tensor to a hash string.

    Args:
        tensor: The tensor to convert.
    """
    hashable_tensor = "_".join(
        tensor.reshape(-1).long().detach().cpu().numpy().astype(str).tolist()
    )
    return hashable_tensor

def from_hash(hashable_tensor: str, shape: tuple) -> torch.Tensor:
    """Converts a hash string back to the original tensor.

    Args:
        hashable_tensor (str): The hash string.
        shape (tuple): The shape of the original tensor.
    """
    return torch.tensor([float(x) for x in hashable_tensor.split("_")]).resize(
        shape
    )

def record_action(tree_dict: Dict, state: str, action: str):
    """Record the action in the tree dictionary.

    Args:
        tree_dict (Dict): The tree dictionary.
        state (str): The state as a hash string.
        action (str): The action as a hash string.
    """
    if state in tree_dict:
        tree_dict[state].append(action)
    else:
        tree_dict[state] = [action]


# In[79]:


def _recompose_possible_states(reduced_memory_states_dict: Dict):
    """Recompose the possible states from the reduced memory states.

    Args:
        reduced_memory_states_dict (Dict): The reduced memory states.
    """
    final_states = reduced_memory_states_dict["final_states"]
    previous_actions = reduced_memory_states_dict["previous_actions"]
    possible_states = [
        torch.cat(
            [
                final_states[i],
                previous_actions,
            ],
            dim=1,
        )
        for i in range(len(final_states))
    ]
    return possible_states


# In[80]:


def select_future_state(
    possible_states: List[torch.Tensor],
    q_values: torch.Tensor,
    N_s_a: torch.Tensor,
    repetitions: Dict[int, list],
    c_1: float = 1.25,
    c_2: float = 19652,
    return_idx: bool = False,
) -> torch.Tensor:
    """Select the future state maximizing the upper confidence bound."""
    # q_values (1, K, 1)
    pi = torch.tensor(
        [
            len(repetitions[i])
            for i in range(len(possible_states))
            if i in repetitions
        ]
    ).to(q_values.device)
    if pi.shape[0] != N_s_a.shape[1]:
        print(pi)
        print(pi.shape, q_values.shape, N_s_a.shape)
        pi = pi[: N_s_a.shape[1]]
    ucb = q_values.reshape(-1) + pi * torch.sqrt(
        torch.sum(N_s_a) / (1 + N_s_a)
    ) * (c_1 + torch.log((torch.sum(N_s_a) + c_2 + 1) / c_2))
    if return_idx:
        return ucb.argmax()
    return possible_states[ucb.argmax()]


# In[81]:


def remove_duplicates(reducing_tensor: torch.Tensor):
    """Remove duplicates from a tensor.

    Args:
        reducing_tensor (torch.Tensor): The tensor to remove duplicates from.
    """
    # reducing tensor has shape (1, N_mc, S, S, S)
    n_mc = reducing_tensor.shape[1]
    indexes = []
    idx_map = {}
    for idx in range(n_mc):
        if len(indexes) == 0:
            indexes.append(idx)
            idx_map[idx] = []
        else:
            idx_tensor = reducing_tensor[:, idx]
            for index in indexes:
                if (reducing_tensor[:, index] - idx_tensor == 0).all():
                    idx_map[index].append(idx)
                    break
            else:
                indexes.append(idx)
                idx_map[idx] = []

    # idx_map = {i: len(v) for i, v in enumerate(idx_map.values())}
    old_idx_to_new_idx_map = {}
    for new_idx, (key, values) in enumerate(idx_map.items()):
        old_idx_to_new_idx_map[key] = new_idx
        for second_idx in values:
            old_idx_to_new_idx_map[second_idx] = new_idx
    return (
        reducing_tensor[:, indexes],
        old_idx_to_new_idx_map,
        idx_map,
        indexes,
    )


# In[82]:


def extract_children_states_from_actions(
    state: torch.Tensor,
    actions: torch.Tensor,
    vec_cardinality: int = 5,
):
    """Extract the children states from the actions.

    Args:
        state (torch.Tensor): The state of the game.
        actions (torch.Tensor): The actions to apply to the state.
        vec_cardinality (int, optional): The cardinality of the vectors.
    """
    # state (1, T, S, S, S)
    # actions (1, K, N_steps)
    # we assume actions to be with N_steps = 1,
    #  and N_logits = |F|^(3S/N_steps). Each action is then mapped in a
    #  unique way to a triplet (u, v, w) where each vector has size S.
    # vector cardinality represents the number of values it can take an entry
    #  of u, v or w.
    bs, k, n_steps = actions.shape[:3]
    len_token = 3 * state.shape[2] // n_steps
    actions = map_action_to_triplet(actions, vec_cardinality, len_token)
    actions = actions.reshape(bs, k, n_steps * len_token)
    vec_dim = state.shape[2]
    u = actions[:, :, :vec_dim].reshape(bs, k, vec_dim, 1, 1)
    v = actions[:, :, vec_dim : 2 * vec_dim].reshape(
        bs, k, 1, vec_dim, 1
    )
    w = actions[:, :, 2 * vec_dim :].reshape(bs, k, 1, 1, vec_dim)
    reducing_tensor = u * v * w
    (
        reducing_tensor,
        old_idx_to_new_idx,
        repetition_map,
        not_duplicate_indexes,
    ) = remove_duplicates(reducing_tensor)
    old_state = state[:, 0]
    new_state = old_state.unsqueeze(1) - reducing_tensor
    rolling_states = torch.roll(state, 1)[:, 2:]
    return (
        [
            torch.cat(
                [
                    new_state[:, i : i + 1],  # noqa E203
                    reducing_tensor[:, i : i + 1],  # noqa E203
                    rolling_states,
                ],
                dim=1,
            )
            for i in range(k)
        ],
        old_idx_to_new_idx,
        repetition_map,
        not_duplicate_indexes,
    )


# In[83]:


def _reduce_memory_consumption_before_storing(
    possible_states: List[torch.Tensor],
):
    """Reduce the memory consumption before storing the states.

    Args:
        possible_states (List[torch.Tensor]): The possible states.
    """
    final_states = [state[:, 0:2] for state in possible_states]
    previous_actions = possible_states[0][:, 2:]
    storing_dict = {
        "final_states": final_states,
        "previous_actions": previous_actions,
    }
    return storing_dict


# In[84]:


def game_is_finished(state):
    """Tells if the game is finished or not.

    Args:
        state (torch.Tensor): The state of the game.
    """
    # state size (1, S, S, S)
    return (state == 0).all()


# In[85]:


@torch.no_grad()
def simulate_game(
    model,
    state: torch.Tensor,
    t_time: int,
    max_steps: int,
    game_tree: Dict,
    states_dict: Dict,
    horizon: int = 5,
):
    """Simulates a game from a given state.

    Args:
        model: The model to use for the simulation.
        state (torch.Tensor): The initial state.
        t_time (int): The current time step.
        max_steps (int): The maximum number of steps to simulate.
        game_tree (Dict): The game tree.
        states_dict (Dict): The states dictionary.
        horizon (int): The horizon to use for the simulation.
    """
    idx = t_time
    max_steps = min(max_steps, t_time + horizon)
    state_hash = to_hash(extract_present_state(state))
    trajectory = []
    # selection
    while state_hash in game_tree:
        (
            possible_states_dict,
            old_idx_to_new_idx,
            repetition_map,
            N_s_a,
            q_values,
            actions,
        ) = states_dict[state_hash]
        possible_states = _recompose_possible_states(possible_states_dict)
        state_idx = select_future_state(
            possible_states, q_values, N_s_a, repetition_map, return_idx=True
        )
        trajectory.append((state_hash, state_idx))  # state_hash, action_idx
        future_state = extract_present_state(possible_states[state_idx])
        state = possible_states[state_idx]
        state_hash = to_hash(future_state)
        idx += 1

    # expansion
    if idx <= max_steps:
        trajectory.append((state_hash, None))
        if not game_is_finished(extract_present_state(state)):
            state = state.to(model.device)
            scalars = get_scalars(state, idx).to(state.device)
            actions, probs, q_values = model(state, scalars)
            (
                possible_states,
                cloned_idx_to_idx,
                repetitions,
                not_dupl_indexes,
            ) = extract_children_states_from_actions(
                state,
                actions,
            )
            not_dupl_actions = actions[:, not_dupl_indexes].to("cpu")
            not_dupl_q_values = torch.zeros(not_dupl_actions.shape[:-1]).to(
                "cpu"
            )
            N_s_a = torch.zeros_like(not_dupl_q_values).to("cpu")
            present_state = extract_present_state(state)
            states_dict[to_hash(present_state)] = (
                _reduce_memory_consumption_before_storing(possible_states),
                cloned_idx_to_idx,
                repetitions,
                N_s_a,
                not_dupl_q_values,
                not_dupl_actions,
            )
            game_tree[to_hash(present_state)] = [
                to_hash(extract_present_state(fut_state))
                for fut_state in possible_states
            ]
            leaf_q_value = q_values
    else:
        leaf_q_value = -int(torch.linalg.matrix_rank(state).sum())
    # backup
    backward_pass(trajectory, states_dict, leaf_q_value=leaf_q_value)


def backward_pass(trajectory, states_dict, leaf_q_value: torch.Tensor):
    """Backward pass of the montecarlo algorithm"""
    reward = 0
    for idx, (state, action_idx) in enumerate(reversed(trajectory)):
        if action_idx is None:  # leaf node
            reward += leaf_q_value
        else:
            (
                _,
                old_idx_to_new_idx,
                _,
                N_s_a,
                q_values,
                _,
            ) = states_dict[state]
            if isinstance(reward, torch.Tensor):
                reward = reward.to(q_values.device)
            action_idx = int(action_idx)
            if action_idx in old_idx_to_new_idx:
                not_dupl_index = old_idx_to_new_idx[int(action_idx)]
            else:
                not_dupl_index = action_idx
            reward -= 1
            q_values[:, not_dupl_index] = (
                N_s_a[:, not_dupl_index] * q_values[:, not_dupl_index] + reward
            ) / (N_s_a[:, not_dupl_index] + 1)
            N_s_a[:, not_dupl_index] += 1


# In[86]:


def monte_carlo_tree_search(
    model: torch.nn.Module,
    state: torch.Tensor,
    n_sim: int,
    t_time,
    n_steps: int,
    game_tree: Dict,
    state_dict: Dict,
):
    """Runs the monte carlo tree search algorithm.

    Args:
        model (torch.nn.Module): The model to use for the simulation.
        state (torch.Tensor): The initial state.
        n_sim (int): The number of simulations to run.
        t_time (int): The current time step.
        n_steps (int): The maximum number of steps to simulate.
        game_tree (Dict): The game tree.
        state_dict (Dict): The dictionary containing the states.
    """
    # Note that game tree is not the full tree, but just the one having as root
    #  the current node(state).
    # should we accept also previous updated trajectories for the current node?
    # is it something we should considering when deciding how many simulations
    # we should run? (I think yes)
    state_hash = to_hash(extract_present_state(state))
    if state_hash in state_dict:
        with torch.no_grad():
            N_s_a = state_dict[state_hash][3]
            n_sim -= int(N_s_a.sum())
            n_sim = max(n_sim, 0)

    for _ in range(n_sim):
        simulate_game(model, state, t_time, n_steps, game_tree, state_dict)
    # return next state
    possible_states_dict, _, repetitions, N_s_a, q_values, _ = state_dict[
        state_hash
    ]
    possible_states = _recompose_possible_states(possible_states_dict)
    next_state_idx = select_future_state(
        possible_states, q_values, N_s_a, repetitions, return_idx=True
    )
    next_state = possible_states[next_state_idx]
    return next_state


# In[87]:


@torch.no_grad() # not sure here
def compute_improved_policy(
    state_dict: Dict,
    states: List[str],
    model_n_steps: int,
    model_n_logits: int,
    N_bar: int,
):
    """Compute the improved policy given the state_dict, the list of states.
    The improved policy is computed as (N_s_aˆ(1/tau) / (N_s_aˆ(1/tau)).sum())
    where tau is (log(N_s_a.sum()) / log(N_bar))
    """
    policies = torch.zeros(len(states), model_n_steps, model_n_logits)
    N_bar = torch.tensor(N_bar)
    for idx, state in enumerate(states):
        N_s_a = state_dict[state][3]
        actions = state_dict[state][5]
        if N_s_a.sum() > N_bar:
            tau = (torch.log(N_s_a.sum()) / torch.log(N_bar)).item()
        else:
            tau = 1
        N_s_a = N_s_a ** (1 / tau)
        improved_policy = N_s_a / N_s_a.sum()
        for sample_id in range(actions.shape[1]):
            action_ids = actions[0, sample_id]
            for step_id, action_id in enumerate(action_ids):
                policies[idx, step_id, action_id] += improved_policy[
                    0, sample_id
                ]
    return policies


# In[88]:


def actor_prediction(
    model: AlphaTensorModel,
    input_tensor: torch.Tensor,
    maximum_rank: int,
    mc_n_sim: int,
    N_bar: int,
    return_actions: bool = False,
):
    """Runs the monte carlo tree search algorithm to obtain the next states,
    policies and rewards.

    Args:
        model (AlphaTensorModel): The model to use for the simulation.
        input_tensor (torch.Tensor): The initial state.
        maximum_rank (int): The maximum number of steps to simulate.
        mc_n_sim (int): The number of simulations to run.
        N_bar (int): The parameter used to compute the improved policy.
        return_actions (bool): If True, only actions are returned.
    """
    # input_tensor has shape (1, T, S, S, S)
    state = input_tensor
    rank = 0
    game_tree = {}
    state_dict = {}
    hash_states = []
    states = []
    while rank < maximum_rank:
        print(f"current rank is {rank}")
        states.append(state)
        hash_states.append(to_hash(extract_present_state(state)))
        state = monte_carlo_tree_search(
            model,
            state,
            mc_n_sim,
            rank,
            maximum_rank,
            game_tree,
            state_dict,
        )
        if game_is_finished(extract_present_state(state)):
            break
        rank += 1
    final_state = extract_present_state(state)
    policies = compute_improved_policy(
        state_dict, hash_states, model.n_steps, model.n_logits, N_bar
    )
    reward = (
        int(torch.linalg.matrix_rank(final_state).sum())
        if not game_is_finished(final_state)
        else 0
    )
    rewards = torch.cumsum(
        torch.tensor([-1] * (len(policies) - 1) + [reward]), dim=0
    )
    if return_actions:
        actions = [state_dict[hash_state][5] for hash_state in hash_states]
        return actions
    # policies do not have the batch size, but states still have it
    states = [s.squeeze(0) for s in states]
    return states, policies, rewards


# In[89]:


def swap_data(
    states: List[torch.Tensor],
    actions: List[torch.Tensor],
):
    """Swaps the last action with a random one and updates the states
    accordingly for a single game.

    Args:
        states (List[torch.Tensor]): All the states for a single game.
        actions (List[torch.Tensor]): All the actions through the game.
    """
    last_action = actions[-1]
    swap_index = torch.randint(0, len(states) - 1, (1,)).item()
    actions[-1] = actions[swap_index]
    actions[swap_index] = last_action

    actual_state = states[swap_index]
    for i in range(swap_index + 1, len(states) + 1):
        prev_action = actions[i - 1]
        triplet = map_action_to_triplet(prev_action, vector_size=actual_state.shape[-1])
        vector_size = actual_state.shape[-1] // 3
        bs = actual_state.shape[0]
        u = triplet[:, :vector_size].reshape(bs, -1, 1, 1)
        v = triplet[:, vector_size : 2 * vector_size].reshape(bs, 1, -1, 1)
        w = triplet[:, 2 * vector_size :].reshape(bs, 1, 1, -1)
        reduced_state = u * v * w
        fut_state = actual_state[:, 0] - reduced_state
        new_state = actual_state[:, 1:].roll(1, dims=1)
        new_state[:, 0] = reduced_state
        actual_state = torch.cat([fut_state, new_state], dim=1)
        states[i] = actual_state
    return states, actions


class Trainer:
    """Trainer for the AlphaTensor model. The trainer does not require an
    explicit loss since the loss is computed by the model itself. The trainer
    is responsible for both the training step and the acting one, storing
    acting performance in a buffer.
    """

    def __init__(
        self,
        model: AlphaTensorModel,
        tensor_size: int,
        n_steps: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        device: str,
        len_data: int,
        pct_synth: float,
        n_synth_data: int,
        limit_rank: int,
        n_cob: int,
        cob_prob: float,
        data_augmentation: bool,
        loss_params: Tuple[float, float] = None,
        random_seed: int = None,
        checkpoint_dir: str = None,
        checkpoint_data_dir: Path = None,
        extra_devices: List[str] = None,
    ):
        """Initializes the trainer.

        Args:
            model (AlphaTensorModel): The model to train.
            tensor_size (int): Flattened size of the matrices to be multiplied.
            n_steps (int): Number of steps used to get a single action out of
            a triplet.
            batch_size (int): Batch size.
            optimizer (torch.optim.Optimizer): The optimizer used to train the
            model.
            device (str): The name of the torch device used for training.
            len_data (int): Number of training samples used (both actor
            generated and synthetic).
            pct_synth (float): Initial percentage of synthetic samples used
            for training.
            n_synth_data (int): Number of synthetic training samples.
            limit_rank (int): Maximum rank for synthetically-generated
            matrices.
            n_cob (int): Number of change of basis (cob) used for a single
            training sample.
            cob_prob (float): Probability of applying a change of basis.
            data_augmentation (bool): Whether to randomly swap the last
            operation of an episode with another operation.
            loss_params (Tuple[float, float]): Alpha and Beta parameters used
            in the loss function.
            random_seed (int): Randomizing seed.
            checkpoint_dir (str): Directory used to store model checkpoints.
            checkpoint_data_dir (str): Directory used to store games as JSON
            files.
            extra_devices (List[str]): Extra devices names used for multi-GPU
            training.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataset = TensorGameDataset(
            len_data,
            pct_synth,
            tensor_size,
            n_synth_data,
            limit_rank,
            f_prob_distribution,
            device=device,
            n_steps=n_steps,
            action_memory_len=(model.tensor_length - 1),
            random_seed=random_seed,
        )
        print("Got initial dataset")
        self.batch_size = batch_size
        self.max_rank = limit_rank
        if loss_params is None:
            self.alpha = 1
            self.beta = 1
        else:
            self.alpha, self.beta = loss_params
        self.checkpoint_dir = Path(
            checkpoint_dir if checkpoint_dir else BASE_CHECKPOINT_DIR
        )
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_data_dir = (
            checkpoint_data_dir
            if checkpoint_data_dir
            else Path(BASE_CHECKPOINT_DATA_DIR)
        )
        self.checkpoint_data_dir.mkdir(exist_ok=True, parents=True)
        self.change_of_basis = ChangeOfBasis(
            tensor_size, n_cob, cob_prob, device, random_seed
        )
        self.data_augmentation = data_augmentation
        self.extra_devices = extra_devices
        print("Trainer inited")

    def train_step(self):
        """Executes a single training step by optimizing the current model
        parameters."""
        self.dataset.recompute_synthetic_indexes()
        self.model.train()
        total_loss = 0
        dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print("Training AlphaTensor")
        # accumulation_steps = 4
        for states, scalars, policies, rewards in tqdm.tqdm(dl):
            loss_policy, loss_value = self.model(
                states, scalars, policies, rewards
            )
            loss = self.alpha * loss_policy + self.beta * loss_value
            self.optimizer.zero_grad()
            loss.backward()
            # if i % accumulation_steps == 0:
            #     self.optimizer.step()
            # self.optimizer.step()
            total_loss += loss.item()
        print(f"Total loss: {total_loss}")

    @torch.no_grad()
    def act_step(
        self,
        input_tensor: torch.Tensor,
        n_games: int,
        mc_n_sim: int,
        N_bar: int,
    ):
        """Runs actors in parallel to generate multiple games starting from
        the same input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor used to generate the
            games.
            n_games (int): Number of games to generate / actors to be run in
            parallel.
            mc_n_sim (int): Number of simulations used in the Monte Carlo tree
            search.
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy.
        """
        self.model.eval()
        best_reward = -1e10
        best_game = None

        for actor_id in range(n_games):
            input_tensor_cob = self.change_of_basis(input_tensor).to(
                self.device
            )
            print(f"Running actor {actor_id} / {n_games}")
            states, policies, rewards = actor_prediction(
                self.model,
                input_tensor_cob,
                self.max_rank,
                mc_n_sim,
                N_bar,
            )
            print(f"Actor {actor_id} finished. Final reward: {rewards[-1]}")
            if rewards[-1] > best_reward:
                print("New best actor!")
                best_reward = rewards[-1]
                best_game = (states, policies, rewards)
            self.dataset.add_game(states, policies, rewards)
            if self.data_augmentation:
                states, policies = swap_data(states, policies)
                self.dataset.add_game(states, policies, rewards)
        if best_game is not None:
            self.dataset.add_best_game(*best_game)

    def train(
        self,
        n_epochs: int,
        n_games: int,
        mc_n_sim: int,
        N_bar: int,
        initial_lr: float,
        lr_decay_factor: float,
        lr_decay_steps: int,
        starting_epoch: int = 0,
    ):
        """Trains the model for a given number of epochs.

        Args:
            n_epochs (int): Number of training epochs.
            n_games (int): Number of games to generate / actors to be run in
            parallel at each step.
            mc_n_sim (int): Number of simulations used in the Monte Carlo tree
            search at each step.
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy.
            initial_lr (float): Initial learning rate.
            lr_decay_factor (float): Learning rate's decay factor.
            lr_decay_steps (int): Number of learning rate's decay steps.
            starting_epoch (int, optional): Epoch from which to start / resume
            training.
        """
        self.model = self.model.to(self.device)
        if starting_epoch + 1 > n_epochs // 50:
            self.dataset.change_training_split(0.7, 0.05)
        if (
            starting_epoch + 1 > n_epochs // 10
        ):  # when restarting from a checkpoint
            mc_n_sim = mc_n_sim * 4
        for epoch in range(starting_epoch, n_epochs):
            if epoch + 1 == n_epochs // 50:
                self.dataset.change_training_split(0.7, 0.05)
            if epoch + 1 == n_epochs // 10:
                mc_n_sim = mc_n_sim * 4
            # apply learning rate decay each epoch if epoch < lr_decay_steps
            if 0 < epoch < lr_decay_steps - 1:
                lr = initial_lr * lr_decay_factor ** (epoch / lr_decay_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            print(f"Epoch {epoch} / {n_epochs}")
            self.train_step()
            if epoch % 10 == 0:
                self.act_step(
                    self.dataset.input_tensor, n_games, mc_n_sim, N_bar
                )
            # save checkpoint
            if (epoch + 1) % 50 == 0:
                checkpoint_name = f"checkpoint_{epoch + 1}.pt"
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                torch.save(
                    checkpoint,
                    self.checkpoint_dir / checkpoint_name,
                )
                print(f"Saving {checkpoint_name} in {self.checkpoint_dir}/")
                self.dataset.save_game_data(self.checkpoint_data_dir)
        print("Training finished")


# In[90]:


class LoadCheckpointDataOp(): # Operation
    """An operation which loads the games played while training an
    OpenAlphaTensor model."""

    def __init__(self):
        super().__init__()
        self._loaded = False

    def execute(self, games_store_dir: Path, trainer: Trainer):
        """Load the games played while training an OpenAlphaTensor model.

        Args:
            games_store_dir: The directory where the games are stored.
            trainer: The trainer to load the games into.
        """
        games_store_dir = games_store_dir or BASE_CHECKPOINT_DATA_DIR
        print(f"loading checkpoint {games_store_dir}")
        # if games_store_dir contains games, load them
        if (
            games_store_dir.exists()
            and (games_store_dir / "game_data.json").exists()
        ):
            trainer.dataset.load_games(games_store_dir)
        self._loaded = True

    def get_result(self) -> bool:
        """Returns whether the games were loaded or not."""
        return self._loaded


# In[91]:


class TrainingOperation(): # Operation
    """Operation which trains an AlphaTensor model to learn more efficient
    matrix multiplications."""

    def __init__(self):
        super().__init__()
        self._trained_model = None

        self._load_checkpoint_data_op = LoadCheckpointDataOp()

    def execute(
        self,
        model: AlphaTensorModel,
        input_size: int,
        n_steps: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        device: str,
        len_data: int,
        pct_synth: float,
        n_synth_data: int,
        limit_rank: int,
        max_epochs: int,
        n_actors: int,
        mc_n_sim: int,
        N_bar: int,
        last_epoch: int,
        lr: float,
        lr_decay_factor: float,
        lr_decay_steps: int,
        loss_params: Tuple[float, float] = None,
        random_seed: int = None,
        checkpoint_dir: str = None,
        checkpoint_data_dir: str = None,
        n_cob: int = 0,
        cob_prob: float = 0.0,
        data_augmentation: bool = False,
        extra_devices: List[str] = None,
    ):
        """Trains an AlphaTensor model to learn more efficient matrix
        multiplications.

        Args:
            model (AlphaTensorModel): The model to be trained.
            input_size (int): Flattened size of the matrices to be multiplied.
            n_steps (int): Number of steps used to get a single action out of
            a triplet.
            batch_size (int): Batch size.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            device (str): The name of the torch device used for training.
            len_data (int): Number of training samples used (both actor
            generated and synthetic).
            pct_synth (float): Initial percentage of synthetic samples used
            for training.
            n_synth_data (int): Number of synthetic training samples.
            limit_rank (int): Maximum rank for synthetically-generated
            matrices.
            max_epochs (int): Number of training epochs.
            n_actors (int): Number of actors to play a single each game at
            each training step.
            mc_n_sim (int): Number of simulations during Monte Carlo tree
            search.
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy.
            last_epoch (int): Latest epoch reached during training from which
            checkpoint data will be loaded.
            lr (float): Learning rate.
            lr_decay_factor (float): Learning rate's decay factor.
            lr_decay_steps (int): Number of learning rate's decay steps.
            loss_params (Tuple[float, float]): Alpha and Beta parameters used
            in the loss function.
            random_seed (int): Randomizing seed.
            checkpoint_dir (str): Directory used to store model checkpoints.
            checkpoint_data_dir (str): Directory used to store games as JSON
            files.
            n_cob (int): Number of change of basis (cob) used for a single
            training sample.
            cob_prob (float): Probability of applying a change of basis.
            data_augmentation (bool): Whether to randomly swap the last
            operation of an episode with another operation.
            extra_devices (List[str]): Extra devices names used for multi-GPU
            training.
        """
        checkpoint_data_dir = Path(checkpoint_data_dir or "games")
        print("Building trainer")

        # build trainer
        trainer = Trainer(
            model=model,
            tensor_size=input_size,
            n_steps=n_steps,
            batch_size=batch_size,
            optimizer=optimizer,
            device=device,
            len_data=len_data,
            pct_synth=pct_synth,
            n_synth_data=n_synth_data,
            limit_rank=limit_rank,
            loss_params=loss_params,
            random_seed=random_seed,
            checkpoint_dir=checkpoint_dir,
            checkpoint_data_dir=checkpoint_data_dir,
            data_augmentation=data_augmentation,
            cob_prob=cob_prob,
            n_cob=n_cob,
            extra_devices=extra_devices,
        )

        # load checkpoint data
        self._load_checkpoint_data_op.execute(
            games_store_dir=checkpoint_data_dir,
            trainer=trainer,
        )
        print("Start training")
        # train
        trainer.train(
            n_epochs=max_epochs,
            n_games=n_actors,
            mc_n_sim=mc_n_sim,
            N_bar=N_bar,
            starting_epoch=last_epoch,
            initial_lr=lr,
            lr_decay_factor=lr_decay_factor,
            lr_decay_steps=lr_decay_steps,
        )
        self._trained_model = trainer.model

    def get_trained_model(self):
        """Returns the trained model."""
        return self._trained_model


# In[92]:


class BuildModelOp():
    def __init__(self):
        super().__init__()
        self._model = None

    def execute(
        self,
        tensor_length: int,
        input_size: int,
        scalars_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
    ):
        """Builds the OpenAlphaTensor model.

        Args:
            tensor_length (int): Number of tensors to as history.
            input_size (int): Flattened size of the matrices to be multiplied.
            scalars_size (int): Size of the scalar vectors fed to the torso
            model.
            emb_dim (int): Embedding dimension.
            n_steps (int): Number of steps used to get a single action out of
            a triplet.
            n_logits (int): Number of logits output by the policy head.
            n_samples (int): Number of samples used by the policy head at
            evaluation time.
        """
        self._model = AlphaTensorModel(
            tensor_length=tensor_length,
            input_size=input_size,
            scalars_size=scalars_size,
            emb_dim=emb_dim,
            n_steps=n_steps,
            n_logits=n_logits,
            n_samples=n_samples,
        )

    def get_model(self) -> AlphaTensorModel:
        return self._model


# In[93]:


class SaveModelOp():
    """An operation which saves an OpenAlphaTensor model.
    The model parameters are stored in a json file, while the model weights
    are stored in a .pt file."""

    def execute(
        self,
        model: AlphaTensorModel,
        save_dir: str,
    ):
        """Saves the OpenAlphaTensor model.

        Args:
            model (AlphaTensorModel): OpenAlphaTensor model to be saved.
            save_dir (str): Directory where the model will be saved.
        """
        save_dir = Path(save_dir if save_dir else ".")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "final_model.pt")
        model_params = {
            "input_size": model.input_size,
            "tensor_length": model.tensor_length,
            "scalars_size": 1,
            "emb_dim": model.emb_dim,
            "n_steps": model.n_steps,
            "n_logits": model.n_logits,
            "n_samples": model.n_samples,
        }
        # save parameters in a json file
        with open(save_dir / "model_params.json", "w") as f:
            json.dump(model_params, f)


# In[94]:


class BuildOptimizerOp():
    """An operation which builds an optimizer for an OpenAlphaTensor model."""

    def __init__(self):
        super().__init__()
        self._optimizer = None

    def execute(
        self,
        optimizer_name: str,
        model: AlphaTensorModel,
        lr: float,
        weight_decay: float,
    ):
        """Builds the optimizer for the OpenAlphaTensor model.

        Args:
            optimizer_name (str): Name of the optimizer used.
            model (AlphaTensorModel): OpenAlphaTensor model to be trained.
            lr (float): Learning rate.
            weight_decay (float): Weight decay used by the optimizer.
        """
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
        self._optimizer = optimizer

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Returns the built optimizer."""
        return self._optimizer


# In[95]:


def optimizer_to(optim: torch.optim.Optimizer, device: str):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# In[96]:


class LoadCheckPointOp():
    """An operation which loads a checkpoint during training of an
    OpenAlphaTensor model."""

    def __init__(self):
        super().__init__()
        self._last_epoch = None
        self._model = None
        self._optimizer = None

    def execute(
        self,
        model: AlphaTensorModel,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
    ):
        """Load a checkpoint from a directory.

        Args:
            model: The model to load the checkpoint into.
            optimizer: The optimizer to load the checkpoint into.
            checkpoint_dir: The directory to load the checkpoint from.
        """
        checkpoint_dir = checkpoint_dir or BASE_CHECKPOINT_DIR
        if (
            Path(checkpoint_dir).exists()
            and len(list(Path(checkpoint_dir).glob("*.pt"))) > 0
        ):

            def key_func(x):
                return int(x.stem.split("_")[-1])

            checkpoint_path = sorted(
                Path(checkpoint_dir).glob("*.pt"), key=key_func
            )[-1]
            print(f"Loading checkpoint from {checkpoint_path}")
            old_device = model.device
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(old_device)
            print(f"Loaded model to {old_device}")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_to(optimizer, old_device)
            last_epoch = int(checkpoint_path.stem.split("_")[-1])
        else:
            last_epoch = 0

        self._last_epoch = last_epoch
        self._model = model
        self._optimizer = optimizer

    def get_last_epoch(self) -> int:
        """Returns the last epoch of the loaded checkpoint."""
        return self._last_epoch

    def get_model(self) -> AlphaTensorModel:
        """Returns the model loaded from the checkpoint."""
        return self._model

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Returns the optimizer loaded from the checkpoint."""
        return self._optimizer


# In[97]:


class TrainAlphaTensorRootOp():
    """Root operation which trains an AlphaTensor model to learn more
    efficient matrix multiplications."""

    def __init__(self):
        super().__init__()
        self._model = None
        self._optimizer = None

        self._build_model_op = BuildModelOp()
        self._build_optimizer_op = BuildOptimizerOp()
        self._load_checkpoint_op = LoadCheckPointOp()
        self._training_op = TrainingOperation()
        self._save_model_op = SaveModelOp()

    def execute(
        self,
        tensor_length: int,
        input_size: int,
        scalars_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
        optimizer_name: str,
        lr: float,
        lr_decay_factor: float,
        lr_decay_steps: int,
        weight_decay: float,
        loss_params: Tuple[float, float],
        checkpoint_dir: str,
        checkpoint_data_dir: str,
        epochs: int,
        batch_size: int,
        len_data: int,
        n_synth_data: int,
        pct_synth: float,
        limit_rank: int,
        n_actors: int,
        mc_n_sim: int,
        N_bar: int,
        device: str,
        save_dir: str,
        random_seed: int,
        n_cob: int,
        cob_prob: float,
        data_augmentation: bool,
        extra_devices: List[str],
    ):
        """Trains an AlphaTensor model to learn more efficient matrix
        multiplications.

        Args:
            tensor_length (int): Number of step tensors fed to the model
            (history and current state),
            input_size (int): Flattened size of the matrices to be multiplied,
            scalars_size (int): Size of the scalar vectors fed to the torso
            model,
            emb_dim (int): Embedding dimension,
            n_steps (int): Number of steps used to get a single action out of
            a triplet,
            n_logits (int): Number of logits output by the policy head,
            n_samples (int): Number of samples used by the policy head at
            evaluation time,
            optimizer_name (str): Name of the optimizer used,
            lr (float): Learning rate,
            lr_decay_factor (float): Learning rate's decay factor,
            lr_decay_steps (int): Number of learning rate's decay steps,
            weight_decay (float): Weight decay used by the optimizer,
            loss_params (Tuple[float, float]): Alpha and Beta parameters used
            in the loss function,
            checkpoint_dir (str): Directory used to store model checkpoints,
            checkpoint_data_dir (str): Directory used to store games as JSON
            files,
            epochs (int): Number of training epochs,
            batch_size (int): Batch size,
            len_data (int): Number of training samples used (both actor
            generated and synthetic),
            n_synth_data (int): Number of synthetic training samples,
            pct_synth (float): Initial percentage of synthetic samples used
            for training,
            limit_rank (int): Maximum rank for synthetically-generated
            matrices,
            n_actors (int): Number of actors to play a single each game at
            each training step,
            mc_n_sim (int): Number of simulations during Monte Carlo tree
            search,
            N_bar (int): N_bar parameter used to compute tau when improving
            the policy,
            device (str): The name of the torch device used for training,
            save_dir (str): Directory where the final trained model will be
            stored,
            random_seed (int): Randomizing seed,
            n_cob (int): Number of change of basis (cob) used for a single
            training sample,
            cob_prob (float): Probability of applying a change of basis,
            data_augmentation (bool): Whether to randomly swap the last
            operation of an episode with another operation,
            extra_devices (List[str]): Extra devices names used for multi-GPU
            training.
        """
        if self._model is None:
            self._build_model_op.execute(
                tensor_length=tensor_length,
                input_size=input_size,
                scalars_size=scalars_size,
                emb_dim=emb_dim,
                n_steps=n_steps,
                n_logits=n_logits,
                n_samples=n_samples,
            )
            self._model = self._build_model_op.get_model().to(device)

        if self._build_model_op.get_model() is not None:
            self._build_optimizer_op.execute(
                optimizer_name=optimizer_name,
                model=self._build_model_op.get_model(),
                lr=lr,
                weight_decay=weight_decay,
            )
            self._optimizer = self._build_optimizer_op.get_optimizer()

        if self._model is not None and self._optimizer is not None:
            self._load_checkpoint_op.execute(
                self._model, self._optimizer, checkpoint_dir
            )

        if self._load_checkpoint_op.get_model() is not None:
            self._model = self._load_checkpoint_op.get_model()
            self._optimizer = self._load_checkpoint_op.get_optimizer()
            starting_epoch = self._load_checkpoint_op.get_last_epoch()
            self._training_op.execute(
                model=self._model,
                input_size=input_size,
                n_steps=n_steps,
                batch_size=batch_size,
                optimizer=self._optimizer,
                device=device,
                len_data=len_data,
                pct_synth=pct_synth,
                n_synth_data=n_synth_data,
                limit_rank=limit_rank,
                max_epochs=epochs,
                n_actors=n_actors,
                mc_n_sim=mc_n_sim,
                N_bar=N_bar,
                last_epoch=starting_epoch,
                lr=lr,
                lr_decay_factor=lr_decay_factor,
                lr_decay_steps=lr_decay_steps,
                loss_params=loss_params,
                random_seed=random_seed,
                checkpoint_dir=checkpoint_dir,
                checkpoint_data_dir=checkpoint_data_dir,
                n_cob=n_cob,
                cob_prob=cob_prob,
                data_augmentation=data_augmentation,
                extra_devices=extra_devices,
            )
        if self._training_op.get_trained_model() is not None:
            self._model = self._training_op.get_trained_model()
            self._save_model_op.execute(
                model=self._model,
                save_dir=save_dir,
            )

    def get_result(self) -> AlphaTensorModel:
        """Returns the trained torch model"""
        return self._model


# In[98]:


def train_alpha_tensor(
    tensor_length: int,
    input_size: int,
    scalars_size: int,
    emb_dim: int,
    n_steps: int,
    n_logits: int,
    n_samples: int,
    optimizer_name: str,
    lr: float,
    lr_decay_factor: float,
    lr_decay_steps: int,
    weight_decay: float,
    loss_params: Tuple[float, float],
    checkpoint_dir: str,
    checkpoint_data_dir: str,
    epochs: int,
    batch_size: int,
    len_data: int,
    n_synth_data: int,
    pct_synth: float,
    limit_rank: int,
    n_actors: int,
    mc_n_sim: int,
    N_bar: int,
    device: str,
    save_dir: str,
    random_seed: int,
    n_cob: int,
    cob_prob: float,
    data_augmentation: bool,
    extra_devices: List[str],
):
    """Trains an AlphaTensor model to learn more efficient matrix
    multiplications and returns it.

    Args:
        tensor_length (int): Number of tensors to as history.
        input_size (int): Flattened size of the matrices to be multiplied.
        scalars_size (int): Size of the scalar vectors fed to the torso model.
        emb_dim (int): Embedding dimension.
        n_steps (int): Number of steps used to get a single action out of a
        triplet.
        n_logits (int): Number of logits output by the policy head.
        n_samples (int): Number of samples used by the policy head at
        evaluation time.
        optimizer_name (str): Name of the optimizer used.
        lr (float): Learning rate.
        lr_decay_factor (float): Learning rate's decay factor.
        lr_decay_steps (int): Number of learning rate's decay steps.
        weight_decay (float): Weight decay used by the optimizer.
        loss_params (Tuple[float, float]): Alpha and Beta parameters used in
        the loss function.
        checkpoint_dir (str): Directory used to store model checkpoints.
        checkpoint_data_dir (str): Directory used to store games as JSON files.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        len_data (int): Number of training samples used (both actor generated
        and synthetic).
        n_synth_data (int): Number of synthetic training samples.
        pct_synth (float): Initial percentage of synthetic samples used for
        training.
        limit_rank (int): Maximum number of steps per episode and maximum rank
        for synthetically-generated matrices.
        n_actors (int): Number of actors to play a single each game at each
        training step.
        mc_n_sim (int): Number of simulations during Monte Carlo tree search.
        N_bar (int): N_bar parameter used to compute tau when improving the
        policy.
        device (str): The name of the torch device used for training.
        save_dir (str): Directory where the final trained model will be stored.
        random_seed (int): Randomizing seed.
        n_cob (int): Number of change of basis (cob) used for a single
        training sample.
        cob_prob (float): Probability of applying a change of basis.
        data_augmentation (bool): Whether to randomly swap the last operation
        of an episode with another operation.
        extra_devices (List[str]): Extra devices names used for multi-GPU
        training.
    """
    root_op = TrainAlphaTensorRootOp()
    root_op.execute(
        tensor_length=tensor_length,
        input_size=input_size,
        scalars_size=scalars_size,
        emb_dim=emb_dim,
        n_steps=n_steps,
        n_logits=n_logits,
        n_samples=n_samples,
        optimizer_name=optimizer_name,
        lr=lr,
        lr_decay_factor=lr_decay_factor,
        lr_decay_steps=lr_decay_steps,
        weight_decay=weight_decay,
        loss_params=loss_params,
        checkpoint_dir=checkpoint_dir,
        checkpoint_data_dir=checkpoint_data_dir,
        epochs=epochs,
        batch_size=batch_size,
        len_data=len_data,
        n_synth_data=n_synth_data,
        pct_synth=pct_synth,
        limit_rank=limit_rank,
        n_actors=n_actors,
        mc_n_sim=mc_n_sim,
        N_bar=N_bar,
        device=device,
        save_dir=save_dir,
        random_seed=random_seed,
        n_cob=n_cob,
        cob_prob=cob_prob,
        data_augmentation=data_augmentation,
        extra_devices=extra_devices,
    )
    return root_op.get_result()


# In[99]:


def compute_largest_divisor(n: int) -> int:
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i
    return 1


def main():
    batch_size = 8
    max_epochs = 6000
    action_memory = 7
    optimizer = "adamw"
    weight_decay = 1e-5
    lr = 1e-4
    lr_decay_factor = 0.1
    lr_decay_steps = 5000
    device = "cuda"
    len_data = 2048
    pct_synth = 0.9
    n_synth_data = 1000
    limit_rank = 80
    alpha = 1.0
    beta = 1.0
    random_seed = 42
    checkpoint_dir = None
    checkpoint_data_dir = None
    matrix_size = 4
    embed_dim = 1024
    actions_sampled = 32
    n_actors = 1
    mc_n_sim = 2 # more
    n_cob = 100
    cob_prob = 0.9983  # 1 - 0.0017
    data_augmentation = True
    N_bar = 100
    save_dir = None
    cardinality_vector = 5
    input_size = matrix_size**2
    n_steps = compute_largest_divisor(input_size)
    n_actions = cardinality_vector ** (3 * input_size // n_steps)
    loss_params = (alpha, beta)

    train_alpha_tensor(
        tensor_length=action_memory + 1,
        input_size=input_size,
        scalars_size=1,
        emb_dim=embed_dim,
        n_steps=n_steps,
        n_logits=n_actions,
        n_samples=actions_sampled,
        device=device,
        len_data=len_data,
        n_synth_data=n_synth_data,
        pct_synth=pct_synth,
        batch_size=batch_size,
        epochs=max_epochs,
        lr=lr,
        lr_decay_factor=lr_decay_factor,
        lr_decay_steps=lr_decay_steps,
        weight_decay=weight_decay,
        optimizer_name=optimizer,
        loss_params=loss_params,
        limit_rank=limit_rank,
        random_seed=random_seed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_data_dir=checkpoint_data_dir,
        n_actors=n_actors,
        mc_n_sim=mc_n_sim,
        n_cob=n_cob,
        cob_prob=cob_prob,
        data_augmentation=data_augmentation or False,
        N_bar=N_bar,
        extra_devices=[],
        save_dir=save_dir,
    )


# In[100]:


# torch.cuda.empty_cache() # put it somewhere
main()

