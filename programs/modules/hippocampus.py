import numpy as np
import torch
from torch import nn

from .unit_mask import UnitMask


class Hippocampus(nn.Module):
    def __init__(self, num_units, dim_hidden, mask_p, num_attention_groups=8):
        super().__init__()
        self.dim_inputs = num_units
        self.dim_outputs = num_units
        self.num_units = num_units
        self.dim_hidden = dim_hidden
        self.num_groups = num_attention_groups
        self.mask_p = mask_p

        self.mask = UnitMask(self.mask_p)

        self.model = nn.Sequential(
            nn.Linear(self.dim_inputs, self.dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.dim_hidden, self.dim_outputs)
        )

        def eye_mask_fn(x: torch.Tensor):
            mask = torch.eye(x.shape[0], device=x.device) == 1
            return x.masked_fill(mask.unsqueeze(-1), -np.inf)

        self.eye_mask = eye_mask_fn
        self.temperature = nn.Parameter(torch.ones(self.num_units))

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.detach()
        x = x.view(batch_size, self.dim_inputs)
        # project to units
        key = self.model(x)

        x, mask = self.mask(x)  # s_b * num_units,  s_b * num_units
        query = self.model(x)

        weights = -torch.abs(key - query.unsqueeze(1))  # s_b(q) * s_b(k) * num_units
        weights = weights * self.temperature
        weights = self.eye_mask(weights)

        weights = weights.view(batch_size, batch_size // self.num_groups, self.num_groups, self.num_units)
        weights = torch.softmax(weights, dim=1)
        weights = weights.view(batch_size, batch_size, self.num_units) / 8

        return weights, mask

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}, ' \
               'dim_attention_global:{dim_attention_global}, ' \
               'mask_p:{mask_p}'.format(**self.__dict__)
