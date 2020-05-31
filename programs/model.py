import torch
import torch.nn as nn

from .modules import Encoder
from .modules import Hippocampus

NUM_UNITS = 64
MASK_P = 0.2
SIZE = 32


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_units = NUM_UNITS
        self.size = SIZE
        self.encoder = Encoder(num_units=NUM_UNITS, dim_inputs=SIZE * SIZE, dim_hidden=128)
        self.hippocampus = Hippocampus(num_units=NUM_UNITS, dim_hidden=NUM_UNITS * 4, mask_p=MASK_P)
        self.temperature = nn.Parameter(torch.ones(NUM_UNITS))

    @staticmethod
    def gen_noise(x):
        noise = 0.1 * torch.randn_like(x)
        return noise

    def forward(self, x):
        x = x.view(-1, SIZE * SIZE)

        x1 = x + self.gen_noise(x)
        y1 = self.encoder(x1)  # s_b * n_u
        w, mask = self.hippocampus(y1)  # s_b(q) * s_b * num_units

        x2 = x + self.gen_noise(x)
        y2 = self.encoder(x2)

        return (y1, w), y2, mask

    def encode(self, x):
        x = x.view(-1, SIZE * SIZE)
        y = self.encoder(x)

        return y
