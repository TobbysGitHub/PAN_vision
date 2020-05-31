import torch
from torch import nn


class UnitMask(nn.Module):

    def __init__(self, mask_p):
        super().__init__()
        self.drop_p = mask_p

    def forward(self, x):
        """
        :param x: s_b * n_u
        :return: s_b * n_u
        """
        batch_size, num_unit = x.shape
        mask = torch.rand_like(x) < self.drop_p
        mask_num = torch.sum(mask, dim=1).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        x = x * (torch.Tensor([num_unit]).to(x.device).float() / mask_num.float())

        return x, mask

    def extra_repr(self) -> str:
        return 'drop_p={drop_p}'.format(**self.__dict__)
