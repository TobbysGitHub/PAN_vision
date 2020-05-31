from torch import nn
from .unit_linear import UnitLinear


class Encoder(nn.Module):

    def __init__(self, num_units, dim_inputs, dim_hidden):
        super().__init__()

        self.num_units = num_units
        self.dim_inputs = dim_inputs
        self.dim_hidden = dim_hidden
        self.dim_outputs = num_units
        self.model = nn.Sequential(
            UnitLinear(num_units, dim_inputs, dim_hidden),
            nn.LeakyReLU(),
            UnitLinear(num_units, dim_hidden, 1),
        )

    def forward(self, x):
        """
        :param x: s_b * d_input
        """
        # x = x.detach()
        x = x.view(-1, 1, self.dim_inputs) \
            .expand(-1, self.num_units, -1)  # s_b * n_u * d_input

        x = self.model(x).squeeze(-1)

        return x  # s_b * n_u

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}'.format(**self.__dict__)
