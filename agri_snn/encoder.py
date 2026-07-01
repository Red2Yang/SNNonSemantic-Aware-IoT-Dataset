import torch
from spikingjelly.activation_based import encoding


class MaskedPoissonEncoder:
    def __init__(self, T: int = 16):
        self.T = T
        self.base_encoder = encoding.PoissonEncoder()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            mask = torch.ones_like(x)

        x_min = x.min(dim=1, keepdim=True)[0]
        x_pos = x - x_min
        x_max = x_pos.max(dim=1, keepdim=True)[0]
        x_max = torch.clamp(x_max, min=1e-6)
        x_norm = x_pos / x_max

        spike_seq = []
        for t in range(self.T):
            spike = self.base_encoder(x_norm) * mask
            spike_seq.append(spike.unsqueeze(0))
        return torch.cat(spike_seq, dim=0)   # [T, N, C]