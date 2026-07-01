import torch
import torch.nn as nn
import logging
from spikingjelly.activation_based import neuron, layer, surrogate, base

logging.getLogger('spikingjelly').setLevel(logging.ERROR)


class DualBranchAgriSNN(nn.Module):
    def __init__(self, ms_dim=9, rgb_dim=8, hidden_dim=64,
                 tau=2.0, step_mode='m'):
        super().__init__()
        sg = surrogate.ATan()

        self.ms_branch = nn.Sequential(
            layer.Linear(ms_dim, hidden_dim, step_mode=step_mode),
            neuron.LIFNode(tau=tau, surrogate_function=sg,
                           step_mode=step_mode)
        )
        self.rgb_branch = nn.Sequential(
            layer.Linear(rgb_dim, hidden_dim, step_mode=step_mode),
            neuron.LIFNode(tau=tau, surrogate_function=sg,
                           step_mode=step_mode)
        )
        self.fusion_gate = nn.Sequential(
            layer.Linear(hidden_dim * 2, hidden_dim, step_mode=step_mode),
            neuron.LIFNode(tau=tau, surrogate_function=sg,
                           step_mode=step_mode)
        )
        self.action_net = nn.Sequential(
            layer.Linear(hidden_dim, 32, step_mode=step_mode),
            neuron.LIFNode(tau=tau, surrogate_function=sg,
                           step_mode=step_mode),
            layer.Linear(32, 4, step_mode=step_mode),
            neuron.LIFNode(tau=tau, surrogate_function=sg,
                           step_mode=step_mode)
        )

    def forward(self, x_ms_seq, x_rgb_seq):
        ms_feat = self.ms_branch(x_ms_seq) 
        rgb_feat = self.rgb_branch(x_rgb_seq) 
        combined = torch.cat([ms_feat, rgb_feat], dim=-1)
        fused = self.fusion_gate(combined) 

        spikes = self.action_net(fused) 
        rates = spikes.mean(dim=0)  
        return rates

    def reset(self):
        for m in self.modules():
            if isinstance(m, base.MemoryModule):
                m.reset()