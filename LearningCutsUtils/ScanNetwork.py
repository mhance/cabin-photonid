import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F

from cabin import OneToOneLinear,EfficiencyScanNetwork


# This should really go into CABIN at some point.
def __getitem__EfficiencyScanNetwork(self, key):
    if key < len(self.nets):
        return self.nets[key]
    else:
        raise KeyError(f"Key '{key}' not found in nets.")

EfficiencyScanNetwork.__getitem__ = __getitem__EfficiencyScanNetwork


class ScanNetwork(torch.nn.Module):
    def __init__(self,features,pt,mu,effics,weights=None,activationscale=2.):
        super().__init__()
        self.features = features
        self.pt = pt
        self.mu = mu
        self.effics = effics
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.nets = \
            torch.nn.ModuleList([
                torch.nn.ModuleList([
                    EfficiencyScanNetwork(features,effics,weights,activationscale)
                    for j in range(len(self.mu))
                ])
                for i in range(len(self.pt))
            ])

    def forward(self, x):
        ### returns a list of network outputs with len(effics)
        outputs=[[self.nets[i][j](x[i][j]) for j in range(len(self.mu))] for i in range(len(self.pt))]
        return outputs

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(device)
