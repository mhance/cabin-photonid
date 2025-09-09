import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from LearningCutsUtils.EfficiencyScanNetwork import EfficiencyScanNetwork

from cabin import OneToOneLinear


class PtScanNetwork(torch.nn.Module):
    def __init__(self,features,params,weights=None,activationscale=2.):
        super().__init__()
        self.features = features
        self.pt = params[0]
        self.effics = params[1]
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.nets = torch.nn.ModuleList([EfficiencyScanNetwork(features, self.effics, weights, activationscale) for i in range(len(self.pt))])

    def forward(self, x):
        ### returns a list of len(pt) of lists of network outputs with len(effics)
        ### dimensions [[tens,tens,tens],
        ###             [  tensors     ],
        ###             [   tensors    ]]
        outputs=[self.nets[i](x[i]) for i in range(len(self.pt))]
        return outputs

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(device)

