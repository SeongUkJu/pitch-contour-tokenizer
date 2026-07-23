import torch
import torch.nn as nn

class Perplexity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                min_e: torch.Tensor, 
                **kwargs):
        e_mean = torch.mean(min_e, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-12)))

        return perplexity
