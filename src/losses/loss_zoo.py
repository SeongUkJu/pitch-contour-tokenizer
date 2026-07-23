import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import _reduction as _Reduction


class MSELoss(nn.Module):
    def __init__(self, 
                 reduction: str="mean",
                 use_weight: bool=True) -> None:
        super().__init__()
        self.reduction = reduction
        self.use_weight = use_weight

    def forward(self, 
                pred: Tensor, 
                target: Tensor, 
                weight: Optional[Tensor]=None,
                **kwargs) -> Tensor:
        if isinstance(pred, tuple):
            pred, target = pred[0], target[0]

        if not (target.size() == pred.size()):
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size ({pred.size()}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.",
                stacklevel=2,
            )

        pred, target = torch.broadcast_tensors(pred, target)

        if (weight is not None) and (self.use_weight):
            if weight.size() != pred.size():
                raise ValueError('Weights and input must have the same size.')
            
            weight = ~weight

            squared_errors = torch.pow(pred-target, 2)
            if self.reduction == 'none':
                loss = squared_errors * weight
            elif self.reduction == 'sum':
                loss = torch.sum(squared_errors * weight)
            elif self.reduction == 'mean':
                weighted_squared_errors = torch.sum(squared_errors * weight, dim=-1) / torch.sum(weight, dim=-1)
                loss = torch.mean(weighted_squared_errors)
            else:
                raise ValueError(
                    f"Invalid reduction mode: {self.reduction}. Expected one of 'none', 'sum', 'mean'."
                )
        
        else:
            loss = torch._C._nn.mse_loss(pred, target, _Reduction.get_enum(self.reduction))
        
        return loss



class OffsetMSELoss(MSELoss):
    def __init__(self, 
                 receptive_field: int,
                 reduction: str="mean",
                 use_weight: bool=True) -> None:
        super().__init__(reduction, use_weight)
        self.rf = receptive_field


    def forward(self, 
                pred: Tensor, 
                target: Tensor, 
                weight: Optional[Tensor]=None,
                **kwargs) -> Tensor:
        if isinstance(pred, tuple):
            pred, target = pred[0], target[0]
        b,c,r,t = pred.shape
        pred, target = pred.reshape(b,c,r,t//self.rf,self.rf), target.unsqueeze(-2).reshape(b,c,1,t//self.rf,self.rf)
        squared_errors = torch.pow(pred-target, 2) # b,c_in,rolled_size,t_comp,rf

        if (weight is not None) and (self.use_weight):
            weight = weight.unsqueeze(-2).unsqueeze(-2).reshape(b,c,1,t//self.rf,self.rf)
            weighted_squared_errors = torch.sum(squared_errors * ~weight, dim=-1) / torch.sum(~weight, dim=-1)
            min_squared_errors = torch.min(weighted_squared_errors, dim=-2).values
        else:
            avg_squared_errors = torch.mean(squared_errors, dim=-1)
            min_squared_errors = torch.min(avg_squared_errors, dim=-2).values
        loss = torch.mean(min_squared_errors)

        return loss



@torch.compile
class VectorQuantizeLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta


    def forward(self, 
                z: torch.Tensor, 
                z_q: torch.Tensor) -> torch.Tensor:
        codebook_loss = torch.nn.functional.mse_loss(z_q, z.detach())
        commitment_loss = torch.nn.functional.mse_loss(z_q.detach(), z)
        loss = codebook_loss + self.beta * commitment_loss

        return loss, codebook_loss, commitment_loss




