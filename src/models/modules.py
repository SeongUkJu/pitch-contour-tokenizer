from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig

from utils.model_utils import calc_conv_params
from losses.loss_zoo import VectorQuantizeLoss

class Conv1DBlock(nn.Module):
   def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int = 3, 
                dilation: int = 1,
                activation: str='relu') -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, 
                            out_channels, 
                            kernel_size=kernel_size, 
                            padding=padding, 
                            dilation=dilation,)

        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation=='leaky':
            self.activation = nn.LeakyReLU()
        
        self.conv_block = nn.Sequential(self.conv, self.activation)


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.conv_block(x)




class ConvT1DBlock(nn.Module):
   def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int, 
                stride: int,
                dilation: int=1,
                activation: str='relu') -> None:
        super().__init__()
        self.convt = nn.ConvTranspose1d(in_channels, 
                                        out_channels, 
                                        kernel_size=kernel_size,
                                        stride=stride, 
                                        dilation=dilation,
                                        padding=0,
                                        output_padding=0)
        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation=='leaky':
            self.activation = nn.LeakyReLU()

        self.convt_block = nn.Sequential(self.convt, self.activation)


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.convt_block(x)




class ConvEncoder(nn.Module):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.kernel_size = hparams.kernel_size
        self.dilation = hparams.dilation
        self.dropout = hparams.cnn_dropout
        self.activation = hparams.activation

        self.params = calc_conv_params(hparams, mode='encoder')
        self.encoder = self.build()


    def build(self) -> nn.Sequential:
        encoder = nn.Sequential()
        for idx, param in enumerate(self.params):
            encoder.add_module(f'conv_{idx}', 
                           Conv1DBlock(param['in_channels'], param['out_channels'], 
                                       kernel_size=self.kernel_size, 
                                       dilation=self.dilation,
                                       activation=self.activation))
            if idx == len(self.params)-1:
                encoder[-1].conv_block[1] = nn.Identity()

            if self.dropout: 
                encoder.add_module(f'dropout_{idx}', 
                               nn.Dropout1d(self.dropout))
            
            if param['max_pool'] != 1:
                encoder.add_module(f'maxpool_{idx}', 
                               nn.MaxPool1d(param['max_pool']))

        return encoder


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)




class ConvTDecoder(nn.Module):
    def __init__(self, hparams: DictConfig) -> None:
        super().__init__()
        self.dilation = hparams.dilation
        self.dropout = hparams.cnn_dropout
        self.activation = hparams.activation
        self.use_offset_params = hparams.get('use_offset_params', True)
        self.offset_params = hparams.get('offset_params', False)

        self.params = calc_conv_params(hparams, 'decoder')
        if self.offset_params and self.use_offset_params:
            self.params[-1]['out_channels'] = hparams.latent_dim
            self.final_params = {'in_channels': hparams.latent_dim, 
                                    'out_channels': hparams.in_channels, 
                                    'kernel_size': 2, 
                                    'stride': 2, 
                                    'dilation': hparams.dilation, 
                                    'padding': 0, 
                                    'output_padding': 0}
        
        self.decoder = self.build()


    def build(self) -> nn.Sequential:
        decoder = nn.Sequential()
        for idx, param in enumerate(self.params):
            decoder.add_module(f'convt_{idx}', 
                           ConvT1DBlock(param['in_channels'], param['out_channels'], 
                                       kernel_size=param['max_pool'],
                                       stride=param['max_pool'],
                                       dilation=self.dilation,
                                       activation=self.activation))

            if (idx == len(self.params)-1) and (self.offset_params is False):
                decoder[-1].convt_block[1] = nn.Identity()

            if self.dropout and (self.offset_params is False): 
                decoder.add_module(f'dropout_{idx}', 
                               nn.Dropout1d(self.dropout))
        
        if self.offset_params and self.use_offset_params:
            decoder.add_module(f'offset_layer',
                                nn.ConvTranspose1d(**self.final_params))

        return decoder


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)




class VectorQuantizer(nn.Module):
    def __init__(self, 
                 n_e: int, 
                 e_dim: int, 
                 beta: float,
                 ortho_reg: float=0.0,
                 use_simvq: bool=False) -> None:
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.embedding = nn.Embedding(n_e, e_dim)
        self.vq_loss = VectorQuantizeLoss(beta)

        self.ortho_reg = ortho_reg

        self.use_simvq = use_simvq
        if self.use_simvq:
            self.embedding_proj = nn.Linear(e_dim, e_dim)    # learnable transform


    def init_embedding(self, 
                       init: str,
                       centroid: torch.Tensor | None=None):
        if init=='uniform':
            self.embedding.weight.data.uniform_(-1/self.n_e, 1/self.n_e) # n_e,e_dim
            if self.use_simvq:
                self.embedding.weight.requires_grad = False      # frozen anchor

        elif init=='gaussian':
            self.embedding.weight.data.normal_(mean=0.0, std=self.e_dim**-0.5) # n_e,e_dim
            if self.use_simvq:
                self.embedding.weight.requires_grad = False      # frozen anchor

        elif init=='kmeans' and centroid is not None:
            self.embedding.weight.data = centroid.to(self.embedding.weight.device)

        else:
            raise Exception("init must be 'uniform', 'gaussian' or 'kmeans'")


    def get_codebook(self):
        if self.use_simvq:
            codebook = self.embedding_proj(self.embedding.weight)
        else:
            codebook = self.embedding.weight

        return codebook


    def orthogonal_loss_fn(self, 
                           codebook:torch.Tensor) -> torch.Tensor:
        # codebook: (n_e, e_dim)
        normed = F.normalize(codebook, p=2, dim=-1, eps=1e-12)  # (n_e, e_dim)
        cosine_sim = torch.einsum('i d, j d -> i j', normed, normed)  # (n_e, n_e)

        return (cosine_sim ** 2).sum() / (self.n_e ** 2) - (1 / self.n_e)


    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        z_flatten = z.contiguous().view(-1, self.e_dim) if z.ndim != 2 else z # b*tcomp,e_dim(c)

        codebook = self.get_codebook()

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        z_sq = torch.sum(z_flatten**2, dim=-1, keepdim=True)
        e_sq = torch.sum(codebook**2, dim=-1)
        ez2 = 2*torch.matmul(z_flatten, codebook.t())
        d = z_sq + e_sq - ez2 # b*tcomp, n_e

        # find closest encodings
        min_ids = torch.argmin(d, dim=-1).unsqueeze(1) # b*tcomp,1
        min_e = torch.zeros(min_ids.shape[0], self.n_e).to(z.device) # b*tcomp,n_e
        min_e.scatter_(1, min_ids, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_e, codebook).view(z.shape) # b*tcomp,e_dim

        loss, codebook_loss, commitment_loss = self.vq_loss(z, z_q)

        ortho_loss = None
        if self.ortho_reg:
            ortho_loss = self.orthogonal_loss_fn(codebook)
        
        z_q = z + (z_q - z).detach()

        return z_q, (loss, codebook_loss, commitment_loss, ortho_loss), min_e



