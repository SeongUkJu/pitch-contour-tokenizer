from typing import Tuple

import torch
import torchaudio
import torch.nn as nn
from omegaconf.dictconfig import DictConfig

from models.modules import ConvEncoder, ConvTDecoder, VectorQuantizer

class Conv1DAE(nn.Module):
    def __init__(self, 
                 hparams: DictConfig,) -> None:
        super().__init__()
        self.hparams = hparams
        self.separate = hparams.separate

        self.mf = hparams.median_field
        self.rf = hparams.receptive_field
        self.latent_dim = hparams.latent_dim

        self.encoder = ConvEncoder(hparams)
        self.decoder = ConvTDecoder(hparams)


    def subtract_median_from_x(self, 
                                x: torch.Tensor,
                                mask: torch.Tensor | None=None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c_in, _ = x.shape
        field = self.mf if self.mf!=self.rf else self.rf
        x_r = x.reshape(b,c_in,-1,field) # b,c_in,m_comp,mf/rf

        if mask is not None:
            mask_r = mask.reshape(b,1,-1,field) # b,c_in,m_comp,field
            x_nan = x_r.masked_fill(mask_r, float('nan'))
            median = x_nan.nanmedian(dim=-1).values.unsqueeze(-1)
            median = median.nan_to_num(0.0)
        else:
            median = x_r.median(dim=-1).values.unsqueeze(-1) # b,c_in,m_comp,1 | b,c_in,1,1
    
        x_in = (x_r - median).reshape(b,c_in,-1) # b,c_in,t

        return x_in, median


    def add_median_to_recon(self, 
                            recon: torch.Tensor, 
                            median: torch.Tensor) -> torch.Tensor:
        b, c_in, _ = recon.shape

        if self.mf != self.rf:
            recon_r = recon.reshape(b,c_in,-1,self.mf) # b,c_in,m_comp,mf
        else:
            recon_r = recon.reshape(b,c_in,-1,self.rf) # b,c_in,t_comp,rf

        x_out = (recon_r + median).reshape(b,c_in,-1) # b,c_in,t

        return x_out


    def separate_receptive_field(self, 
                                x: torch.Tensor) -> torch.Tensor:
        b, c_in, _ = x.shape
        return x.reshape(b,c_in,-1,self.rf).permute(0,2,1,3).reshape(-1,c_in,self.rf) # b*t_comp,c_in,rf


    def combine_receptive_field(self, 
                                recon: torch.Tensor, 
                                b: int) -> torch.Tensor:
        _, c_in, _ = recon.shape
        return recon.reshape(b,-1,c_in,self.rf).permute(0,2,1,3).reshape(b,c_in,-1) # b,c_in,t


    def reshape_latent(self,
                        z: torch.Tensor,
                        b: int) -> torch.Tensor:
        _,c,_ = z.shape
        return z.reshape(b,-1,c,1).squeeze(-1) # b*t_comp,c,1 -> b,t_comp,c | b,c,1 -> b,1,c


    def encode(self, 
                x: torch.Tensor,
                mask: torch.Tensor | None=None) -> Tuple[torch.Tensor]:
        # subtract_median_from_x
        x_in, median = self.subtract_median_from_x(x, mask=mask) # (b,c_in,t), (b,c_in,m_comp,1 | b,c_in,1,1)

        # reshape input
        if self.separate:
            x_in = self.separate_receptive_field(x_in) # b*t_comp,c_in,rf

        # encode input
        z = self.encoder(x_in) # b*t_comp,c,1 | b,c,1

        return z, median


    def decode(self, 
                z: torch.Tensor, 
                median: torch.Tensor, 
                b: torch.Tensor) -> torch.Tensor:
        # decode latent
        recon = self.decoder(z) # (b*t_comp,c,1 | b,c,1), b*t_comp,c_in,rf | b,c_in,t

        # reshape recon
        if self.separate:
            recon = self.combine_receptive_field(recon, b) # b,c_in,t

        # add median from recon
        x_out = self.add_median_to_recon(recon, median) # b,c_in,t

        return x_out


    def forward(self, 
                x: torch.Tensor,
                mask: torch.Tensor | None=None) -> Tuple[torch.Tensor]:
        if x.ndim==2: # b,t
            x = x.unsqueeze(1) # b,c_in,t

        b, _, _ = x.shape

        # Encode
        z, median = self.encode(x, mask=mask) # (b*t_comp,c,1 | b,c,1), (b,c_in,m_comp,1 | b,c_in,1,1)

        # Decode
        x_out = self.decode(z, median, b) # b,c_in,t

        # out
        z = self.reshape_latent(z, b) # b,t_comp,c | b,1,c
        target, pred = x, x_out

        return target, pred, z





class Conv1DVQVAE(Conv1DAE):
    def __init__(self,
                 hparams: DictConfig) -> None:
        super().__init__(hparams)
        self.quantizer = VectorQuantizer(n_e=hparams.num_codebooks, 
                                         e_dim=self.latent_dim, 
                                         beta=hparams.beta, 
                                         ortho_reg=hparams.get('ortho_reg', 0.0),
                                         use_simvq=hparams.get('use_simvq', False))


    def forward(self, 
                x: torch.Tensor,
                mask: torch.Tensor | None=None) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor], torch.Tensor]:
        if x.ndim==2: # b,t
            x = x.unsqueeze(1) # b,c_in,t

        b, _, _ = x.shape

        # Encode
        z, median = self.encode(x, mask=mask)

        # Quantize latent
        z_q, vq_loss, min_e = self.quantizer(z)

        # Decode
        x_out = self.decode(z_q, median, b)

        # out
        z = self.reshape_latent(z, b)
        z_q = self.reshape_latent(z_q, b)
        target, pred = (x,z) , (x_out,z_q)

        return target, pred, vq_loss, min_e




class OffsetConv1DVQVAE(Conv1DVQVAE):
    def __init__(self,
                 hparams: DictConfig) -> None:
        super().__init__(hparams)
        self.offset_params = hparams.get('offset_params', False)


    def process_offset(self, 
                    recon: torch.Tensor) -> torch.Tensor:
        if not self.separate:
            b,c,_ = recon.shape
            recon = recon.reshape(b,c,-1,self.rf*2).permute(0,2,1,3).reshape(-1,c,self.rf*2)

        b_r,c,_ = recon.shape
        if self.offset_params.get('tempo_sr', False):
            recon = [torchaudio.functional.resample(recon, orig_freq=100, new_freq=t) for t in self.offset_params.tempo_sr]
            recon = torch.cat([o.unfold(dimension=-1, size=self.rf, step=1) for o in recon], dim=-2)
        else:
            recon = torch.cat([o.unfold(dimension=-1, size=self.rf, step=1) for o in recon], dim=-2)

        if self.offset_params.get('zoom_ratio', False):
            zoom_ratio = torch.tensor(self.offset_params.zoom_ratio, device=recon.device, dtype=recon.dtype).view(1,1,1,-1,1)
            recon = (recon.unsqueeze(-2) * zoom_ratio).reshape(b_r,c,-1,self.rf)

        if self.offset_params.get('y_offset', False):
            zoom_ratio = torch.tensor(self.offset_params.y_offset, device=recon.device, dtype=recon.dtype).view(1,1,1,-1,1)
            recon = (recon.unsqueeze(-2) + zoom_ratio).reshape(b_r,c,-1,self.rf)

        return recon


    def combine_receptive_field(self, 
                                recon: torch.Tensor, 
                                b: int) -> torch.Tensor:
        _, c_in, n_r, _ = recon.shape
        return recon.reshape(b,-1,c_in,n_r,self.rf).permute(0,2,3,1,4).reshape(b,c_in,n_r,-1) # b,c_in,n_r,t


    def add_median_to_recon(self, 
                            recon: torch.Tensor, 
                            median: torch.Tensor) -> torch.Tensor:
        b, c_in, n_r, _ = recon.shape

        if self.mf != self.rf:
            recon_r = recon.reshape(b,c_in,n_r,-1,self.mf)
        else:
            recon_r = recon.reshape(b,c_in,n_r,-1,self.rf)

        x_out = (recon_r + median.unsqueeze(-3)).reshape(b,c_in,n_r,-1)

        return x_out


    def decode(self, 
                z: torch.Tensor, 
                median: torch.Tensor, 
                b: torch.Tensor) -> torch.Tensor:
        # decode latent
        recon = self.decoder(z)

        # roll recon
        recon_r = self.process_offset(recon)

        # reshape recon
        recon_r = self.combine_receptive_field(recon_r, b)

        # add median from recon
        x_out = self.add_median_to_recon(recon_r, median)

        return x_out


    def forward(self, 
                x: torch.Tensor,
                mask: torch.Tensor | None=None) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor], torch.Tensor]:
        if x.ndim==2: # b,t
            x = x.unsqueeze(1) # b,c_in,t

        b, _, _ = x.shape

        # Encode
        z, median = self.encode(x, mask=mask)

        # Quantize latent
        z_q, vq_loss, min_e = self.quantizer(z)

        # Decode
        x_out = self.decode(z_q, median, b)

        # out
        z = self.reshape_latent(z, b)
        z_q = self.reshape_latent(z_q, b)
        target, pred = (x,z) , (x_out,z_q)

        return target, pred, vq_loss, min_e




