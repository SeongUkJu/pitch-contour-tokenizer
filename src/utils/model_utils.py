from typing import List
from omegaconf.dictconfig import DictConfig


def shrink_to_L(pools: list[int], L: int) -> list[int]:
    if len(pools) <= L:
        return sorted(pools, reverse=True)
    div = []
    for i in range(len(pools)//L + 1):
        div.append(pools[i*L:(i+1)*L])
    div[-1] = div[-1] + [1] * (L - len(div[-1]))

    out = sorted(div[0], reverse=True)
    for pool in div[1:]:
        out = sorted([t*p for t, p in zip(out, sorted(pool))], reverse=True)

    return out


def prime_factorization(num_layers:int, 
                        target: int) -> List[int]:
    factors = []
    for p in [2,3,5,7]:
        while target % p == 0:
            factors.append(p)
            target //= p
    if target > 1:
        factors.append(target)

    factors = factors + [1] * (num_layers - len(factors))
    factors = sorted(factors, reverse=True)
    return factors




def calc_conv_params(hparams: DictConfig, 
                     mode: str='encoder') -> List[dict]:
    num_layers = hparams.num_layers
    hidden_dim = hparams.latent_dim
    in_channels = hparams.in_channels
    receptive_field = hparams.receptive_field

    parameters = [{'in_channels':0, 'out_channels':0, 'max_pool':0} for i in range(num_layers)]
    pools = prime_factorization(num_layers, receptive_field)
    pools = shrink_to_L(pools, num_layers)

    in_channels_ = in_channels
    if hparams.use_gradual_size:
        channels = prime_factorization(num_layers, hidden_dim)

        if mode=='encoder':
            for idx, (c, p) in enumerate(zip(channels, pools)):
                out_channels = c if idx==0 else in_channels_*c                
                parameters[idx]['in_channels'] = in_channels_
                parameters[idx]['out_channels'] = out_channels
                parameters[idx]['max_pool'] = p
                in_channels_ = out_channels

        elif mode=='decoder':
            for idx, (c, p) in enumerate(zip(reversed(channels), reversed(pools))):
                if idx==0:
                    in_channels_ = hidden_dim
                out_channels = in_channels_//c
                parameters[idx]['in_channels'] = in_channels_
                parameters[idx]['out_channels'] = out_channels
                parameters[idx]['max_pool'] = p
                in_channels_ = out_channels
            parameters[-1]['out_channels'] = in_channels

    else:
        if mode=='encoder':
            for idx, p in enumerate(pools):
                parameters[idx]['in_channels'] = in_channels
                parameters[idx]['out_channels'] = hidden_dim
                parameters[idx]['max_pool'] = p

                in_channels = parameters[idx]['out_channels']

        elif mode=='decoder':
            for idx, p in enumerate(reversed(pools)):
                parameters[idx]['in_channels'] = hidden_dim
                parameters[idx]['out_channels'] = hidden_dim
                parameters[idx]['max_pool'] = p

            parameters[-1]['out_channels'] = in_channels

    return parameters




