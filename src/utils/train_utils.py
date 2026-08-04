import os
import random

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from omegaconf.dictconfig import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int=42) -> None:
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)



def get_centroid(model: nn.Module,
                dataset: Dataset,
                device: torch.device,
                cfg: DictConfig) -> torch.Tensor:
    # Use the trainset's current sampled windows as-is (no resampling).
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False,
                            num_workers=4, pin_memory=True)

    latent = []
    model.eval()
    model = model.to(device)
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Collecting Latent'):
            x, weight, _ = batch
            if model.hparams.in_channels == 2:
                target, pred, _, _ = model(torch.cat([x, weight], dim=-2).to(device, non_blocking=True))
            else:
                target, pred, _, _ = model(x.to(device, non_blocking=True))

            latent.append(target[-1])

    latent = torch.cat(latent, dim=0).reshape(-1, model.hparams.latent_dim)

    print("Start Clustering")
    kmeans = KMeans(n_clusters=model.quantizer.n_e, random_state=cfg.random_seed)
    kmeans.fit(latent.detach().cpu().numpy())

    centroid = torch.Tensor(kmeans.cluster_centers_)

    return centroid