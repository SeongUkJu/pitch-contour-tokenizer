import os
import random

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from omegaconf.dictconfig import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets.dataset_zoo import FullPitchDataset

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
    rng_state = random.getstate()
    saved_is_valid = dataset.is_valid
    saved_sampled_data = getattr(dataset, 'sampled_data', None)
    saved_loaded_data = dataset.loaded_data if not isinstance(dataset, FullPitchDataset) else None

    dataset.is_valid = True
    if isinstance(dataset, FullPitchDataset):
        dataset.sampled_data = dataset.sample_receptive_field()
    else:
        dataset.loaded_data = dataset._load_valid_data()
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

    dataset.is_valid = saved_is_valid
    if isinstance(dataset, FullPitchDataset):
        if saved_sampled_data is not None:
            dataset.sampled_data = saved_sampled_data
    else:
        dataset.loaded_data = saved_loaded_data
    random.setstate(rng_state)

    return centroid