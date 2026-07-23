from pathlib import Path
from datetime import datetime

import hydra
import wandb
from omegaconf import OmegaConf

import torch

import trainers
from losses import loss_zoo
from models import model_zoo
from metrics import metric_zoo
from datasets import dataset_zoo
from schedulers import scheduler_zoo

from utils import data_utils
from utils.train_utils import set_seed, get_centroid

T = datetime.now().strftime('%m%d_%H%M%S')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


@hydra.main(config_path="configs")
def main(cfg):
    # Setup
    set_seed(cfg.random_seed)
    hparams = cfg.model.params
    group_name = f"{cfg.model.get('name')}_{cfg.dataset.get('name')}_{cfg.loss.get('name')}__{hparams.get('receptive_field')}_{hparams.get('latent_dim')}_{hparams.get('num_layers')}__{hparams.get('num_codebooks')}_{hparams.get('beta')}"
    run_name = f"{group_name}_{T}"

    save_dir = Path(f"{cfg.dir.save_dir}/{run_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.get('project'):
        run = wandb.init(project=cfg.wandb.project,
                         name=run_name,
                         dir=str(save_dir),
                         mode="online",
                         group=group_name,
                         notes=cfg.wandb.get('notes', None))
    else:
        run = wandb.init(mode="disabled")

    for key, value in dict(wandb.config).items():
        OmegaConf.update(cfg, key, value, merge=True)

    _cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.config.update(_cfg_dict)
    with open(save_dir / "config.yaml", "w") as f:
        OmegaConf.save(OmegaConf.create(_cfg_dict), f)


    # Dataset
    if cfg.train.get('train_all', False):
        dataset_params = OmegaConf.to_container(cfg.dataset.params, resolve=True)
        trainset = getattr(dataset_zoo, cfg.dataset.name)(cfg.data.data_pth, **dataset_params, is_valid=False)
        validset = getattr(dataset_zoo, cfg.dataset.name)(cfg.data.data_pth, **dataset_params, is_valid=True)
        testset = validset
    else:
        dataset_maker_name = cfg.train.get('dataset_maker', 'BaseDatasetMaker')
        dataset_maker_class = getattr(data_utils, dataset_maker_name)
        dataset_maker = dataset_maker_class(cfg)
        trainset, validset, testset = dataset_maker.get_datasets()
    print(f"Length of dataset: {len(trainset)}, {len(validset)}, {len(testset) if testset is not None else None}")


    # Criterion
    criterion_class = getattr(loss_zoo, cfg.loss.name)
    criterion = criterion_class(**cfg.loss.params)


    # Metric
    metric = [getattr(metric_zoo, metric_name)() for metric_name in cfg.metric] if cfg.metric else None


    # Model
    model_class = getattr(model_zoo, cfg.model.name)
    model = model_class(hparams)
    model = model.to(DEV)

    if 'VQ' in model.__class__.__name__:
        if model.hparams.init == 'uniform':
            model.quantizer.init_embedding(init='uniform')

        elif model.hparams.init == 'gaussian':
            model.quantizer.init_embedding(init='gaussian')

        elif model.hparams.init == 'kmeans':
            centroid = get_centroid(model, trainset, DEV, cfg)
            model.quantizer.init_embedding(init='kmeans', centroid=centroid)

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # Optimizer
    optimizer_class = getattr(torch.optim, cfg.optim.name)
    optimizer = optimizer_class(model.parameters(), **cfg.optim.params)


    # Scheduler
    scheduler_class = getattr(scheduler_zoo, cfg.scheduler.name)
    scheduler = scheduler_class(optimizer, **cfg.scheduler.params)


    # Trainer
    trainer_name = cfg.train.get('trainer', 'Trainer')
    trainer_class = getattr(trainers, trainer_name)
    trainer = trainer_class(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            trainset=trainset,
                            validset=validset,
                            criterion=criterion,
                            metric=metric,
                            device=DEV,
                            save_dir=save_dir,
                            config=cfg,)

    trainer.train()


    if testset is not None:
        trainer.model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
        trainer.evaluate(testset, external_dataset=True)


    if cfg.exp.get('run', False):
        from experiments import run_exp, run_exp_ae

        model_name = cfg.model.name

        if model_name == 'Conv1DAE':
            run_exp_ae(cfg, trainer.model, testset, save_dir, run=run, device=DEV)

        elif model_name in ('Conv1DVQVAE', 'OffsetConv1DVQVAE'):
            run_exp(cfg, trainer.model, testset, save_dir, run=run, device=DEV)

        else:
            print(f"[EXP] Unknown model: {model_name}, skipping exp.")


    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
