import torch
from torch.utils.data import DataLoader

import json
from pathlib import Path
from omegaconf import DictConfig

import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from utils.infer_utils import *
from utils.data_utils import ClsDatasetMaker


# ---------- resume / skip helpers ----------

def _load_exp_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def _save_exp_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _should_run(name: str, state: dict, skip_list: set, force: bool) -> bool:
    if name in skip_list:
        return False
    if state.get(name, False) and not force:
        return False
    return True


def _mark_done(name: str, state: dict, state_path: Path) -> None:
    state[name] = True
    _save_exp_state(state_path, state)


# ---------- VQVAE ----------

def run_exp(cfg: DictConfig, model, testset, save_dir: Path, run=None, device='cuda'):
    if not cfg.get('exp', False):
        return

    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    model.eval()

    window = cfg.dataset.params.window
    n_codes = cfg.model.params.num_codebooks
    offset = cfg.exp.offset

    skip = set(cfg.exp.get('skip', []))
    force = cfg.exp.get('force', False)
    state_path = save_dir / f'.exp_state_n{n_codes}.json'
    state = _load_exp_state(state_path)

    # KLD & Code Flip
    if _should_run('kld', state, skip, force) or _should_run('code_flip', state, skip, force):
        data_ids, start_ids, end_ids, _ = get_segment_meta(testset)
        for end_offset in [window//2 - offset, window - offset]:
            if _should_run('kld', state, skip, force):
                get_total_mean_klds(
                    dataset=testset, data_ids=data_ids, start_ids=start_ids, end_ids=end_ids,
                    model=model, n_codes=n_codes,
                    offset=offset, end_offset=end_offset,
                    prefix=f"EXP/KLD", pth=save_dir, run=run, device=device
                )
            if _should_run('code_flip', state, skip, force):
                get_total_avg_acc(
                    model=model, dataset=testset, data_ids=data_ids, start_ids=start_ids, end_ids=end_ids,
                    offset=offset, end_offset=end_offset,
                    prefix=f'EXP/CODE_FLIP', pth=save_dir, run=run, device=device
                )
        if _should_run('kld', state, skip, force):
            _mark_done('kld', state, state_path)
        if _should_run('code_flip', state, skip, force):
            _mark_done('code_flip', state, state_path)

    # Classification
    if cfg.exp.get('nia'):
        nia_cfg = cfg.exp.nia
        cls_dm = ClsDatasetMaker(nia_cfg)
        ntrainset, nvalidset, ntestset = cls_dm.get_datasets()

        label_map = ntrainset.label_map
        idx2label = {v: k for k, v in label_map.items()}
        eng_idx2label = idx2label  # metadata labels are already English (e.g. southern_flick)
        num_labels = len(eng_idx2label)

        label_token_cnt, P_token_given_label, P_label = compute_label_token_cnt(
            trainset=ntrainset, model=model,
            num_labels=num_labels, codebook_size=n_codes,
            pth=save_dir, device=device
        )
        P_label_uniform = np.ones(num_labels) / num_labels

        best_T = fit_temperature_with_log(
            validset=nvalidset, model=model,
            P_token_given_label=P_token_given_label, P_label=P_label,
            metric='f1', run=run, device=device
        )
        if _should_run('nia/map', state, skip, force):
            evaluate_with_log(
                testset=ntestset, model=model,
                P_token_given_label=P_token_given_label, P_label=P_label,
                temperature=best_T, num_labels=num_labels, idx2label=eng_idx2label,
                prefix=f'EXP/NIA/MAP', run=run, device=device
            )
            _mark_done('nia/map', state, state_path)
        # evaluate_with_log(
        #     testset=ntestset, model=model,
        #     P_token_given_label=P_token_given_label, P_label=P_label_uniform,
        #     temperature=0.0, num_labels=num_labels, idx2label=eng_idx2label,
        #     prefix=f'EXP/n{n_codes}/NIA/MLE', run=run
        # )
    else:
        print("[EXP] exp.nia not configured — skipping NIA classification.")

    # NIA VOCAL
    if cfg.exp.get('nia_vocal'):
        nia_cfg = cfg.exp.nia_vocal
        dm = ClsDatasetMaker(nia_cfg)
        ntrainset, nvalidset, ntestset = dm.get_datasets()

        label_map = ntrainset.label_map
        idx2label={v:k for k,v in label_map.items()}
        eng_idx2label = idx2label  # metadata labels are already English (e.g. southern_flick)
        num_labels = len(eng_idx2label)

        label_token_cnt, P_token_given_label, P_label = compute_label_token_cnt(
            trainset=ntrainset, model=model,
            num_labels=num_labels, codebook_size=n_codes,
            pth=save_dir, prefix='VOCAL', device=device
        )
        P_label_uniform = np.ones(num_labels) / num_labels

        best_T = fit_temperature_with_log(
            validset=nvalidset, model=model,
            P_token_given_label=P_token_given_label, P_label=P_label,
            metric='f1', run=run, device=device
        )
        if _should_run('nia/vocal/map', state, skip, force):
            evaluate_with_log(
                testset=ntestset, model=model,
                P_token_given_label=P_token_given_label, P_label=P_label,
                temperature=best_T, num_labels=num_labels, idx2label=eng_idx2label,
                prefix=f'EXP/NIA/VOCAL/MAP', run=run, device=device
            )
            _mark_done('nia/vocal/map', state, state_path)
        # evaluate_with_log(
        #     testset=ntestset, model=model,
        #     P_token_given_label=P_token_given_label, P_label=P_label_uniform,
        #     temperature=0.0, num_labels=num_labels, idx2label=eng_idx2label,
        #     prefix=f'EXP/n{n_codes}/NIA/VOCAL/MLE', run=run
        # )
    else:
        print("[EXP] exp.nia_vocal not configured — skipping NIA vocal classification.")


# ---------- AE (multi n_codes via separate wandb runs) ----------

def run_exp_ae(cfg: DictConfig, model, trainset, testset, save_dir: Path, run=None, device='cuda'):
    if not cfg.get('exp', False):
        return

    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    model.eval()

    batch_size = cfg.train.batch_size
    window = cfg.dataset.params.window
    offset = cfg.exp.offset

    # n_codes_list resolution order:
    #   1) cfg.exp.n_codes_list (explicit list)
    #   2) cfg.exp.n_codes (legacy single int, e.g. src/etc/run_exp.py)
    #   3) default [128, 256, 512]
    if 'n_codes_list' in cfg.exp:
        n_codes_list = list(cfg.exp.n_codes_list)
    elif 'n_codes' in cfg.exp:
        n_codes_list = [int(cfg.exp.n_codes)]
    else:
        n_codes_list = [128, 256, 512]

    # spawn_new_runs: only spin up fresh wandb runs when we genuinely need multiple.
    # If caller supplied a run AND only one n_codes is requested, reuse the external run
    # (e.g. resume_exp.py --n_codes 256, or etc/run_exp.py per-iteration usage).
    spawn_new_runs = (run is None) or (len(n_codes_list) > 1)

    skip = set(cfg.exp.get('skip', []))
    force = cfg.exp.get('force', False)

    # Latent extraction (n_codes-independent — done once):
    # scaler/umap/kmeans are fit on trainset latents; testset is only transformed.
    dl = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False)
    latent = []
    with torch.inference_mode():
        for x, weight, meta in tqdm(dl, desc='Collect Latent (train)'):
            _, _, z = model(x.to(device))
            latent.append(z.squeeze().detach().cpu())
    latent = torch.cat(latent, dim=0).numpy()

    scaler = StandardScaler()
    latent_z = scaler.fit_transform(latent)

    umap_embed = UMAP(n_components=5, random_state=cfg.random_seed, low_memory=True, n_jobs=-1)
    latent_red = umap_embed.fit_transform(latent_z)

    # Segment meta (n_codes-independent — done once)
    data_ids, start_ids, end_ids, _ = get_segment_meta(testset)

    # Capture wandb context (only needed if we're about to finish the external run and spawn new ones)
    if spawn_new_runs:
        if run is not None:
            wandb_project = cfg.wandb.get('project', None)
            wandb_group = run.group
            wandb_base_name = run.name
            wandb_notes = run.notes
            run.finish()
        else:
            wandb_project = cfg.wandb.get('project', None) if cfg.get('wandb', False) else None
            wandb_group = None
            wandb_base_name = save_dir.name
            wandb_notes = None

    for n_codes in n_codes_list:
        state_path = save_dir / f'.exp_state_ae_n{n_codes}.json'
        state = _load_exp_state(state_path)

        if spawn_new_runs:
            run_name = f"{wandb_base_name}_n{n_codes}"
            new_run = wandb.init(
                project=wandb_project, name=run_name, group=wandb_group, notes=wandb_notes,
                dir=str(save_dir), mode='online' if wandb_project else 'disabled',
            )
        else:
            new_run = run

        kmeans = KMeans(n_clusters=n_codes, random_state=cfg.random_seed)
        # multi-threaded KMeans fit is nondeterministic even with fixed random_state
        with threadpool_limits(limits=1):
            kmeans.fit(latent_red)

        # KLD & Code Flip
        for end_offset in [window//2 - offset, window - offset]:
            if _should_run('kld', state, skip, force):
                get_total_mean_klds_ae(
                    dataset=testset, data_ids=data_ids, start_ids=start_ids, end_ids=end_ids,
                    model=model, scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                    n_codes=n_codes, offset=offset, end_offset=end_offset,
                    prefix=f'EXP/KLD', pth=save_dir, run=new_run, device=device
                )
            if _should_run('code_flip', state, skip, force):
                get_total_avg_acc_ae(
                    model=model, scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                    dataset=testset, data_ids=data_ids, start_ids=start_ids, end_ids=end_ids,
                    offset=offset, end_offset=end_offset,
                    prefix=f'EXP/CODE_FLIP', pth=save_dir, run=new_run, device=device
                )
        if _should_run('kld', state, skip, force):
            _mark_done('kld', state, state_path)
        if _should_run('code_flip', state, skip, force):
            _mark_done('code_flip', state, state_path)

        # Classification
        if cfg.exp.get('nia'):
            nia_cfg = cfg.exp.nia
            cls_dm = ClsDatasetMaker(nia_cfg)
            ntrainset, nvalidset, ntestset = cls_dm.get_datasets()

            label_map = ntrainset.label_map
            idx2label = {v: k for k, v in label_map.items()}
            eng_idx2label = idx2label  # metadata labels are already English (e.g. southern_flick)
            num_labels = len(eng_idx2label)

            label_token_cnt, P_token_given_label, P_label = compute_label_token_cnt_ae(
                trainset=ntrainset, model=model,
                scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                num_labels=num_labels, n_codes=n_codes,
                pth=save_dir, device=device
            )
            P_label_uniform = np.ones(num_labels) / num_labels

            best_T = fit_temperature_with_log_ae(
                validset=nvalidset, model=model,
                scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                P_token_given_label=P_token_given_label, P_label=P_label,
                metric='f1', run=new_run, device=device
            )
            if _should_run('nia/map', state, skip, force):
                evaluate_with_log_ae(
                    testset=ntestset, model=model,
                    scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                    P_token_given_label=P_token_given_label, P_label=P_label,
                    temperature=best_T, num_labels=num_labels, idx2label=eng_idx2label,
                    prefix=f'EXP/NIA/MAP', run=new_run, device=device
                )
                _mark_done('nia/map', state, state_path)
            # evaluate_with_log_ae(
            #     testset=ntestset, model=model,
            #     scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
            #     P_token_given_label=P_token_given_label, P_label=P_label_uniform,
            #     temperature=0.0, num_labels=num_labels, idx2label=eng_idx2label,
            #     prefix=f'EXP/n{n_codes}/NIA/MLE', run=new_run
            # )
        else:
            print("[EXP] exp.nia not configured — skipping NIA classification.")

        # NIA VOCAL
        if cfg.exp.get('nia_vocal'):
            nia_cfg = cfg.exp.nia_vocal
            dm = ClsDatasetMaker(nia_cfg)
            ntrainset, nvalidset, ntestset = dm.get_datasets()

            label_map = ntrainset.label_map
            idx2label={v:k for k,v in label_map.items()}
            eng_idx2label = idx2label  # metadata labels are already English (e.g. southern_flick)
            num_labels = len(eng_idx2label)

            label_token_cnt, P_token_given_label, P_label = compute_label_token_cnt_ae(
                trainset=ntrainset, model=model,
                scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                num_labels=num_labels, n_codes=n_codes,
                pth=save_dir, prefix='VOCAL', device=device
            )
            P_label_uniform = np.ones(num_labels) / num_labels

            best_T = fit_temperature_with_log_ae(
                validset=nvalidset, model=model,
                scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                P_token_given_label=P_token_given_label, P_label=P_label,
                metric='f1', run=new_run, device=device
            )
            if _should_run('nia/vocal/map', state, skip, force):
                evaluate_with_log_ae(
                    testset=ntestset, model=model,
                    scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
                    P_token_given_label=P_token_given_label, P_label=P_label,
                    temperature=best_T, num_labels=num_labels, idx2label=eng_idx2label,
                    prefix=f'EXP/NIA/VOCAL/MAP', run=new_run, device=device
                )
                _mark_done('nia/vocal/map', state, state_path)
            # evaluate_with_log_ae(
            #     testset=ntestset, model=model,
            #     scaler=scaler, umap_embed=umap_embed, kmeans=kmeans,
            #     P_token_given_label=P_token_given_label, P_label=P_label_uniform,
            #     temperature=0.0, num_labels=num_labels, idx2label=eng_idx2label,
            #     prefix=f'EXP/n{n_codes}/NIA/VOCAL/MLE', run=new_run
            # )
        else:
            print("[EXP] exp.nia_vocal not configured — skipping NIA vocal classification.")

        if spawn_new_runs:
            new_run.finish()
