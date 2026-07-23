from models import model_zoo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


def get_model(path: str) -> nn.Module:
    cfg = OmegaConf.load(f'{path}/config.yaml')
    ckpt = torch.load(f'{path}/best_model.pt', map_location='cpu')
    model = getattr(model_zoo, cfg.model.name)(cfg.model.params)
    model.load_state_dict(ckpt)

    return model



def get_min_offset_contour(pred_contour: torch.Tensor,
                            target_contour: torch.Tensor, 
                            receptive_field: int,
                            weight: torch.Tensor | None=None,
                            return_ids: bool=False):
    b,c,r,t = pred_contour.shape
    pr = pred_contour.reshape(b,c,r,t//receptive_field,receptive_field)
    tr = target_contour.unsqueeze(-2).reshape(b,c,1,t//receptive_field,receptive_field)
    squared_errors = torch.pow(pr-tr, 2)

    if weight is not None:
        weight = weight.unsqueeze(-2).unsqueeze(-2).reshape(b,c,1,t//receptive_field,receptive_field)
        sqe = torch.sum(squared_errors * ~weight, dim=-1) / torch.sum(~weight, dim=-1)
    else:
        sqe = torch.mean(squared_errors, dim=-1)

    min_ids = torch.argmin(sqe, dim=-2).unsqueeze(-2).unsqueeze(-1).expand(-1,-1,-1,-1,receptive_field)
    pred_contour = pr.gather(dim=-3, index=min_ids).squeeze().reshape(b,c,-1)
    selected_ids = min_ids[...,0].squeeze()

    if return_ids:
        return pred_contour, selected_ids

    return pred_contour


def get_selected_offset(idx, rf, offset_params):
    y_offset    = getattr(offset_params, 'y_offset', [0])
    zoom_ratio  = getattr(offset_params, 'zoom_ratio', [1])
    tempo_sr    = getattr(offset_params, 'tempo_sr', [1])

    n_y, n_z = len(y_offset), len(zoom_ratio)
    n_windows_per_tempo = [(int(rf*2*sr/100 + 1) - rf + 1) if rf*2*sr/100 % 1 != 0 else (int(rf*2*sr/100) - rf + 1) for sr in tempo_sr]

    y_i = idx % n_y
    z_i = (idx // n_y) % n_z
    fin = (idx // n_y) // n_z

    cumsum = 0
    for t_i, nw in enumerate(n_windows_per_tempo):
        if fin < cumsum + nw:
            w_i = fin - cumsum
            break
        cumsum += nw

    return t_i, z_i, y_i, w_i, fin


def _log(log_dict, run=None):
    if run is not None:
        run.log(log_dict)
    elif wandb.run is not None:
        wandb.log(log_dict)
    else:
        print("[WANDB] No active run, skipping log.")


# Exp Code Flip
def get_segment_meta(dataset, dur=None):
    if dur is None:
        dur = dataset.window * 3
    
    data_ids, start_ids, end_ids, durations = [], [], [], []
    for idx, (x, _, _, is_filtered_full) in tqdm(enumerate(dataset.loaded_data)):
        if x.shape[-1] < dur:
            continue
        mask = (~is_filtered_full).int()
        diff = torch.diff(mask, prepend=mask[:1])
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]

        if mask[0] == 1:
            starts = torch.cat((torch.tensor([0], device=mask.device), starts))
        if mask[-1] == 1:
            ends = torch.cat((ends, torch.tensor([mask.shape[0]], device=mask.device)))

        valid_ids = torch.where((ends - starts) >= dur)[0]
        starts, ends = starts[valid_ids], ends[valid_ids]
        data_ids.extend([idx] * valid_ids.shape[0])
        start_ids.extend(starts)
        end_ids.extend(ends)
        durations.extend(ends - starts)

    sorted_indices = sorted(range(len(durations)), key=lambda i: durations[i], reverse=True)
    data_ids = [data_ids[i] for i in sorted_indices]
    start_ids = [start_ids[i] for i in sorted_indices]
    end_ids = [end_ids[i] for i in sorted_indices]
    durations = [durations[i] for i in sorted_indices]

    return data_ids, start_ids, end_ids, durations


def get_segments(dataset, data_ids, start_ids, end_ids, idx, offset, end_offset=None):
    offset_cnt = dataset.window // offset if end_offset is None else end_offset // offset + 1
    start_idx, end_idx = start_ids[idx], end_ids[idx]
    valid_len = end_idx - start_idx - dataset.window + offset if end_offset is None else end_idx - start_idx - end_offset
    segment_len = valid_len // dataset.window * dataset.window
    required_len = segment_len + (offset_cnt - 1) * offset
    x, _, _, _ = dataset.loaded_data[data_ids[idx]]
    x = x[start_idx:start_idx + required_len]
    segments = torch.stack([x[i * offset:i * offset + segment_len].reshape(-1, dataset.window) for i in range(offset_cnt)])
    return segments


def extract_min_e(model, x):
    return model.encode_tokens(x) if hasattr(model, 'encode_tokens') else model(x)[-1]


def get_inference_outputs(segments, model, offset, end_offset, device='cuda'):
    model.eval()
    with torch.inference_mode():
        min_e = extract_min_e(model, segments.reshape(-1, model.rf).to(device))
        min_e = min_e.argmax(dim=-1).reshape(end_offset // offset + 1, -1).detach().cpu()
    return min_e


def get_acc(inference_output, offset, window=128):
    flip_idx = window // 2 // offset + 1
    gt = inference_output[0]
    if flip_idx > inference_output.shape[0]:
        acc = (gt == inference_output[1:]).float().mean(dim=-1)
    else:
        acc = torch.cat([
            (gt == inference_output[1:flip_idx]).float().mean(dim=-1),
            (gt[1:] == inference_output[flip_idx:, :-1]).float().mean(dim=-1)
        ])
    return acc


def get_total_avg_acc(model, dataset, data_ids, start_ids, end_ids, offset, end_offset, device='cuda', prefix='test', run=None, pth=None):
    total_avg_acc = []
    for idx in tqdm(range(len(data_ids))):
        res = get_segments(dataset, data_ids, start_ids, end_ids, idx, offset, end_offset)
        inf_out = get_inference_outputs(res, model, offset, end_offset, device)
        total_avg_acc.append(get_acc(inf_out, offset).mean())
    total_avg_acc = np.array(total_avg_acc)

    if pth is not None:
        np.save(f"{pth}/code_flip.npy", total_avg_acc)

    _log({f'{prefix}/code_flip_acc/{offset}-{end_offset}':np.mean(total_avg_acc)}, run)
    return total_avg_acc

def get_total_avg_acc_ae(model, scaler, umap_embed, kmeans,
                         dataset, data_ids, start_ids, end_ids,
                         offset, end_offset, device='cuda', prefix='test', run=None, pth=None):
    all_z, all_shapes = [], []

    model.eval()
    with torch.inference_mode():
        for idx in tqdm(range(len(data_ids)), desc='Extract latent'):
            res = get_segments(dataset, data_ids, start_ids, end_ids, idx, offset, end_offset)
            b, s, t = res.shape
            x = res.reshape(-1, t).to(device)
            _, _, z = model(x)
            all_z.append(z.squeeze().detach().cpu().numpy())
            all_shapes.append((b, s))

    all_z_cat = np.concatenate(all_z, axis=0)
    z_scaled = scaler.transform(all_z_cat)
    z_umap = umap_embed.transform(z_scaled)
    code_ids = kmeans.predict(z_umap)

    total_avg_acc = []
    cursor = 0
    for b, s in all_shapes:
        n = b * s
        seg_code_ids = torch.tensor(code_ids[cursor:cursor+n]).reshape(b, s)
        total_avg_acc.append(get_acc(seg_code_ids, offset).mean())
        cursor += n

    total_avg_acc = np.array(total_avg_acc)

    if pth is not None:
        np.save(f"{pth}/code_flip_ae_{kmeans.n_clusters}.npy", total_avg_acc)

    _log({f'{prefix}/code_flip_acc/{offset}-{end_offset}': np.mean(total_avg_acc)}, run)
    return total_avg_acc





# EXP KLD
def get_df(n_codes, res, model, offset, used_df=True, device='cuda'):
    model.eval()
    with torch.inference_mode():
        b, s, t = res.shape
        min_e = extract_min_e(model, res.reshape(-1, t).to(device))
        min_e = min_e.reshape(b, s, -1).argmax(dim=-1).detach().cpu()

    keys = [offset * i for i in range(b)]
    df = pd.DataFrame(0, columns=range(n_codes), index=keys)
    df.index.name = 'offset'

    for i, k in enumerate(keys):
        df.loc[k] = torch.bincount(min_e[i], minlength=n_codes).numpy()

    df_sorted = df[df.sum(axis=0).sort_values(ascending=False).index]
    used_cols = df_sorted.columns[df_sorted.sum(axis=0) > 0]
    df_used = df_sorted[used_cols]

    return df_used if used_df else df_sorted


def kld_discrete(p, q, epsilon=1e-12):
    q, p = q + epsilon, p + epsilon
    q, p = q / q.sum(), p / p.sum()
    return (p * torch.log(p / q)).sum()


def compute_baseline_distribution_from_df(df):
    baseline = torch.tensor(df.sum(axis=0).values, dtype=torch.float32)
    return baseline / baseline.sum()


def kld_from_baseline_df(df):
    baseline = compute_baseline_distribution_from_df(df)
    kld_dict = {}
    for offset_idx in sorted(df.index):
        p = torch.tensor(df.loc[offset_idx].values, dtype=torch.float32)
        p = p / p.sum()
        kld_dict[offset_idx] = kld_discrete(p, baseline).item()
    return kld_dict, baseline


def get_mean_kld(kld_dict):
    kld_values = [kld_dict[o] for o in sorted(kld_dict.keys())]
    return np.mean(kld_values)


def get_total_mean_klds(dataset, data_ids, start_ids, end_ids, model, n_codes, offset, end_offset, device='cuda',
                        prefix='test', run=None, pth=None):
    mean_klds = {}
    for i in tqdm(range(len(data_ids))):
        res = get_segments(dataset, data_ids, start_ids, end_ids, i, offset, end_offset)
        output = get_df(n_codes, res, model, offset, used_df=False, device=device)
        kld_dict, _ = kld_from_baseline_df(output)
        mean_klds[i] = get_mean_kld(kld_dict)

    if pth is not None:
        np.save(f"{pth}/kld.npy", mean_klds)

    _log({f'{prefix}/kld/{offset}-{end_offset}':np.mean(sorted(mean_klds.values()))}, run)

    return mean_klds

def get_total_mean_klds_ae(dataset, data_ids, start_ids, end_ids,
                           model, scaler, umap_embed, kmeans,
                           n_codes, offset, end_offset, device='cuda',
                           prefix='test', run=None, pth=None):
    all_z, all_shapes = [], []

    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(len(data_ids)), desc='Extract latent'):
            res = get_segments(dataset, data_ids, start_ids, end_ids, i, offset, end_offset)
            b, s, t = res.shape
            x = res.reshape(-1, t).to(device)
            _, _, z = model(x)
            all_z.append(z.squeeze().detach().cpu().numpy())
            all_shapes.append((b, s))

    all_z_cat = np.concatenate(all_z, axis=0)
    z_scaled = scaler.transform(all_z_cat)
    z_umap = umap_embed.transform(z_scaled)
    code_ids = kmeans.predict(z_umap)

    mean_klds = {}
    cursor = 0
    for i, (b, s) in enumerate(all_shapes):
        n = b * s
        seg_code_ids = torch.tensor(code_ids[cursor:cursor+n]).reshape(b, s)
        cursor += n

        keys = [offset * j for j in range(b)]
        df = pd.DataFrame(0, columns=range(n_codes), index=keys)
        df.index.name = 'offset'
        for j, k in enumerate(keys):
            df.loc[k] = torch.bincount(seg_code_ids[j], minlength=n_codes).numpy()

        df_sorted = df[df.sum(axis=0).sort_values(ascending=False).index]
        kld_dict, _ = kld_from_baseline_df(df_sorted)
        mean_klds[i] = get_mean_kld(kld_dict)

    if pth is not None:
        np.save(f"{pth}/kld_ae_{n_codes}.npy", mean_klds)

    _log({f'{prefix}/kld/{offset}-{end_offset}': np.mean(list(mean_klds.values()))}, run)

    return mean_klds




def visualize_kld_from_baseline(kld_dict, baseline):
    offsets = sorted(kld_dict.keys())
    kld_values = [kld_dict[o] for o in offsets]

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    axes[0].bar(range(len(offsets)), kld_values, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=np.mean(kld_values), color='r', linestyle='--',
                    label=f'Mean KLD: {np.mean(kld_values):.4f}')
    axes[0].set_xlabel('Offset Index')
    axes[0].set_ylabel('KL(Offset || Baseline)')
    axes[0].set_title('KL Divergence from Baseline Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    codebook_indices = np.arange(len(baseline))
    axes[1].bar(codebook_indices, baseline.numpy(), alpha=0.7, width=1.0)
    axes[1].set_xlabel('Used Codebook')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Baseline Distribution (Average of All Offsets)')

    plt.tight_layout()
    plt.show()

    print(f"Mean KLD: {np.mean(kld_values):.4f}")
    print(f"Std KLD:  {np.std(kld_values):.4f}")
    print(f"Min KLD:  {np.min(kld_values):.4f} (offset {offsets[np.argmin(kld_values)]})")
    print(f"Max KLD:  {np.max(kld_values):.4f} (offset {offsets[np.argmax(kld_values)]})")





# EXP NIA
def compute_label_token_cnt(trainset, model, num_labels, codebook_size, pth=None, prefix='', device='cuda'):
    label_token_cnt = np.zeros((num_labels, codebook_size))
    dl = DataLoader(trainset, batch_size=256, shuffle=False, drop_last=False)
    
    model.eval()
    with torch.inference_mode():
        for x, _, _, y in tqdm(dl):
            min_e = extract_min_e(model, x.to(device))
            code_idx = min_e.argmax(dim=-1).detach().cpu().numpy()
            np.add.at(label_token_cnt, (y.numpy(), code_idx), 1)

    P_token_given_label = label_token_cnt / label_token_cnt.sum(axis=-1, keepdims=True)
    P_label = label_token_cnt.sum(axis=-1) / label_token_cnt.sum()

    if pth is not None:
        suffix = '_vocal' if 'VOCAL' in prefix else ''
        torch.save(label_token_cnt, f'{pth}/label_token_cnt{suffix}.pt')

    return label_token_cnt, P_token_given_label, P_label


def get_proba(dataset, model, P_token_given_label, P_label, temperature=1.0, device='cuda'):
    dl = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    y_true, y_pred = [], []

    model.eval()
    with torch.inference_mode():
        for x, _, _, y in tqdm(dl):
            min_e = extract_min_e(model, x.to(device))
            code_idx = min_e.argmax(dim=-1).detach().cpu().numpy()
            y_true.append(y)
            y_pred.append(code_idx)

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    log_prior = np.log(P_label + 1e-12)
    log_likelihood = np.log(P_token_given_label + 1e-12)
    log_prob = temperature * log_prior.reshape(-1,1) + log_likelihood[:,y_pred]


    log_prob -= log_prob.max(axis=0, keepdims=True)
    prob = np.exp(log_prob)
    prob /= prob.sum(axis=0)

    return prob, y_true


def predict_by_map_with_temperature(dataset, model, P_token_given_label, P_label, temperature=1.0, device='cuda'):
    dl = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    y_true, y_pred = [], []

    model.eval()
    with torch.inference_mode():
        for x, _, _, y in tqdm(dl):
            min_e = extract_min_e(model, x.to(device))
            code_idx = min_e.argmax(dim=-1).detach().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(code_idx)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    log_prior = np.log(P_label + 1e-12)
    log_likelihood = np.log(P_token_given_label + 1e-12)
    log_prob = temperature * log_prior.reshape(-1,1) + log_likelihood[:,y_pred]

    return log_prob.argmax(axis=0), y_true



def fit_temperature_with_log(validset, model, P_token_given_label, P_label,
                              temperatures=np.round(np.linspace(0.00, 1.00, 50), 5),
                              metric='f1', device='cuda', run=None):
    best_T, best_score, = 1.0, 0.0
    log_table = wandb.Table(columns=['temperature', 'score']) if (run is not None or wandb.run is not None) else None

    for T in temperatures:
        y_pred, y_true = predict_by_map_with_temperature(validset, model, P_token_given_label, P_label, T, device)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, average='macro')
        elif metric == 'acc':
            score = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"T={T:.2f}: score={score:.5f}")

        if log_table is not None:
            log_table.add_data(T, score)

        if score > best_score:
            best_score, best_T = score, T

    log_dict = {'best_temperature': best_T, 'best_val_score': best_score}
    if log_table is not None:
        log_dict['temperature_search'] = log_table
    _log(log_dict, run)

    print(f"\nBest T={best_T}, best_score={best_score:.5f}")
    return best_T


def evaluate_with_log(testset, model, P_token_given_label, P_label, temperature,
                       num_labels, idx2label, prefix='test', device='cuda', run=None):
    y_prob, y_true = get_proba(testset, model, P_token_given_label, P_label, temperature, device)
    y_score = y_prob.T        # (N, 6)
    y_pred = y_score.argmax(axis=1)

    y_true_onehot = label_binarize(y_true, classes=list(range(num_labels)))
    ap_per_class = average_precision_score(y_true_onehot, y_score, average=None)
    mAP = average_precision_score(y_true_onehot, y_score, average='macro')

    report = classification_report(y_true, y_pred,
                                   target_names=list(idx2label.values()),
                                   output_dict=True)
    print(classification_report(y_true, y_pred, target_names=list(idx2label.values())))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm*100, display_labels=idx2label.values())
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    disp.im_.colorbar.remove()
    plt.tight_layout()

    log_dict = {
        f'{prefix}/mAP': mAP,
        f'{prefix}/accuracy': report['accuracy'],
        f'{prefix}/f1_macro_avg': report['macro avg']['f1-score'],
        f'{prefix}/f1_weighted': report['weighted avg']['f1-score'],
        f'{prefix}/confusion_matrix': wandb.Image(fig),
    }
    for i, ap in enumerate(ap_per_class):
        log_dict[f'{prefix}/AP_{idx2label[i]}'] = ap
        print(f"{idx2label[i]}: AP={ap:.4f}")
    print(f"\nmAP: {mAP:.4f}")

    for label_name in idx2label.values():
        if label_name in report:
            log_dict[f'{prefix}/f1_{label_name}'] = report[label_name]['f1-score']
            log_dict[f'{prefix}/precision_{label_name}'] = report[label_name]['precision']
            log_dict[f'{prefix}/recall_{label_name}'] = report[label_name]['recall']

    _log(log_dict, run)
    plt.close(fig)

    return y_true, y_pred, y_score


def plot_confusion_matrix(y_true, y_pred, idx2label):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm, display_labels=idx2label.values())
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.show()




# AE EXP NIA
def compute_label_token_cnt_ae(trainset, model, scaler, umap_embed, kmeans,
                                num_labels, n_codes, pth=None, prefix='', device='cuda'):
    label_token_cnt = np.zeros((num_labels, n_codes))
    dl = DataLoader(trainset, batch_size=1024, shuffle=False, drop_last=False)

    model.eval()
    with torch.inference_mode():
        for x, _, _, y in tqdm(dl):
            _, _, z = model(x.to(device))
            z = z.squeeze().detach().cpu().numpy()
            z_scaled = scaler.transform(z)
            z_umap = umap_embed.transform(z_scaled)
            code_idx = kmeans.predict(z_umap)
            np.add.at(label_token_cnt, (y.numpy(), code_idx), 1)

    P_token_given_label = label_token_cnt / label_token_cnt.sum(axis=-1, keepdims=True)
    P_label = label_token_cnt.sum(axis=-1) / label_token_cnt.sum()

    if pth is not None:
        suffix = '_vocal' if 'VOCAL' in prefix else ''
        torch.save(label_token_cnt, f'{pth}/label_token_cnt_ae_{n_codes}{suffix}.pt')

    return label_token_cnt, P_token_given_label, P_label


def get_proba_ae(dataset, model, scaler, umap_embed, kmeans,
                 P_token_given_label, P_label, temperature=1.0, device='cuda'):
    dl = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False)
    y_true, y_pred = [], []

    model.eval()
    with torch.inference_mode():
        for x, _, _, y in tqdm(dl):
            _, _, z = model(x.to(device))
            z = z.squeeze().detach().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(z)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    z_scaled = scaler.transform(y_pred)
    z_umap = umap_embed.transform(z_scaled)
    code_idx = kmeans.predict(z_umap)

    log_prior = np.log(P_label + 1e-12)
    log_likelihood = np.log(P_token_given_label + 1e-12)
    log_prob = temperature * log_prior.reshape(-1, 1) + log_likelihood[:, code_idx]

    log_prob -= log_prob.max()
    prob = np.exp(log_prob)
    prob /= prob.sum(axis=0)

    return prob, y_true


def predict_by_map_with_temperature_ae(dataset, model, scaler, umap_embed, kmeans,
                                        P_token_given_label, P_label, temperature=1.0, device='cuda'):
    dl = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False)
    y_true, y_pred = [], []

    model.eval()
    with torch.inference_mode():
        for x, _, _, y in tqdm(dl):
            _, _, z = model(x.to(device))
            z = z.squeeze().detach().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(z)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    z_scaled = scaler.transform(y_pred)
    z_umap = umap_embed.transform(z_scaled)
    code_idx = kmeans.predict(z_umap)

    log_prior = np.log(P_label + 1e-12)
    log_likelihood = np.log(P_token_given_label + 1e-12)
    log_prob = temperature * log_prior.reshape(-1, 1) + log_likelihood[:, code_idx]

    return log_prob.argmax(axis=0), y_true


def fit_temperature_with_log_ae(validset, model, scaler, umap_embed, kmeans,
                                 P_token_given_label, P_label,
                                 temperatures=np.round(np.linspace(0.00, 1.00, 50), 5),
                                 metric='f1', device='cuda', run=None):
    best_T, best_score = 1.0, 0.0
    log_table = wandb.Table(columns=['temperature', 'score']) if (run is not None or wandb.run is not None) else None

    for T in temperatures:
        y_pred, y_true = predict_by_map_with_temperature_ae(
            validset, model, scaler, umap_embed, kmeans, P_token_given_label, P_label, T, device)

        score = f1_score(y_true, y_pred, average='macro') if metric == 'f1' else np.mean(y_true == y_pred)
        print(f"T={T:.2f}: score={score:.5f}")

        if log_table is not None:
            log_table.add_data(T, score)

        if score > best_score:
            best_score, best_T = score, T

    log_dict = {'best_temperature': best_T, 'best_val_score': best_score}
    if log_table is not None:
        log_dict['temperature_search'] = log_table
    _log(log_dict, run)

    print(f"\nBest T={best_T}, best_score={best_score:.5f}")
    return best_T


def evaluate_with_log_ae(testset, model, scaler, umap_embed, kmeans,
                          P_token_given_label, P_label, temperature,
                          num_labels, idx2label, prefix='test_ae', device='cuda', run=None):
    y_prob, y_true = get_proba_ae(testset, model, scaler, umap_embed, kmeans,
                                   P_token_given_label, P_label, temperature, device)
    y_score = y_prob.T
    y_pred = y_score.argmax(axis=1)

    y_true_onehot = label_binarize(y_true, classes=list(range(num_labels)))
    ap_per_class = average_precision_score(y_true_onehot, y_score, average=None)
    mAP = average_precision_score(y_true_onehot, y_score, average='macro')

    report = classification_report(y_true, y_pred,
                                   target_names=list(idx2label.values()),
                                   output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=list(idx2label.values()), zero_division=0))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm*100, display_labels=idx2label.values())
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    disp.im_.colorbar.remove()
    plt.tight_layout()

    log_dict = {
        f'{prefix}/mAP': mAP,
        # f'{prefix}/accuracy': report['accuracy'],
        f'{prefix}/f1_macro_avg': report['macro avg']['f1-score'],
        # f'{prefix}/f1_weighted': report['weighted avg']['f1-score'],
        f'{prefix}/confusion_matrix': wandb.Image(fig),
    }
    for i, ap in enumerate(ap_per_class):
        log_dict[f'{prefix}/AP_{idx2label[i]}'] = ap
        print(f"{idx2label[i]}: AP={ap:.4f}")
    print(f"\nmAP: {mAP:.4f}")

    for label_name in idx2label.values():
        if label_name in report:
            log_dict[f'{prefix}/f1_{label_name}'] = report[label_name]['f1-score']
            log_dict[f'{prefix}/precision_{label_name}'] = report[label_name]['precision']
            log_dict[f'{prefix}/recall_{label_name}'] = report[label_name]['recall']

    _log(log_dict, run)
    plt.close(fig)

    return y_true, y_pred, y_score