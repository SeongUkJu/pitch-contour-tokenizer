import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.dataset_zoo import FullPitchThresholdDataset
from utils.infer_utils import get_model, get_min_offset_contour


def main():
    p = argparse.ArgumentParser(description="Reconstruct pitch contours with a trained checkpoint.")
    p.add_argument('--run_dir', required=True, help='training run dir containing config.yaml and best_model.pt')
    p.add_argument('--input', required=True, help='a contour CSV file or a directory of CSVs')
    p.add_argument('--out', default='out', help='output directory')
    p.add_argument('--num_plots', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / 'config.yaml')
    model = get_model(str(run_dir)).to(device).eval()

    in_pth = Path(args.input)
    data_arg = [in_pth] if in_pth.is_file() else str(in_pth)
    ds_params = OmegaConf.to_container(cfg.dataset.params, resolve=True)
    dataset = FullPitchThresholdDataset(data_arg, **ds_params, is_valid=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    n_plotted, rows, w_idx = 0, [], 0
    with torch.inference_mode():
        for x, weight, meta in loader:
            x = x.to(device)
            out = model(x)
            if 'VQ' in model.__class__.__name__:
                target, pred = out[0][0], out[1][0]
            else:
                target, pred = out[0], out[1]
            if pred.size() != target.size():   # offset models: (b,c,rolls,t) candidates
                pred = get_min_offset_contour(pred, target, model.rf)
            target, pred = target.detach().cpu(), pred.detach().cpu()

            for i in range(target.shape[0]):
                t_midi = target[i, 0].numpy() * 12.0   # norm_midi=False -> tonic 0, norm = midi/12
                p_midi = pred[i, 0].numpy() * 12.0
                rows.append(pd.DataFrame({'window': w_idx,
                                          'frame': np.arange(len(t_midi)),
                                          'target_midi': t_midi,
                                          'recon_midi': p_midi}))
                if n_plotted < args.num_plots:
                    fig = plt.figure(figsize=(12, 4))
                    plt.plot(t_midi, label='target')
                    plt.plot(p_midi, label='recon')
                    plt.xlabel('frame (10 ms)')
                    plt.ylabel('MIDI')
                    plt.legend()
                    plt.tight_layout()
                    fig.savefig(out_dir / f'recon_{n_plotted}.png')
                    plt.close(fig)
                    n_plotted += 1
                w_idx += 1

    pd.concat(rows, ignore_index=True).to_csv(out_dir / 'reconstructions.csv', index=False)
    print(f"Wrote {w_idx} windows to {out_dir / 'reconstructions.csv'} and {n_plotted} plots.")


if __name__ == '__main__':
    main()
