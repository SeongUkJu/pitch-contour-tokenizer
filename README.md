# Pitch Contour Tokenization using VQ-VAE and Its Application on Korean Traditional Music Analysis

**Seonguk Ju, Seola Cho, Sooin Chung, Danbinaerin Han, Dasaem Jeong**

The official implementation of the ISMIR 2026 paper *"Pitch Contour
Tokenization using VQ-VAE and Its Application on Korean Traditional
Music Analysis"*.

[**Paper**](#) &nbsp;|&nbsp;
[**Demo**](https://seongukju.github.io/pitch-contour-tokenizer-page/) &nbsp;|&nbsp;
[**Data & Weights**](https://github.com/SeongUkJu/pitch-contour-tokenizer/releases)
<!-- TODO(camera-ready): paper / arXiv links -->

[![Code License: MIT](https://img.shields.io/badge/code%20license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
<!-- TODO(camera-ready): arXiv badge -->

The model learns a discrete vocabulary of local pitch-contour patterns
directly from unlabeled audio:

* **Contour tokenizer** — A 1D convolutional VQ-VAE over fixed-length
  (1.28 s, 100 Hz) pitch-contour segments. Each median-subtracted
  segment is encoded, quantized to one of 256 codebook entries — its
  discrete token — and decoded back to a contour.
* **Transformation-minimized loss** — The reconstruction is evaluated
  under the best alignment among candidate temporal and pitch-domain
  transformations (tempo, zoom, y-offset) and only the minimum error is
  back-propagated, so tokens stay stable across segmentation positions
  and small variations in timing and pitch range.

| Config | Model | Loss | Notes |
|---|---|---|---|
| `contour_ae` | Conv1DAE | MSE | AE baseline (+ post-hoc KMeans codes) |
| `contour_vqvae` | Conv1DVQVAE | MSE | VQ-VAE baseline |
| `foffset_vqvae` | OffsetConv1DVQVAE | Offset-MSE | tempo + zoom + y-offset candidates (paper main) |
| `xoffset_vqvae` | OffsetConv1DVQVAE | Offset-MSE | tempo offsets only |

---

## Install

**With [uv](https://docs.astral.sh/uv/) (recommended)** — installs the exact
locked versions from `uv.lock`:

```bash
# one-time uv install, if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync   # creates .venv/ and installs all dependencies
```

Then either prefix commands with `uv run` (no activation needed), e.g.
`uv run python src/train.py ...`, or activate the environment with
`source .venv/bin/activate` and use the commands below as written.

**With pip:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10–3.13. Tested with Python 3.10, PyTorch 2.8 (CUDA 12.8)
and 2.13 (CPU).

---

## Download data & weights

The paper's pitch-contour datasets and pretrained checkpoints are attached
to [GitHub Releases](https://github.com/SeongUkJu/pitch-contour-tokenizer/releases):

| Asset | Contents | Size |
|---|---|---|
| `checkpoints.tar.gz` | 4 pretrained models (`best_model.pt` + training `config.yaml` each) | ~1 MB |
| `inhouse_data.tar.gz` | pansori training corpus (CREPE contour CSVs) + `rename_map.csv` | ~2.0 GB |
| `nia_pansori_meta.tar.gz` | NIA Gugak contours, annotated pansori contours, metadata CSVs | ~0.5 GB |
| `SHA256SUMS.txt` | checksums for the assets above | — |

Extract from the repository root:

```bash
sha256sum -c SHA256SUMS.txt              # optional integrity check
tar xzf checkpoints.tar.gz               # → weights/<model>/
mkdir -p data
tar xzf inhouse_data.tar.gz -C data/     # → data/inhouse_data/
tar xzf nia_pansori_meta.tar.gz -C data/ # → data/{nia_gugak_data,pansori_annotated_data,meta}/
```

Checkpoint ↔ config ↔ paper mapping:

| Checkpoint dir | Config | Paper method |
|---|---|---|
| `weights/autoencoder` | `contour_ae` | Autoencoder baseline |
| `weights/vqvae` | `contour_vqvae` | VQ-VAE |
| `weights/temporal_align_vqvae` | `xoffset_vqvae` | + temporal align |
| `weights/all_transformation_vqvae` | `foffset_vqvae` | + all transformations (paper main) |

With the released data in place, the presets `data=crepe`, `split=manifest`
and `exp=nia_pair` point at it directly, e.g.
`python src/train.py --config-name foffset_vqvae data=crepe split=manifest exp=nia_pair exp.run=true`.

---

## Reproducing the paper

### 1 · Prepare pitch-contour CSVs

Use the released datasets above, or provide your own per-clip pitch-contour
CSVs (e.g. [PESTO](https://github.com/SonyCSLParis/pesto) or
[CREPE](https://github.com/marl/crepe) output) with at least the columns:

- `frequency` — F0 in Hz per frame
- `confidence` — voicing confidence in [0, 1]

Frames are assumed to be 10 ms (100 Hz). CSVs are discovered recursively
(`rglob`) under `data.data_pth`, so any directory nesting works.
Frames with `confidence < 0.5` are masked/interpolated (see
`src/configs/dataset/full_threshold.yaml`).

### 2 · Train

```bash
python src/train.py --config-name foffset_vqvae data.data_pth=/path/to/csvs
```

- CLI dotted overrides take precedence over the YAML files; see
  `src/configs/` for the full set of knobs.
- Clips are split 8:1:1 into train/valid/test with a fixed shuffle
  (`split.params.split_seed`, independent of `random_seed`), so seed
  sweeps share an identical split. On the released corpus, add
  `split=manifest` to use the paper's exact partition
  (`splits/inhouse_crepe_split.csv`).
- WandB is disabled by default. Enable with `wandb.project=<name>`
  (requires login).
- Checkpoints + resolved `config.yaml` land in `weights/<run_name>/`.
- VQ codebooks are initialized with k-means over encoder latents
  (`model.params.init=kmeans`).

### 3 · Post-training experiments

Add `exp.run=true` to run the paper's analyses after training:

- **KLD & Code-Flip** (token consistency) — run on the held-out test
  split of *your* data; no extra assets needed. For the autoencoder,
  the post-hoc token space (scaler/UMAP/k-means) is fit on the train
  split and applied to the test split.
- **NIA sigimsae classification** — requires the NIA Gugak dataset
  (available via [AI-Hub](https://aihub.or.kr), Korea) plus a metadata
  CSV with columns `filename,onset,offset,label,split`, where `label`
  is one of `southern_flick`, `southern_vibrato`, `western_vibrato`,
  `soft_vibrato`, `upward_accent`, `descending_slide`.
  Fill in the `nia:`/`nia_vocal:` blocks in `src/configs/exp/default.yaml`
  (a commented template is provided there). When unset, these stages are
  skipped.

### 4 · Inference

```bash
python src/infer.py --run_dir weights/all_transformation_vqvae --input /path/to/contour.csv --out out/
```

Works with your own training runs (`weights/<run_name>`) or the released
checkpoints. Writes `reconstructions.csv`
(`window, frame, target_midi, recon_midi`) and target-vs-reconstruction plots.

### Reproducibility notes

- **Released checkpoints → paper numbers.** Inference and the post-training
  analyses are deterministic (the analysis-side k-means is pinned to a single
  thread via `threadpoolctl`), so the released checkpoints reproduce the
  reported numbers exactly.
- **The paper's split ships with the repo.** The released corpus is a
  (romanized) renamed copy of the internal data, so the default
  `split=random` shuffle sorts files differently and yields a different
  8:1:1 partition. `split=manifest` (`splits/inhouse_crepe_split.csv`)
  reproduces the paper's exact train/valid/test membership — use it
  whenever comparing against the released checkpoints or paper numbers.
- **Retraining VQ models does not reproduce the released weights.** The VQ
  codebook is initialized with k-means over encoder latents, and scikit-learn's
  multi-threaded k-means is non-deterministic across runs even with a fixed
  `random_state`: parallel floating-point reductions change the summation
  order, and the resulting tiny differences in the initial codebook are
  amplified over training. Retraining with identical code, data, and seed
  therefore converges to different (equally valid) weights; qualitative
  conclusions were consistent across reruns.
- **Retraining the autoencoder is bit-exact reproducible** in the same
  software/hardware environment (fixed seed, deterministic cuDNN) — its
  training involves no k-means.
- Released checkpoints were trained with Python 3.10, PyTorch 2.8.0
  (CUDA 12.8), and scikit-learn 1.6.1 on an NVIDIA RTX 4090.

---

## Directory layout

```
src/
    train.py                  # config-driven training entry point
    trainers.py               # AETrainer / VQTrainer loops
    experiments.py            # post-training analyses (KLD, code-flip, NIA)
    infer.py                  # checkpoint inference on a contour CSV
    models/
        model_zoo.py          # Conv1DAE, Conv1DVQVAE, OffsetConv1DVQVAE
        modules.py            # encoder / decoder / vector-quantizer blocks
    losses/loss_zoo.py        # MSELoss, OffsetMSELoss, VectorQuantizeLoss
    datasets/dataset_zoo.py   # contour-CSV datasets (+ NIA / Pansori variants)
    metrics/metric_zoo.py     # Perplexity, Accuracy
    schedulers/scheduler_zoo.py  # warmup + cosine LR schedule
    utils/                    # data split, k-means init, inference helpers
    configs/                  # Hydra configs (one YAML per model preset)

weights/                      # pretrained checkpoints (from Releases; gitignored)
    autoencoder/              # best_model.pt + config.yaml per model
    vqvae/
    temporal_align_vqvae/
    all_transformation_vqvae/ # paper main

data/                         # datasets (from Releases; gitignored)
    inhouse_data/             # pansori training corpus (CREPE contours)
    nia_gugak_data/           # NIA Gugak contours
    pansori_annotated_data/   # mode-annotated pansori contours
    meta/                     # NIA & pansori metadata CSVs

pyproject.toml + uv.lock      # locked environment (uv)
requirements.txt              # pip fallback
```

---

## Citation

```bibtex
% TODO(camera-ready): replace with the final ISMIR 2026 BibTeX entry
@inproceedings{ju2026pitchcontour,
  title     = {Pitch Contour Tokenization using {VQ-VAE} and Its Application
               on Korean Traditional Music Analysis},
  author    = {Ju, Seonguk and Cho, Seola and Chung, Sooin and
               Han, Danbinaerin and Jeong, Dasaem},
  booktitle = {Proceedings of the 27th International Society for Music
               Information Retrieval Conference (ISMIR)},
  year      = {2026},
}
```

## License

[MIT](LICENSE).
