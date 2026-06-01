# ViT from Scratch

A PyTorch implementation of the Vision Transformer, written as a study project. The idea: rebuild the core ViT mechanisms yourself, then reuse them across three different training paradigms. The code stays short enough to actually read.

The same backbone serves for:

- supervised classification,
- a masked autoencoder (MAE),
- a DINO-style teacher-student setup.

This repo is meant for someone who wants to read the code alongside the papers, follow the path from image to patches to tokens to attention, and see what actually changes between these three objectives.

> **Note:** the notebooks and code comments are written in French. The explanations, equations, and visualizations should still be followable with the code context.

## Motivation

Vision Transformers are easy to run with existing libraries. What is less easy is seeing clearly what each block does when everything comes pre-packaged.

I built this project to keep that part visible. The code isolates the important building blocks, the notebooks show intermediate tensors and figures, and the training recipes stay at small scale so you can read them, modify them, and rerun them without turning the repo into a benchmarking pipeline.

## What the repo covers

- Patch embedding and image tokenization into sequences
- Positional embeddings: learned, sine-cosine, and RoPE / RoPE 2D
- Multi-head self-attention and pre-norm encoder blocks
- Supervised classification with a class token
- Masked autoencoding with patch reconstruction
- DINO with teacher EMA, multi-crop views, and kNN evaluation
- Attention visualizations, reconstructions, and training curves

## References

The project started from the papers, then got turned into PyTorch modules and explanatory notebooks.

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2021)
- [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (Caron et al., 2021)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [RoFormer](https://arxiv.org/abs/2104.09864) (Su et al., 2021)

## Repo structure

```text
src/vit_from_scratch/
  model.py, config.py                    # VisionTransformer, configuration
  patch_embedding.py, embedding.py       # patchification, positions, RoPE
  attention.py, encoder.py, mlp.py       # Transformer blocks
  classification.py                      # supervised training
  masked_autoencoder.py                  # MAE
  dino.py                                # DINO
  train.py, training.py, training_loop.py
  data.py, evaluation.py, visualization.py
  artifacts.py, progress.py, run_utils.py

notebooks/
  01_vit_building_blocks.ipynb
  02_embedding_methods.ipynb
  03_training_walkthrough.ipynb
  04_training_methods.ipynb
  experiment_results.ipynb

configs/training/                        # YAML configs per approach and dataset
tests/                                   # unit tests, smoke tests, notebook checks
scripts/download_external_images.py      # external images for qualitative figures
```

## Datasets and artifacts

The repo versions code, configs, and notebooks without outputs. Datasets, checkpoints, runs, and external images stay local and are gitignored. In practice, the `data/` and `runs/` directories are regenerated locally.

The configurations cover experiments on `fake`, `CIFAR-10`, `STL-10`, and `TinyImageNet`, depending on the approach.

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -q
python -m vit_from_scratch.train --config configs/training/classification_fake.yaml
```

The most useful path through the project:

1. `notebooks/01_vit_building_blocks.ipynb`
2. `notebooks/03_training_walkthrough.ipynb`
3. `notebooks/04_training_methods.ipynb`
4. `notebooks/experiment_results.ipynb`

## Code organization

The code is structured around a single ViT backbone and specialized training heads or routines.

- `model.py` defines the base ViT.
- `classification.py` handles the supervised case.
- `masked_autoencoder.py` adds the reconstruction objective.
- `dino.py` handles teacher-student logic and multi-crop views.
- `evaluation.py` groups metrics and qualitative outputs.
- `train.py` provides a common CLI to launch different approaches from YAML files.

The idea is to compare methods without duplicating the architecture each time.

## Results and reading the outputs

The project produces readable runs rather than large performance tables. The useful outputs are:

- training and validation curves,
- attention maps,
- MAE reconstructions,
- figures on external images,
- kNN probes to judge representation quality in DINO.

`notebooks/experiment_results.ipynb` walks through these artifacts and explains what to look for in a successful, mediocre, or unstable run.

## Limitations

- Experiments are intentionally small and were developed on a Mac M2.
- The repo does not attempt to reproduce large-scale ImageNet recipes.
- No pretrained weights are published.
- The notebooks are a central part of the deliverable, not a side accompaniment.
- MAE and DINO are implemented as pedagogical, trainable versions, not as industrial reproductions of the original papers.

## Status

Backbone ViT, three training approaches, YAML configs, tests, and explanatory notebooks are all in place.

## License

See [LICENSE](LICENSE).
