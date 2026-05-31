# ViT from Scratch

Implementation PyTorch d'un Vision Transformer ecrite comme projet d'etude. Le but est simple : reconstruire les mecanismes centraux du ViT, puis les reutiliser dans trois cadres d'apprentissage differents avec un code assez court pour rester lisible.

Le meme backbone sert ici a :

- la classification supervisee,
- un masked autoencoder (MAE),
- une variante teacher-student de type DINO.

Ce depot est pense pour quelqu'un qui veut lire le code en meme temps que les papiers, suivre le passage image -> patches -> tokens -> attention, et comparer ce qui change vraiment entre ces trois objectifs.

## Motivation

Les Vision Transformers sont devenus faciles a lancer avec des bibliotheques existantes. En revanche, il est moins facile de voir clairement ce que fait chaque bloc quand tout arrive deja empaquete.

J'ai construit ce projet pour garder cette partie visible. Le code isole les briques importantes, les notebooks montrent les tenseurs et les figures intermediaires, et les recettes d'entrainement restent a petite echelle pour qu'on puisse les relire, les modifier et les rerunner sans transformer le depot en pipeline de benchmark.

## Ce que le depot couvre

- patch embedding et tokenisation d'images en sequences,
- embeddings positionnels learned, sine-cosine et RoPE / RoPE2D,
- multi-head self-attention et blocs encodeur pre-norm,
- classification supervisee avec class token,
- masked autoencoding avec reconstruction de patches masques,
- DINO avec teacher EMA, multi-crop views et evaluation kNN,
- visualisations d'attention, reconstructions et courbes d'entrainement.

## References de depart

Le projet est parti des papiers, puis a ete transforme en modules PyTorch et en notebooks d'explication.

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer](https://arxiv.org/abs/2104.09864)

## Structure du depot

```text
src/vit_from_scratch/
  model.py, config.py                    # VisionTransformer, configuration
  patch_embedding.py, embedding.py       # patchification, positions, RoPE
  attention.py, encoder.py, mlp.py       # blocs Transformer
  classification.py                      # entrainement supervise
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

configs/training/                        # configs YAML par approche et dataset
tests/                                   # tests unitaires, smoke tests, notebooks clean
scripts/download_external_images.py      # images externes pour figures qualitatives
```

## Jeux de donnees et artefacts

Le depot versionne le code, les configs et les notebooks sans outputs. Les datasets, checkpoints, runs et images externes restent locaux et sont ignores par Git. En pratique, les dossiers `data/` et `runs/` sont regeneres localement.

Les configurations couvrent des experiences sur `fake`, `CIFAR-10`, `STL-10` et `TinyImageNet`, selon l'approche choisie.

## Demarrage rapide

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -q
python -m vit_from_scratch.train --config configs/training/classification_fake.yaml
```

Le chemin le plus utile pour decouvrir le projet est :

1. `notebooks/01_vit_building_blocks.ipynb`
2. `notebooks/03_training_walkthrough.ipynb`
3. `notebooks/04_training_methods.ipynb`
4. `notebooks/experiment_results.ipynb`

## Organisation du code

Le code est structure autour d'un backbone ViT unique et de tetes ou routines d'entrainement specialisees.

- `model.py` definit le ViT de base.
- `classification.py` gere le cas supervise.
- `masked_autoencoder.py` ajoute l'objectif de reconstruction.
- `dino.py` gere la logique teacher-student et les vues multi-crop.
- `evaluation.py` regroupe les metriques et les sorties qualitatives.
- `train.py` fournit une CLI commune pour lancer les differentes approches a partir des fichiers YAML.

L'idee est de comparer les methodes sans dupliquer toute l'architecture a chaque fois.

## Resultats et lecture des sorties

Le projet produit des runs lisibles plutot que de gros tableaux de performance. Les sorties utiles sont :

- les courbes d'entrainement et de validation,
- les cartes d'attention,
- les reconstructions MAE,
- les figures sur images externes,
- les sondes kNN pour juger la qualite des representations en DINO.

Le notebook `notebooks/experiment_results.ipynb` sert de guide de lecture pour ces artefacts et explique ce qu'il faut regarder dans un run reussi, mediocre ou instable.

## Limitations

- Les experiences sont volontairement petites et ont ete developpees sur Mac M2.
- Le depot n'essaie pas de reproduire les recettes ImageNet a grande echelle.
- Aucun poids pre-entraine n'est publie.
- Les notebooks sont une partie centrale du livrable, pas un simple accompagnement.
- MAE et DINO sont implementes comme versions pedagogiques et entrainables, pas comme reproductions industrielles des papiers originaux.

## Statut

Le socle du projet est en place : backbone ViT, trois approches d'entrainement, configs YAML, tests, et notebooks d'explication.

## Licence

Voir [LICENSE](LICENSE).
