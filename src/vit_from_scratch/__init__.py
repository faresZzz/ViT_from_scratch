"""vit_from_scratch — pedagogical Vision Transformer building blocks."""

from importlib import import_module

from vit_from_scratch.attention import MultiHeadSelfAttention
from vit_from_scratch.artifacts import (
    ExperimentPaths,
    create_experiment_run,
    find_latest_checkpoint,
    load_checkpoint,
    load_history,
    resolve_resume_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
    save_figure,
    save_history,
    save_named_checkpoint,
    save_json,
)
from vit_from_scratch.classification import (
    build_classification_model,
    classification_eval_step,
    classification_loss,
    classification_train_step,
    train_classification,
)
from vit_from_scratch.config import ViTConfig
from vit_from_scratch.data import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    DataLoaders,
    STL10_CLASSES,
    STL10_MEAN,
    STL10_STD,
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
    TinyImageNetValDataset,
    build_dataloaders,
    build_transforms,
    limit_dataset,
)
from vit_from_scratch.embedding import (
    CosinePositionEmbedding,
    LearnedPositionEmbedding,
    PositionEmbeddingType,
    apply_rope,
    build_rope_cache,
    build_rope_cache_2d,
)
from vit_from_scratch.dino import (
    DINOCenter,
    DINOViewConfig,
    build_dino_views,
    dino_loss,
    dino_train_step,
    train_dino,
    update_teacher_ema,
)
from vit_from_scratch.evaluation import (
    compute_attention_entropy_and_peak_mass,
    compute_classification_metrics,
    compute_feature_std_mean,
    compute_knn_accuracy,
    compute_mae_reconstruction_metrics,
    compute_softmax_diagnostics,
    evaluate_classification_model,
    evaluate_dino,
    evaluate_external_dino_attention,
    evaluate_masked_autoencoder,
    predict_external_classification_images,
    reconstruct_external_images,
)
from vit_from_scratch.external_images import list_external_images, load_external_images
from vit_from_scratch.encoder import TransformerEncoderBlock
from vit_from_scratch.masked_autoencoder import (
    MaskedAutoencoder,
    mae_eval_step,
    mae_train_step,
    masked_reconstruction_loss,
    patchify,
    random_patch_mask,
    train_masked_autoencoder,
    unpatchify,
)
from vit_from_scratch.mlp import MLP
from vit_from_scratch.model import VisionTransformer
from vit_from_scratch.patch_embedding import PatchEmbedding
from vit_from_scratch.progress import iter_progress
from vit_from_scratch.run_utils import (
    RunValidationConfig,
    aggregate_weighted_metrics,
    best_metric_from_history,
    close_figure,
    find_named_checkpoint,
    paths_from_checkpoint,
    to_serializable,
    validate_training_run_args,
)
from vit_from_scratch.training_loop import (
    append_history,
    build_checkpoint_state,
    cosine_momentum_schedule,
    load_resume_state,
    prepare_training_run,
    run_epoch,
    save_figure_safely,
    update_best_metric,
)
from vit_from_scratch.training import (
    accuracy,
    evaluate,
    get_device,
    move_batch_to_device,
    set_seed,
    train,
    train_one_epoch,
)
from vit_from_scratch.visualization import (
    make_patch_grid,
    plot_class_attention,
    plot_confusion_matrix,
    plot_dino_attention_diagnostics,
    plot_predictions,
    plot_reconstruction_errors,
    plot_training_curves,
    unnormalize_image,
)

__version__ = "0.1.0"

_LAZY_MODULE_EXPORTS = {
    "classification": "vit_from_scratch.classification",
    "mae": "vit_from_scratch.masked_autoencoder",
    "dino": "vit_from_scratch.dino",
}


def __getattr__(name: str):
    """Load heavy training modules only when users request the module object."""

    if name in _LAZY_MODULE_EXPORTS:
        module = import_module(_LAZY_MODULE_EXPORTS[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ViTConfig",
    "PatchEmbedding",
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "STL10_CLASSES",
    "STL10_MEAN",
    "STL10_STD",
    "TINY_IMAGENET_MEAN",
    "TINY_IMAGENET_STD",
    "TinyImageNetValDataset",
    "DataLoaders",
    "classification",
    "mae",
    "dino",
    "build_transforms",
    "limit_dataset",
    "build_dataloaders",
    "ExperimentPaths",
    "create_experiment_run",
    "find_latest_checkpoint",
    "load_checkpoint",
    "resolve_resume_checkpoint",
    "save_json",
    "save_history",
    "load_history",
    "save_checkpoint",
    "save_named_checkpoint",
    "save_best_checkpoint",
    "save_figure",
    "PositionEmbeddingType",
    "LearnedPositionEmbedding",
    "CosinePositionEmbedding",
    "build_rope_cache",
    "build_rope_cache_2d",
    "apply_rope",
    "build_classification_model",
    "classification_loss",
    "classification_train_step",
    "classification_eval_step",
    "train_classification",
    "MaskedAutoencoder",
    "patchify",
    "unpatchify",
    "random_patch_mask",
    "masked_reconstruction_loss",
    "mae_train_step",
    "mae_eval_step",
    "train_masked_autoencoder",
    "DINOCenter",
    "DINOViewConfig",
    "build_dino_views",
    "dino_loss",
    "update_teacher_ema",
    "dino_train_step",
    "train_dino",
    "compute_classification_metrics",
    "compute_mae_reconstruction_metrics",
    "compute_softmax_diagnostics",
    "compute_feature_std_mean",
    "compute_knn_accuracy",
    "compute_attention_entropy_and_peak_mass",
    "evaluate_classification_model",
    "evaluate_masked_autoencoder",
    "evaluate_dino",
    "predict_external_classification_images",
    "reconstruct_external_images",
    "evaluate_external_dino_attention",
    "list_external_images",
    "load_external_images",
    "MultiHeadSelfAttention",
    "MLP",
    "TransformerEncoderBlock",
    "VisionTransformer",
    "iter_progress",
    "RunValidationConfig",
    "to_serializable",
    "paths_from_checkpoint",
    "best_metric_from_history",
    "validate_training_run_args",
    "aggregate_weighted_metrics",
    "find_named_checkpoint",
    "close_figure",
    "run_epoch",
    "prepare_training_run",
    "load_resume_state",
    "append_history",
    "build_checkpoint_state",
    "update_best_metric",
    "save_figure_safely",
    "cosine_momentum_schedule",
    "set_seed",
    "get_device",
    "move_batch_to_device",
    "accuracy",
    "train_one_epoch",
    "evaluate",
    "train",
    "unnormalize_image",
    "make_patch_grid",
    "plot_training_curves",
    "plot_predictions",
    "plot_class_attention",
    "plot_confusion_matrix",
    "plot_reconstruction_errors",
    "plot_dino_attention_diagnostics",
    "__version__",
]
