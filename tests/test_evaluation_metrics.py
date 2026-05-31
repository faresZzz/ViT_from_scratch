from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from vit_from_scratch.evaluation import (
    compute_attention_entropy_and_peak_mass,
    compute_classification_metrics,
    compute_feature_std_mean,
    compute_knn_accuracy,
    compute_mae_reconstruction_metrics,
    compute_softmax_diagnostics,
)


def test_compute_classification_metrics_returns_confusion_and_topk():
    logits = torch.tensor(
        [
            [5.0, 1.0, 0.5],
            [0.5, 4.0, 3.0],
            [0.2, 2.0, 4.5],
            [3.0, 2.5, 2.0],
        ]
    )
    labels = torch.tensor([0, 2, 1, 1])

    metrics = compute_classification_metrics(logits, labels, num_classes=3)

    assert metrics["confusion_matrix"] == [
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
    assert metrics["top1_accuracy"] == pytest.approx(0.25)
    assert metrics["top3_accuracy"] == pytest.approx(1.0)
    assert metrics["topk"] == 3
    assert metrics["per_class_accuracy"] == pytest.approx([1.0, 0.0, 0.0])
    assert 0.0 < metrics["mean_confidence"] < 1.0


def test_compute_mae_reconstruction_metrics_returns_expected_scalars():
    target = torch.zeros(2, 3, 4, 4)
    reconstructed = torch.full_like(target, 0.5)

    metrics = compute_mae_reconstruction_metrics(reconstructed, target, max_value=1.0)

    assert metrics["reconstruction_mse"] == pytest.approx(0.25)
    assert metrics["reconstruction_mae"] == pytest.approx(0.5)
    assert metrics["reconstruction_psnr"] == pytest.approx(10.0 * math.log10(4.0))


def test_compute_softmax_diagnostics_and_feature_std_mean():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    features = torch.tensor(
        [
            [1.0, 3.0, 5.0],
            [3.0, 5.0, 7.0],
        ]
    )

    diagnostics = compute_softmax_diagnostics(logits)

    assert diagnostics["entropy"] > 0.0
    assert 0.5 < diagnostics["confidence"] < 1.0
    assert compute_feature_std_mean(features) == pytest.approx(1.4142135, rel=1e-4)


def test_compute_knn_accuracy_handles_tiny_cosine_problem():
    train_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    train_labels = torch.tensor([0, 0, 1, 1])
    val_features = torch.tensor(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ]
    )
    val_labels = torch.tensor([0, 1])

    accuracy = compute_knn_accuracy(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        k=5,
    )

    assert accuracy == pytest.approx(1.0)


def test_compute_attention_entropy_and_peak_mass_from_attention_maps():
    attention_maps = (
        torch.tensor(
            [
                [
                    [
                        [0.0, 0.8, 0.2],
                        [0.3, 0.4, 0.3],
                        [0.2, 0.5, 0.3],
                    ]
                ]
            ],
            dtype=torch.float32,
        ),
    )

    diagnostics = compute_attention_entropy_and_peak_mass(attention_maps)

    assert diagnostics["attention_entropy"] > 0.0
    assert diagnostics["attention_peak_mass"] == pytest.approx(0.8)
