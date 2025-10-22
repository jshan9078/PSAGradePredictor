"""
Imbalanced Sampler for PSA Grading Dataset

Paper §8.3: "Mini-batches sample grades with probability ∝ 1/freq(k)^η"
"""

import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import numpy as np


def create_imbalanced_sampler(dataset, eta=0.5, replacement=True):
    """
    Create a WeightedRandomSampler that oversample rare grades.

    Args:
        dataset: PSADataset instance
        eta: Exponent for inverse frequency weighting (default 0.5)
             η=0: uniform sampling
             η=1: fully inverse-frequency weighted
             η=0.5: square-root weighted (recommended)
        replacement: Whether to sample with replacement

    Returns:
        WeightedRandomSampler instance
    """
    # Extract grade labels from dataset
    grades = []
    for item in dataset.items:
        if isinstance(item, dict):
            grades.append(item['grade'])
        else:
            grades.append(item[2])  # Tuple format: (front, back, grade)

    # Compute grade frequencies
    grade_counts = Counter(grades)
    total_samples = len(grades)

    # Compute sampling weights: P(k) ∝ 1 / freq(k)^η
    weights = []
    for grade in grades:
        freq = grade_counts[grade] / total_samples
        weight = 1.0 / (freq ** eta)
        weights.append(weight)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=replacement
    )

    return sampler


def print_grade_distribution(dataset, sampler=None, num_samples=10000):
    """
    Print the grade distribution with and without sampling.

    Args:
        dataset: PSADataset instance
        sampler: Optional sampler to test
        num_samples: Number of samples to draw for distribution check
    """
    # Original distribution
    grades = []
    for item in dataset.items:
        if isinstance(item, dict):
            grades.append(item['grade'])
        else:
            grades.append(item[2])

    original_counts = Counter(grades)
    print("Original Grade Distribution:")
    for grade in sorted(original_counts.keys()):
        count = original_counts[grade]
        pct = 100.0 * count / len(grades)
        print(f"  Grade {grade:2d}: {count:5d} ({pct:5.2f}%)")

    # Sampled distribution
    if sampler is not None:
        sampled_indices = list(sampler)[:num_samples]
        sampled_grades = []
        for idx in sampled_indices:
            item = dataset.items[idx]
            if isinstance(item, dict):
                sampled_grades.append(item['grade'])
            else:
                sampled_grades.append(item[2])

        sampled_counts = Counter(sampled_grades)
        print(f"\nSampled Grade Distribution ({num_samples} samples):")
        for grade in sorted(sampled_counts.keys()):
            count = sampled_counts[grade]
            pct = 100.0 * count / num_samples
            print(f"  Grade {grade:2d}: {count:5d} ({pct:5.2f}%)")
