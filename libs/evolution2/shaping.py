"""This module provides functions to shape the scores over a population of mutations."""
import torch


def identity(scores: torch.Tensor) -> torch.Tensor:
  """Use the raw scores without any transformation."""
  return scores


def scaled_rank(scores: torch.Tensor, lower: float = -0.5, upper: float = 0.5) -> torch.Tensor:
  """Transform the scores into relative ranks scaled within the interval [lower, upper]. The
  smallest score becomes `lower`, the largest score becomes `upper`, and the other scores are
  equally spaced in between."""
  size = scores.numel()
  ranks = torch.empty(size, dtype=torch.long, device=scores.device)
  ranks[scores.ravel().argsort()] = torch.arange(size, device=scores.device)
  scaled_values = torch.linspace(lower, upper, size, device=scores.device)
  return scaled_values[ranks].reshape(scores.shape)


def z_scores(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """Transform the scores into standard z-scores by subtracting the mean and dividing by the
  standard deviation (unbiased), both over all elements, whatever the shape."""
  return (scores - scores.mean()) / (scores.std() + eps)


def modified_z_scores(
  scores: torch.Tensor, consistency: float = 1.4826, eps: float = 1e-6
) -> torch.Tensor:
  """Transform the scores into modified z-scores by subtracting the median and dividing by the
  median absolute deviation. This metric highlights outliers in a skewed distribution using more
  extreme values."""
  median = torch.median(scores)
  deviations = scores - median
  mad = torch.median(torch.abs(deviations))
  return deviations / (consistency * mad + eps)
