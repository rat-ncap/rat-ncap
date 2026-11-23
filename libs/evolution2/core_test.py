import pathlib

import pandas as pd
import pytest
import torch

import evolution2
import utils.io


class QuadraticObjective(evolution2.Objective):
  """Scores parameter vectors by their negative squared distance to a fixed target."""
  def get_initial_params(self, seeds: torch.Tensor) -> torch.Tensor:
    return torch.zeros(len(seeds), 4)

  def get_score(self, params: torch.Tensor, seeds: torch.Tensor, **kwargs) -> torch.Tensor:
    return -((params - torch.arange(4.)) ** 2).sum(dim=1)


def train(
  algorithm: evolution2.Algorithm,
  log_dir: pathlib.Path,
  epochs: int,
  save_freq: int = 2,
  **kwargs,
) -> torch.Tensor:
  """Trains on `QuadraticObjective`, logging to `log_dir`, and returns the final parameters."""
  trainer = evolution2.Trainer(
    algorithm,
    QuadraticObjective(),
    epochs=epochs,
    save_freq=save_freq,
    logger=log_dir,
    **kwargs,
  )
  trainer.initialize()
  return trainer.run()


def read_params(log_dir: pathlib.Path, epoch: int) -> torch.Tensor:
  """Reads the parameters saved at `epoch`."""
  return utils.io.read_file(log_dir / 'params' / f'{epoch}.pkl')


ALGORITHMS = {
  'ars_sgd': lambda: evolution2.algos.AugmentedRandomSearchEvolutionStrategy(population_size=8),
  'ars_adam': lambda: evolution2.algos.AugmentedRandomSearchEvolutionStrategy(
    population_size=8,
    test_size=4,
    optimizer=evolution2.optimizers.Adam(),
  ),
}


@pytest.mark.parametrize('name', ALGORITHMS)
def test_resumed_run_matches_uninterrupted_run(tmp_path: pathlib.Path, name: str) -> None:
  # A run killed during epoch 5 (before saving it) resumes from epoch 4 and recomputes epoch 5.
  train(ALGORITHMS[name](), tmp_path / 'full', epochs=8)
  train(ALGORITHMS[name](), tmp_path / 'resumed', epochs=5)
  (tmp_path / 'resumed' / 'algorithm' / '5.pkl').unlink()
  train(ALGORITHMS[name](), tmp_path / 'resumed', epochs=8)

  for epoch in (6, 8):
    torch.testing.assert_close(
      read_params(tmp_path / 'resumed', epoch),
      read_params(tmp_path / 'full', epoch),
      rtol=0,
      atol=0,
    )


def test_resume_skips_an_unreadable_newest_state(tmp_path: pathlib.Path) -> None:
  # A run killed while saving epoch 6 leaves its state truncated, so it resumes from epoch 4.
  train(ALGORITHMS['ars_sgd'](), tmp_path / 'full', epochs=8)
  train(ALGORITHMS['ars_sgd'](), tmp_path / 'resumed', epochs=6)
  (tmp_path / 'resumed' / 'algorithm' / '6.pkl').write_bytes(b'truncated')

  train(ALGORITHMS['ars_sgd'](), tmp_path / 'resumed', epochs=8)

  torch.testing.assert_close(
    read_params(tmp_path / 'resumed', 8),
    read_params(tmp_path / 'full', 8),
    rtol=0,
    atol=0,
  )


def test_resume_keeps_table_rows_up_to_the_restored_epoch(tmp_path: pathlib.Path) -> None:
  train(ALGORITHMS['ars_sgd'](), tmp_path, epochs=5)
  train(ALGORITHMS['ars_sgd'](), tmp_path, epochs=6)

  assert pd.read_csv(tmp_path / 'train.csv')['train/epochs'].tolist() == [1, 2, 3, 4, 5, 6]


def test_resume_from_copies_rows_up_to_the_restored_epoch_and_continues(
  tmp_path: pathlib.Path,
) -> None:
  train(ALGORITHMS['ars_sgd'](), tmp_path / 'full', epochs=6)
  train(ALGORITHMS['ars_sgd'](), tmp_path / 'source', epochs=5)  # Saves epochs 0, 2, 4 and 5.
  (tmp_path / 'source' / 'algorithm' / '5.pkl').unlink()

  train(ALGORITHMS['ars_sgd'](), tmp_path / 'resumed', epochs=6, resume_from=tmp_path / 'source')

  source = pd.read_csv(tmp_path / 'source' / 'train.csv')
  resumed = pd.read_csv(tmp_path / 'resumed' / 'train.csv')
  assert resumed['train/epochs'].tolist() == [1, 2, 3, 4, 5, 6]
  pd.testing.assert_frame_equal(resumed.iloc[:4], source.iloc[:4])
  torch.testing.assert_close(
    read_params(tmp_path / 'resumed', 6),
    read_params(tmp_path / 'full', 6),
    rtol=0,
    atol=0,
  )


def test_run_returns_the_restored_params_when_resuming_a_complete_run(
  tmp_path: pathlib.Path,
) -> None:
  params = train(ALGORITHMS['ars_sgd'](), tmp_path, epochs=4)

  resumed_params = train(ALGORITHMS['ars_sgd'](), tmp_path, epochs=4)

  torch.testing.assert_close(resumed_params, params, rtol=0, atol=0)
  assert resumed_params.abs().sum() > 0  # Not the initial (zero) params.


def test_test_and_save_epochs_follow_their_frequencies(tmp_path: pathlib.Path) -> None:
  # Tests and saves also run at the first and last epochs.
  train(ALGORITHMS['ars_sgd'](), tmp_path, epochs=5, save_freq=3, test_freq=2)

  assert pd.read_csv(tmp_path / 'test.csv')['test/epochs'].tolist() == [0, 2, 4, 5]
  assert sorted(int(path.stem) for path in (tmp_path / 'params').glob('*.pkl')) == [0, 3, 5]


def test_resume_true_requires_a_saved_state(tmp_path: pathlib.Path) -> None:
  with pytest.raises(FileNotFoundError):
    train(ALGORITHMS['ars_sgd'](), tmp_path, epochs=2, resume=True)
