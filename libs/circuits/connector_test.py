import pytest
import torch

import circuits as cc


def make_connector(**kwargs) -> tuple[cc.Connector, cc.Signal]:
  """A connector of a fresh module, connected to a signal with `kwargs`."""
  source = cc.Signal()
  connector = cc.Connector(torch.nn.Module())
  connector.connect(source, **kwargs)
  return connector, source


# ==================================================================================================
# Connector


@pytest.mark.parametrize('delay', [1, 3])
def test_integrate_delivers_the_weighted_source_value_after_the_delay(delay):
  connector, source = make_connector(weight=2., delay=delay)

  delivered = []
  for t in range(6):
    source.value[:] = t + 1  # Values 1, 2, 3, ...
    delivered.append(connector.integrate().item())  # Simulator order: integrate, then populate.
    connector.populate()

  assert delivered == [0.] * delay + [2. * (t + 1) for t in range(6 - delay)]


def test_activation_applies_to_the_weighted_value():
  connector, source = make_connector(weight=-1., activation=torch.relu)
  source.value[:] = 1.

  connector.populate()

  assert connector.integrate().item() == 0.


def test_integrate_sums_all_connections():
  connector, first = make_connector(weight=1.)
  second = cc.Signal(3.)
  connector.connect(second, weight=-0.5)
  first.value[:] = 2.

  connector.populate()

  assert connector.integrate().item() == 0.5
  assert len(connector) == 2


def test_reset_clears_undelivered_values_only_where_reset_is_true():
  connector, source = make_connector(delay=2)
  source.value[:] = 1.
  connector.populate()

  connector.reset(torch.tensor([False]))
  kept = connector.connections['0'].delay.timesteps.clone()
  connector.reset(torch.tensor([True]))

  assert kept.sum() == 1.
  assert connector.connections['0'].delay.timesteps.sum() == 0.


@pytest.mark.parametrize('kwargs', [{'noise': 1.5}, {'delay': 0}], ids=['noise', 'delay'])
def test_connect_rejects_invalid_arguments(kwargs):
  with pytest.raises(ValueError):
    make_connector(**kwargs)
