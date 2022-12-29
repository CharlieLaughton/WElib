from WElib import (
    Walker,
    FunctionProgressCoordinator,
    FunctionStepper,
    Recycler,
    SplitMerger,
    StaticBinner,
)
import pytest


def test_walker_init():
    state = [10.0]
    weight = 1.0
    w = Walker(state, weight)
    assert w.weight == weight
    assert w.state == state
    assert w.state_id is not None
    assert w.pcs == []
    assert w._history == []
    assert w.bin_id is None
    assert w.data == {}


def test_walker_split():
    state = [10.0]
    weight = 1.0
    w = Walker(state, weight)
    w1, w2 = w.split()
    assert isinstance(w1, Walker)
    assert isinstance(w2, Walker)
    assert w1 != w2
    assert w1.state == w2.state
    assert w1.weight == w2.weight
    assert w1.weight == weight / 2
    assert w1.state_id == w2.state_id
    assert w1.data == w2.data
    assert w1._history == w2._history


def test_walker_merge():
    w1 = Walker([10.0], 0.9)
    w2 = Walker([5.0], 0.0)
    w = w1.merge(w2)
    assert w.state == w1.state
    w = w2.merge(w1)
    assert w.state == w1.state
    w3 = Walker([7.0], 0.9)
    n1 = 0
    for i in range(100):
        w = w1.merge(w3)
        if w.state == w1.state:
            n1 += 1
    assert 30 < n1 < 70


def test_walker_copy():
    w1 = Walker([10.0], 0.5)
    w2 = w1.copy()
    assert w2 != w1
    assert w1.state == w2.state


def test_fpc_1pc():
    pc = FunctionProgressCoordinator(lambda x: x[0])
    w = Walker([10.0], 0.5)
    assert w.pcs == []
    w = pc.run(w)
    assert w[0].pcs == [10.0]


def test_fpc_2pc():
    pc = FunctionProgressCoordinator(lambda x: (x[0], x[0] + 1))
    w = Walker([10], 0.5)
    assert w.pcs == []
    w = pc.run(w)
    assert w[0].pcs == [10, 11]


def test_static_binner_1d():
    binner = StaticBinner([1.0, 2.0])
    w1 = Walker([0.9], 0.1)
    w2 = Walker([1.0], 0.2)
    w3 = Walker([2.2], 0.3)
    walkers = [w1, w2, w3]
    pc = FunctionProgressCoordinator(lambda x: x[0])
    walkers = pc.run(walkers)
    assert binner.bin_weights == {}
    walkers = binner.run(walkers)
    assert walkers[0].bin_id == 0
    assert walkers[1].bin_id == 1
    assert walkers[2].bin_id == 2
    assert binner.bin_weights[0] == 0.1
    assert binner.bin_weights[1] == 0.2
    assert binner.bin_weights[2] == 0.3
    binner.reset()
    assert binner.bin_weights[0] == 0.0


def test_static_binner_2d():
    binner = StaticBinner([[1.0, 2.0], [2.0, 3.0]])
    w1 = Walker([0.9], 0.1)
    w2 = Walker([1.0], 0.2)
    w3 = Walker([2.2], 0.3)
    walkers = [w1, w2, w3]
    pc = FunctionProgressCoordinator(lambda x: (x[0], x[0] + 1.0))
    walkers = pc.run(walkers)
    walkers = binner.run(walkers)
    assert walkers[0].bin_id == (0, 0)
    assert walkers[1].bin_id == (1, 1)
    assert walkers[2].bin_id == (2, 2)


def test_static_binner_reverse_1d():
    binner = StaticBinner([2.0, 1.0])
    w1 = Walker([0.9], 0.1)
    w2 = Walker([1.0], 0.2)
    w3 = Walker([2.2], 0.3)
    walkers = [w1, w2, w3]
    pc = FunctionProgressCoordinator(lambda x: x[0])
    walkers = pc.run(walkers)
    walkers = binner.run(walkers)
    assert walkers[0].bin_id == 2
    assert walkers[1].bin_id == 1
    assert walkers[2].bin_id == 0


def test_static_binner_catches_bad_edges():
    binner = StaticBinner([1.0, 3.0, 2.0])
    w1 = Walker([0.9], 0.1)
    w2 = Walker([1.0], 0.2)
    w3 = Walker([2.2], 0.3)
    walkers = [w1, w2, w3]
    pc = FunctionProgressCoordinator(lambda x: x[0])
    walkers = pc.run(walkers)
    with pytest.raises(Exception):
        walkers = binner.run(walkers)


def test_function_stepper():
    stepper = FunctionStepper(lambda x: [x[0] + 1])
    w = Walker([1.0], 0.3)
    w = stepper.run(w)
    assert w[0].state == [2.0]


def test_recycler_forward_1d():
    recycler = Recycler(2.0)
    w1 = Walker([1.0], 0.1)
    w2 = Walker([1.0], 0.2)
    w3 = Walker([1.0], 0.3)
    walkers = [w1, w2, w3]
    pc = FunctionProgressCoordinator(lambda x: x[0])
    walkers = pc.run(walkers)
    w1.update([0.9])
    w3.update([2.2])
    walkers = pc.run(walkers)
    walkers = recycler.run(walkers)
    assert recycler.flux == 0.3
    assert walkers[2] != w3
    assert walkers[2].state == [1.0]


def test_recycler_retrograde_1d():
    recycler = Recycler(1.0, retrograde=True)
    w1 = Walker([2.0], 0.1)
    w2 = Walker([2.0], 0.2)
    w3 = Walker([2.0], 0.3)
    walkers = [w1, w2, w3]
    pc = FunctionProgressCoordinator(lambda x: x[0])
    walkers = pc.run(walkers)
    w1.update([0.9])
    w3.update([2.2])
    walkers = pc.run(walkers)
    walkers = recycler.run(walkers)
    assert recycler.flux == 0.1
    assert walkers[0] != w1
    assert walkers[0].state == [2.0]


def test_recycler_forward_2d():
    recycler = Recycler([2.0, None])
    w1 = Walker([1.0], 0.1)
    w2 = Walker([1.0], 0.2)
    w3 = Walker([1.0], 0.3)
    walkers = [w1, w2, w3]
    pc = FunctionProgressCoordinator(lambda x: (x[0], x[0] + 1.0))
    walkers = pc.run(walkers)
    w2.update([2.2])
    w3.update([3.3])
    walkers = pc.run(walkers)
    walkers = recycler.run(walkers)
    assert recycler.flux == 0.5
    assert walkers[1].state == [1.0]
    assert walkers[2].state == [1.0]
    walkers[1].update([2.2])
    walkers[2].update([3.3])
    walkers = pc.run(walkers)
    recycler2 = Recycler([2.0, 4.0])
    walkers = recycler2.run(walkers)
    assert walkers[1].state == [2.2]
    assert walkers[2].state == [1.0]
    assert recycler2.flux == 0.3


def test_splitmerger():
    w1 = Walker([1.0], 0.1)
    w1.bin_id = 0
    sm = SplitMerger(4)
    walkers = [w1]
    walkers = sm.run(walkers)
    assert len(walkers) == 4
    assert walkers[0].weight == 0.025
