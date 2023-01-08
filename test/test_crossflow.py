from WElib import (
    Walker,
)
from WElib.crossflow import (
    CrossflowFunctionStepper,
    CrossflowFunctionProgressCoordinator,
)

import pytest
from crossflow.clients import Client
from crossflow.tasks import FunctionTask
from distributed import LocalCluster
import numpy as np


@pytest.fixture(scope="module")
def crossflow_client():
    cluster = LocalCluster(n_workers=1)
    client = Client(cluster)
    return client


def test_xfpc_1pc(crossflow_client):
    def _pcfunc(state):
        return state[0]

    pcfunc = FunctionTask(_pcfunc)
    pcfunc.set_inputs(["state"])
    pcfunc.set_outputs(["pcs"])
    pc = CrossflowFunctionProgressCoordinator(crossflow_client, pcfunc)
    w = Walker([10.0], 0.5)
    assert w.pcs == []
    w = pc.run(w)
    assert w[0].pcs == [10.0]


def test_fpc_2pc(crossflow_client):
    def _pcfunc(state):
        return state[0], state[0] + 1

    pcfunc = FunctionTask(_pcfunc)
    pcfunc.set_inputs(["state"])
    pcfunc.set_outputs(["pcs"])
    pc = CrossflowFunctionProgressCoordinator(crossflow_client, pcfunc)
    w = Walker([10], 0.5)
    assert w.pcs == []
    w = pc.run(w)
    assert w[0].pcs == [10, 11]


def test_crossflow_function_stepper(crossflow_client):
    def _stepfunc(state):
        return state + 1.0

    stepfunc = FunctionTask(_stepfunc)
    stepfunc.set_inputs(["state"])
    stepfunc.set_outputs(["newstate"])

    stepper = CrossflowFunctionStepper(crossflow_client, stepfunc)
    w = Walker(np.array([1.0, 2.0]), 0.3)
    w = stepper.run(w)
    assert (w[0].state == np.array([2.0, 3.0])).all()
