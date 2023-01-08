"""
Crossflow-enabled classes for WElib

Provides crossflow-compatible building blocks to code weighted ensemble simulation
workflows.

Classes:

    CrossflowFunctionStepper
    CrossflowFunctionProgressCoordinator
"""
from .base import Recorder


class CrossflowFunctionStepper(object):
    """
    A class for functions that move (update the states of) lists of walkers

    Attributes
    ----------
    recorder : Recorder
        keeps a record of all states visited by all walkers

    Methods
    -------
    run(walkers):
        update the states of a list of walkers
    """

    def __init__(self, client, function, *args):
        """
        Create a stepper

        Parameters
        ----------
        client : Crossflow Client
            a crossflow client that connects to a dask distributed cluster
        function : function
            a function with the call signature: new_state = function(old_state, *args)
        *args : list, optional
            any extra arguments required by the function
        """

        self.client = client
        self._function = function
        self._args = args
        self.recorder = Recorder()

    def run(self, walkers):
        """
        Run the stepper on a list of walkers

        Parameters
        ----------
        walkers : list
            a list of walkers whose states will be updated

        Returns
        -------
        list
            a list of walkers with updated states
        """

        if not isinstance(walkers, list):
            walkers = [walkers]
        self.recorder.record(walkers)
        old_states = [w.state for w in walkers]
        new_states = self.client.map(self._function, old_states, *self._args)
        for n, w in zip(new_states, walkers):
            w.update(n.result())
        self.recorder.record(walkers)
        return walkers


class CrossflowFunctionProgressCoordinator(object):
    """
    A class for crossflow-compatible functions that update the progress coordinates
    of lists of walkers

    Methods
    -------
    run(walkers)
        Update the progress coordinates for a list of walkers
    """

    def __init__(self, client, pcfunc, *args):
        """
        Create a progress coordinator

        Parameters
        ----------
        client : Crossflow Client
            a client attached to a dask.distributed cluster
        pcfunc : function
            a function with the call signature:
                progress_coordinates = function(state, *args)
        *args : list, optional
            any extra arguments required by the function
        """

        self.client = client
        self.pcfunc = pcfunc
        self.args = args

    def run(self, walkers):
        """
        Process a list of walkers, updating their progress coordinates

        Parameters
        ----------
        walkers : list of Walkers
            walkers whose progress coordinates will be updated

        Returns
        -------
        list
            a list of walkers with updated progress coordinates
        """

        if not isinstance(walkers, list):
            walkers = [walkers]

        states = [w.state for w in walkers]
        pcs = self.client.map(self.pcfunc, states, *self.args)
        for p, w in zip(pcs, walkers):
            pr = p.result()
            if isinstance(pr, (list, tuple)):
                w.pcs = list(pr)
            else:
                w.pcs = [pr]
            if w._initial_pcs is None:
                w._initial_pcs = w.pcs
        return walkers
