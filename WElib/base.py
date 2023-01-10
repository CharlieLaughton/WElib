"""
Base classes for WElib

Provides building blocks to code weighted ensemble simulation workflows.

Classes:

    Walker
    FunctionStepper
    FunctionProgressCoordinator
    StaticBinner
    Recycler
    SplitMerger
    Bin (generally only for internal use)
    Recorder
"""
import numpy as np

from pathlib import Path

import json
import itertools


class Walker(object):
    """
    A WE simulation walker

    A Walker has a state (the thing that changes when it walks - e.g. its full set of
    coordinates), and a weight. If it's been set, then it will also have one or more
    progress coordinates (a function of its state), and may also have a bin ID
    (a function of its progress coordinate/s).

    Walkers keep a history of the states they have visited (in the form of the state
    IDs) and also remember details of how they were initialised (state, state ID and
    progress coordinate/s) so they can be "reset". They also feature a general-purpose
    "data" attribute - a dictionary that can be used to hold any relevant metadata.

    Attributes
    ----------
    state : any
        the current state of the walker
    weight : float
        the current weight of the walker
    state_id : integer
        the ID of the current state
    pcs : list
        the current values of the progress coordinate(s), may be empty if not yet
        defined
    bin_id : int or None
        the ID of the bin the walker is in, or None if not yet defined
    data : dict
        general purpose store for optional metadata

    Methods
    -------
    copy():
        return a copy of the walker
    restart()
        reset the walker to its initial state
    update(state):
        update the state of the walker
    merge(other_walker):
        merge the walker with another
    split():
        split the walker into two
    """

    iterator = itertools.count()

    def __init__(self, state, weight, state_id=None):
        """
        Create a new walker

        Parameters
        ----------

            state : any
                initial state of the walker
            weight : (float < 1.0):
                weight of the walker
            state_id : int, optional
                the ID of the initial state. A new unique ID will be
                assigned if none is provided
        """

        self.state = state
        self.weight = weight
        self.state_id = state_id
        if self.state_id is None:
            self.state_id = next(Walker.iterator)
        self.pcs = []
        self.bin_id = None
        self._history = []
        self._initial_state = self.state
        self._initial_state_id = self.state_id
        self._initial_pcs = None
        self.data = {}

    def copy(self):
        """
        Return a copy of the walker

        Returns
        -------
        None
        """

        new = Walker(self.state, self.weight, self.state_id)
        new.pcs = self.pcs
        new.bin_id = self.bin_id
        new._history = self._history.copy()
        new.data = self.data.copy()
        new._initial_state = self._initial_state
        new._initial_state_id = self._initial_state_id
        new._initial_pcs = self._initial_pcs
        return new

    def restart(self):
        """
        Restart the walker

        The initially-assigned state is restored, and associated progress coordinates.
        The history is cleared.

        Returns
        -------
        None
        """

        self.state = self._initial_state
        self.state_id = self._initial_state_id
        self.pcs = self._initial_pcs
        self._history = []

    def update(self, state):
        """
        Update the state of the walker

        The existing progress coordinate and bin data are cleared (set to None)

        Parameters
        ----------
        state : state
            The new state of the walker

        Returns
        -------
        None
        """

        self._history.append(self.state_id)
        self.state_id = next(Walker.iterator)
        self.state = state
        self.pcs = None
        self.bin_id = None

    def merge(self, other_walker):
        """
        Merge the walker with another

        The returned walker is a copy of one of the two, chosen at random on the basis
        of their relative weights. The returned walker has the combined weight of both.

        Parameters
        ----------
        other_walker : Walker
            The walker to be merged with this one

        Returns:
        Walker
            the merged walker
        """

        w_tot = self.weight + other_walker.weight
        i = np.random.choice(
            [0, 1], p=[self.weight / w_tot, other_walker.weight / w_tot]
        )
        if i == 0:
            new_walker = self.copy()
        else:
            new_walker = other_walker.copy()
        new_walker.weight = w_tot
        return new_walker

    def split(self):
        """
        Split the walker in two, each with half the weight

        Returns
        -------
        (Walker, Walker)
            the two walkers
        """

        new_walker = self.copy()
        self.weight /= 2
        new_walker.weight = self.weight
        return self, new_walker

    def __repr__(self):
        return "<WElib.Walker weight = {}, pcs = {}, bin = {}>".format(
            self.weight, self.pcs, self.bin_id
        )


class FunctionStepper(object):
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

    def __init__(self, function, *args):
        """
        Create a stepper

        Parameters
        ----------
        function : function
            a function with the call signature: new_state = function(old_state, *args)
        *args : list, optional
            any extra arguments required by the function
        """

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
        for w in walkers:
            state = self._function(w.state, *self._args)
            w.update(state)
        self.recorder.record(walkers)
        return walkers


class Recorder(object):
    """
    A class for functions that keep a record of states visted by walkers

    Methods
    -------
    record(walkers)
        Add the states of each walker in the list to the archive
    replay(walker)
        Return a list of the states visited by a walker, to date
    purge(walkers)
        Remove unneeded states from the archive
    """

    def __init__(self):
        """
        Initialise a new recorder

        """

        self._states = {}

    def record(self, walkers):
        """
        Add the state of each walker in the list to the archive

        Parameters
        ----------
        walkers : list of Walkers
            walkers whose states are to be recorded
        Returns
        -------
        None
        """

        if not isinstance(walkers, list):
            walkers = [walkers]
        for w in walkers:
            self._states[w.state_id] = w.state

    def replay(self, walker):
        """
        Generate a list of all the states visited by a walker

        Parameters
        ----------
        walker : Walker
            the walker whose history is to be generated

        Returns
        -------
        list : states
            the states visited by the walker, in chronological order
        """

        statelist = []
        for i in walker._history:
            statelist.append(self._states[i])
        statelist.append(walker.state)
        return statelist

    def purge(self, walkers):
        """
        Remove unneeded states from the archive, to save memory

        Parameters
        ----------
        walkers : list of Walkers
            Only states that have been visited at some time by any of these walkers
            will be retained

        Returns
        -------
        None
        """

        all_states = []
        for w in walkers:
            all_states.append(w.state_id)
            all_states += w._history
        all_states = set(all_states)

        for state_id in self._states:
            if state_id not in all_states:
                del self._states[state_id]


class FunctionProgressCoordinator(object):
    """
    A class for functions that update the progress coordinates of lists of walkers

    Methods
    -------
    run(walkers)
        Update the progress coordinates for a list of walkers
    """

    def __init__(self, pcfunc, *args):
        """
        Create a progress coordinator

        Parameters
        ----------
        pcfunc : function
            a function with the call signature:
                progress_coordinates = function(state, *args)
        *args : list, optional
            any extra arguments required by the function
        """

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
        for w in walkers:
            pcs = self.pcfunc(w.state, *self.args)
            if isinstance(pcs, (list, tuple)):
                w.pcs = list(self.pcfunc(w.state, *self.args))
            else:
                w.pcs = [pcs]
            if w._initial_pcs is None:
                w._initial_pcs = w.pcs
        return walkers


class StaticBinner(object):
    """
    A WE simulation bin classifier

    Adds bin ID information to a list of walkers

    IDs are integers if there is just one dimension of binning,
    or tuples if more.

    Initialised with a list of the bin edges.

    Attributes
    ----------
    bin_weights : dict
        keyed by bin ID, the _current_ weight in the bin
    mean_bin_weights : dict
        keyed by bin ID, the _mean_ weight in each bin, since start or reset (see below)

    Methods:
    run(walkers)
        assign bin ID to each walker in a list of walkers, update the bin_weight data
    reset()
        zero the memory of mean bin weights
    """

    def __init__(self, edges):
        """
        Initialise a binner

        Parameters
        ----------
        edges : list (or list of lists) of floats
            the bin edges, in order of ascending or descending value
        """

        self.edges = edges
        e = edges[0]
        n_dim = 1
        while isinstance(e, list):
            n_dim += 1
            e = e[0]
        self.ndim = n_dim
        if n_dim == 1:
            self.edges = [self.edges]
        self.bin_weights = {}
        self._cumulative_bin_weights = {}
        self._ncycles = 0

    def run(self, walkers):
        """
        Assign bin IDs to each walker in a list of walkers

        Parameters
        ----------
        walkers : list of Walkers
            walkers whose bin IDs will be set

        Returns
        list : list of Walkers
            walkers with updated bin IDs
        """

        self._ncycles += 1
        if not isinstance(walkers, list):
            walkers = [walkers]
        pcs = np.atleast_2d([w.pcs for w in walkers])
        if None in pcs:
            raise TypeError("Error: missing progress coordinates...")
        bin_ids = []
        for dim in range(self.ndim):
            bin_ids.append(np.digitize(pcs[:, dim], self.edges[dim]))
        if self.ndim > 1:
            bin_ids = [z for z in zip(*bin_ids)]
        else:
            bin_ids = bin_ids[0]

        for k in self.bin_weights:
            self.bin_weights[k] = 0.0
        for i, bin_id in enumerate(bin_ids):
            walkers[i].bin_id = bin_id
            if bin_id not in self.bin_weights:
                self.bin_weights[bin_id] = 0.0
            if bin_id not in self._cumulative_bin_weights:
                self._cumulative_bin_weights[bin_id] = 0.0
            self.bin_weights[bin_id] += walkers[i].weight
            self._cumulative_bin_weights[bin_id] += walkers[i].weight
        sorted_dict = {}
        for key in sorted(self.bin_weights):
            sorted_dict[key] = self.bin_weights[key]
        self.bin_weights = sorted_dict
        sorted_dict = {}
        for key in sorted(self._cumulative_bin_weights):
            sorted_dict[key] = self._cumulative_bin_weights[key]
        self._cumulative_bin_weights = sorted_dict
        return walkers

    @property
    def mean_bin_weights(self):
        return {k: w / self._ncycles for k, w in self._cumulative_bin_weights.items()}

    def reset(self):
        """
        Reset the cumulative bin weights data to zero

        Returns
        -------
        None
        """

        self._ncycles = 0
        for k in self._cumulative_bin_weights:
            self._cumulative_bin_weights[k] = 0.0


class Recycler(object):
    """
    A WE simulation recycler

    Moves walkers that have reached the target pc(s) back to the start
    Reports the recycled flux as well.

    Attributes
    -----------
    recycled_walkers : list of Walkers
        walkers (if any) that were recycled last time the recycler was run
    flux : float
        the recycled flux last time the recycler was run
    flux_history : list of floats
        the recycled flux each time the recycler has been run
    """

    def __init__(self, target_pcs, retrograde=False):
        """
        Initialise a recycler

        Parameters
        ----------
        target_pcs : float, or list or tuple of floats
            The target progress coordinate for each dimension
        retrograde : bool or list or tuple of bools
            If False, recycling happens when the pc in the associated dimension is
            greater than the associated target_pc. If True, recycling happens if the pc
            falls below the target_pc
        """

        self.target_pcs = target_pcs
        if isinstance(target_pcs, (list, tuple)):
            self.n_dim = len(target_pcs)
        else:
            self.n_dim = 1
            self.target_pcs = [self.target_pcs]
        if self.n_dim == 1:
            self.retrograde = [retrograde]
        else:
            if not isinstance(retrograde, (list, tuple)):
                self.retrograde = [retrograde for i in range(self.n_dim)]
        self.recycled_walkers = []
        self.flux_history = []
        self.flux = None

    def run(self, walkers):
        """
        Run the recycler

        Parameters
        ----------
        walkers : list of walkers
            walkers that will be checked for possible recycling

        Returns
        -------
        list
            list of walkers, where recycled ones in the input are replaced by fresh
            ones with the same weight, but at the initial state
        """

        self.recycled_walkers = []
        self.flux = 0.0
        if not isinstance(walkers, list):
            walkers = [walkers]
        for i in range(len(walkers)):
            if walkers[i].pcs == []:
                raise TypeError("Error - missing progress coordinate")
            recycle = True
            for idim in range(self.n_dim):
                if self.target_pcs[idim] is not None:
                    if not self.retrograde[idim]:
                        recycle = (
                            recycle and walkers[i].pcs[idim] > self.target_pcs[idim]
                        )
                    else:
                        recycle = (
                            recycle and walkers[i].pcs[idim] < self.target_pcs[idim]
                        )
            if recycle:
                replacement = Walker(
                    walkers[i]._initial_state,
                    walkers[i].weight,
                    walkers[i]._initial_state_id,
                )
                self.recycled_walkers.append(walkers[i])
                walkers[i] = replacement
                weight = walkers[i].weight
                self.flux += weight
        self.flux_history.append(self.flux)
        return walkers


class Bin(object):
    """
    A WE simulation bin object - only used internally
    """

    def __init__(self, index):
        self.index = index
        self.walkers = []

    def add(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        self.walkers += walkers

    def split_merge(self, target_size):
        if len(self.walkers) == target_size:
            ids = range(target_size)
        else:
            probs = np.array([w.weight for w in self.walkers])
            old_weight = probs.sum()
            ids = np.random.choice(
                range(len(self.walkers)), target_size, p=probs / old_weight
            )
        new_walkers = []
        for i in ids:
            new_walker = self.walkers[i].copy()
            new_walkers.append(new_walker)
        if len(self.walkers) != target_size:
            new_weight = np.array([w.weight for w in new_walkers]).sum()
            fac = old_weight / new_weight
            for i in range(len(new_walkers)):
                new_walkers[i].weight *= fac
        self.walkers = list(new_walkers)


class SplitMerger(object):
    """
    A WE simulation splitter and merger

    Splits or merges the walkers in each bin to get to the required number of replicas

    Initialised with the desired number of replicas in each bin.

    Methods
    -------
    run(walkers)
        run the split-merger on a list of walkers
    """

    def __init__(self, target_size):
        """
        Initialise a split-merger

        Parameters
        ----------
        target_size : int
            the desired number of walkers in each occupied bin
        """

        self.target_size = target_size

    def run(self, walkers):
        """
        Run the split-merger

        Parameters
        ----------
        walkers : list of Walkers
            walkers that will be split and merged to give the target number per bin

        Returns
        -------
        list : list of Walkers
            A new list of walkers, potentially of different size to the input
        """

        if not isinstance(walkers, list):
            walkers = [walkers]

        bins = {}
        for w in walkers:
            if w.bin_id not in bins:
                bins[w.bin_id] = Bin(w.bin_id)
            bins[w.bin_id].add(w)

        for bin in bins:
            bins[bin].split_merge(self.target_size)

        new_walkers = []
        for bin in bins:
            new_walkers += bins[bin].walkers

        return new_walkers


class Checkpointer(object):
    """
    A simple checkpointing class

    Saves coordinates and metadata for a list of walkers in a specified
    directory

    Methods
    -------
    save : save a list of walkers to files in a checkpoint directory
    load : load a list of walkers from files in a checkpoint directory
    """

    def __init__(self, dirname, state_serializer, ext=".dat", mode="r"):
        self.dirname = Path(dirname)
        if not (
            hasattr(state_serializer, "serialize")
            and hasattr(state_serializer, "deserialize")
        ):
            raise AttributeError(
                "Error: state_serializer must have serialize and deserialize methods"
            )
        self.serializer = state_serializer
        self.ext = ext
        self.mode = mode

        if "w" not in self.mode:
            if not self.dirname.exists():
                raise OSError("Error - checkpoint directory not found")
        self.dirname.mkdir(parents=True, exist_ok=True)

    def save(self, walkers):
        if "w" not in self.mode:
            raise OSError("Error: checkpoint directory is read-only")
        metadata = {}
        for i, w in enumerate(walkers):
            name = "walker_{:05d}{}".format(i, self.ext)
            metadata[name] = {}
            metadata[name]["weight"] = w.weight
            metadata[name]["state_id"] = w.state_id
            metadata[name]["history"] = w._history
            metadata[name]["initial_state"] = self.serializer.serialize(
                w._initial_state
            )
            metadata[name]["initial_state_id"] = w._initial_state_id
            metadata[name]["initial_pcs"] = w._initial_pcs
            state_string = self.serializer.serialize(w.state)
            statefile = self.dirname / name
            statefile.write_text(state_string)

        metadatafile = self.dirname / "_metadata_"
        with metadatafile.open("w") as f:
            json.dump(metadata, f)

    def load(self):
        metadatafile = self.dirname / "_metadata_"
        if not metadatafile.exists():
            raise OSError("Error: no metadata file found")
        with metadatafile.open() as f:
            metadata = json.load(f)
        statefiles = list(self.dirname.glob("walker_*"))
        statefiles.sort()
        walkers = []
        for i, name in enumerate(metadata):
            statefile = Path(statefiles[i])
            state = self.serializer.deserialize(statefile.read_text())
            wt = metadata[name]["weight"]
            state_id = metadata[name]["state_id"]
            w = Walker(state, wt, state_id=state_id)
            w._history = metadata[name]["history"]
            w._initial_state = self.serializer.deserialize(
                metadata[name]["initial_state"]
            )
            w._initial_state_id = metadata[name]["initial_state_id"]
            w._initial_pcs = metadata[name]["initial_pcs"]
            walkers.append(w)
        return walkers
