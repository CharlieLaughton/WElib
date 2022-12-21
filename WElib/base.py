import numpy as np
from pathlib import Path
import json
import itertools

class Walker(object):
    '''
    A WE simulation walker
    '''
    iterator = itertools.count()


    def __init__(self, state, weight, state_id=None):
        self.state = state
        self.weight = weight
        self.state_id = state_id
        if self.state_id is None:
            self.state_id = next(Walker.iterator)
        self.pcs = []
        self.bin_id = None
        self.history = []
        self._initial_state = self.state
        self._initial_state_id = self.state_id
        self._initial_pcs = None
        self.data = {}
        
    def copy(self):
        new = Walker(self.state, self.weight, self.state_id)
        new.pcs = self.pcs
        new.bin_id = self.bin_id
        new.history = self.history.copy()
        new.data = self.data.copy()
        new._initial_pcs = self._initial_pcs
        return new

    def restart(self):
        self.state = self._initial_state
        self.state_id = self._initial_state_id
        self.pcs = self._initial_pcs

    def update(self, state):
        self.history.append(self.state_id)
        self.state_id = next(Walker.iterator)
        self.state = state
        self.pcs = None
        self.bin_id = None

    def merge(self, other_walker):
        w_tot = self.weight + other_walker.weight
        i = np.random.choice([0,1], p=[self.weight / w_tot, other_walker.weight / w_tot])
        if i == 0:
            new_walker = self.copy()
        else:
            new_walker = other_walker.copy()
        new_walker.weight = w_tot
        return new_walker

    def split(self):
        new_walker = self.copy()
        self.weight /= 2
        new_walker.weight = self.weight
        return self, new_walker

    def __repr__(self):
        return ('<WElib.Walker weight {}, progress coordinate {}, bin assignment {}>'.format(self.weight, self.pcs, self.bin_id))

class FunctionStepper(object):
    # Move the walkers according to a supplied function
    def __init__(self, function, *args):
        self.function = function
        self.args = args
        self.recorder = Recorder()

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        self.recorder.record(walkers)
        for w in walkers:
            state = self.function(w.state, *self.args)
            w.update(state)
        self.recorder.record(walkers)
        return walkers

class Recorder(object):
    def __init__(self):
        self.states = {}

    def record(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        for w in walkers:
            self.states[w.state_id] = w.state

    def replay(self, walker):
        statelist = []
        for i in walker.history:
            statelist.append(self.states[i])
        statelist.append(walker.state)
        return statelist

    def purge(self, walkers):
        all_states = []
        for w in walkers:
            all_states.append(w.state_id)
            all_states += w.history
        all_states = set(all_states)
        
        for state_id in self.states:
            if not state_id in all_states:
                del self.states[state_id]

class FunctionProgressCoordinator(object):
    def __init__(self, pcfunc, *args):
        self.pcfunc = pcfunc
        self.args = args

    def run(self, walkers):
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
    '''
    A WE simulation bin classifier
    
    Adds bin ID information to a list of walkers

    IDs are integers if there is just one dimension of binning,
    or tuples if more.
    
    Initialised with a list of the bin edges.
    '''
    def __init__(self, edges):
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
        
    
    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        pcs = np.atleast_2d([w.pcs for w in walkers])
        if None in pcs:
            raise TypeError('Error: missing progress coordinates...')
        bin_ids = []
        for dim in range(self.ndim):
            bin_ids.append(np.digitize(pcs[:, dim], self.edges[dim]))
        if self.ndim > 1:
            bin_ids = [z for z in zip(*bin_ids)]
        else:
            bin_ids = bin_ids[0]
        for i, bin_id in enumerate(bin_ids):
            walkers[i].bin_id = bin_id
            if not bin_id in self.bin_weights:
                self.bin_weights[bin_id] = 0.0
            self.bin_weights[bin_id] += walkers[i].weight
        sorted_dict = {}
        for key in sorted(self.bin_weights):
            sorted_dict[key] = self.bin_weights[key]
        self.bin_weights = sorted_dict
        return walkers

    def reset(self):
        for k in self.bin_weights:
            self.bin_weights[k] = 0.0
class Recycler(object):
    '''
    A WE simulation recycler
    
    Moves walkers that have reached the target pc back to the start
    Reports the recycled flux as well.
    
    Initialised with the
    value of the target pc. If retrograde == False, then recycling happens if
    target_pc is exceeded, else reycling happens if
    the pc falls below target_pc.
    '''
    def __init__(self, target_pcs, retrograde=False):
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
        self.recycled_walkers = []
        self.flux = 0.0
        if not isinstance(walkers, list):
            walkers = [walkers]
        for i in range(len(walkers)):
            if walkers[i].pcs == []:
                raise TypeError('Error - missing progress coordinate')
            recycle = True
            for idim in range(self.n_dim):
                if self.target_pcs[idim] is not None:
                    if not self.retrograde[idim]:
                        recycle = recycle and walkers[i].pcs[idim] > self.target_pcs[idim]
                    else:
                        recycle = recycle and walkers[i].pcs[idim] < self.target_pcs[idim]
            if recycle:
                walkers[i].restart()
                weight = walkers[i].weight
                self.flux += weight 
        self.flux_history.append(self.flux)
        return walkers

class Bin(object):
    '''
    A WE simulation bin object - only used internally
    '''
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
            ids = np.random.choice(range(len(self.walkers)), 
                    target_size, p=probs/old_weight)
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
    '''
    A WE simulation splitter and merger
    
    Splits or merges the walkers in each bin to get to the required number of replicas
    
    Initialised with the desired number of replicas in each bin.
    '''
    def __init__(self, target_size):
        self.target_size = target_size
    
    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
            
        bins = {}
        for w in walkers:
            if not w.bin in bins:
                bins[w.bin] = Bin(w.bin)
            bins[w.bin].add(w)
        
        for bin in bins:
            bins[bin].split_merge(self.target_size)
        
        new_walkers = []
        for bin in bins:
            new_walkers += bins[bin].walkers
            
        return new_walkers

