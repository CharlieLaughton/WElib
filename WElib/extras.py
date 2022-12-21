from crossflow.kernels import SubprocessKernel
from crossflow.filehandling import FileHandler
import mdtraj as mdt
import numpy as np
from pathlib import Path
import json
import itertools

class CrossflowGACheckpointer(object):
    '''
    A simple checkpointing class for Crossflow filehandles, Gromacs/Amber sims

    Saves coordinates and metadata for a list of walkers in a specified
    directory
    '''
    def __init__(self, dirname, mode='r'):
        self.dirname = Path(dirname)
        self.mode = mode
        if not 'w' in self.mode:
            if not self.dirname.exists():
                raise OSError('Error - checkpoint directory not found')
        self.dirname.mkdir(parents=True, exist_ok=True)

    def save(self, walkers):
        if not 'w' in self.mode:
            raise OSError('Error: checkpoint directory is read-only')
        metadata = {}
        for i,w in enumerate(walkers):
            name = 'walker_{:05d}'.format(i)
            metadata[name] = {}
            metadata[name]['weight'] = w.weight
            metadata[name]['state_id'] = w.state_id
            metadata[name]['history'] = w.history
            coordinates = w.state
            if hasattr(coordinates, 'result'):
                coordinates = w.state.result()
            crdfile = self.dirname / name
            if hasattr(coordinates, 'save'):
                if hasattr(coordinates, 'uid'):
                    ext = Path(coordinates.uid).suffix
                    crdfile = self.dirname / (name + ext)
                coordinates.save(crdfile)
            else:
                coordinates = Path(coordinates)
                crdfile = self.dirname / (name + coordinates.suffix)
                crdfile.write_bytes(coordinates.read_bytes())

        metadatafile = self.dirname / '_metadata_'
        with metadatafile.open('w') as f:
            json.dump(metadata, f)

    def load(self):
        metadatafile = self.dirname / '_metadata_'
        if not metadatafile.exists():
            raise OSError('Error: no metadata file found')
        with metadatafile.open() as f:
            metadata = json.load(f)
        crdfiles = list(self.dirname.glob('walker_*'))
        crdfiles.sort()
        walkers = []
        for i, name in enumerate(metadata):
            crds = crdfiles[i]
            wt = metadata[name]['weight']
            state_id = metadata[name]['state_id']
            w = Walker(crds, wt, state_id=state_id)
            w.history = metadata[name]['history']
            walkers.append(w)
        return walkers
        

class CrossflowFunctionStepper(object):
    # Move the walkers according to a supplied function
    def __init__(self, client, function, *args):
        self.client = client
        self.function = function
        self.args = args
        self.recorder = Recorder()
    def run(self, walkers):
        self.recorder.record(walkers)
        states = [w.state for w in walkers]
        newstates = self.client.map(self.function, states, *self.args)
        for w, s in zip(walkers, newstates):
            w.update(s.result())
        self.recorder.record(walkers)
        return walkers

class CrossflowPMEMDCudaStepper(object):
    '''
    A WE simulation stepper
    
    Moves each walker a step forward (by running a bit of MD. Uses pmemd
    via crossflow

    Initialised with a crossflow client, and Amber mdin and prmtop files.
    '''
    def __init__(self, client, mdin, prmtop):
        self.client = client
        self.mdin = mdin
        self.prmtop = prmtop
        self.pmemd = SubprocessKernel('pmemd.cuda -i mdin -c in.ncrst -p x.prmtop -r out.ncrst -o mdlog -AllowSmallBox')
        self.pmemd.set_inputs(['mdin', 'in.ncrst', 'x.prmtop'])
        self.pmemd.set_outputs(['out.ncrst', 'mdlog'])
        self.pmemd.set_constant('mdin', mdin)
        self.pmemd.set_constant('x.prmtop', prmtop)
        self.recorder = Recorder()
        
    def run(self, walkers):
        self.recorder.record(walkers)
        inpcrds = [w.state for w in walkers]
        restarts, logs = self.client.map(self.pmemd, inpcrds)
        state_ids = [w.state_id for w in walkers]
        next_state_id = max(state_ids) + 1
        new_walkers = []
        for i, r in enumerate(restarts):
            if r.status == 'error':
                new_walkers.append(None)
            else:
                state = r.result()
                w = walkers[i]
                w.update(state)
                new_walkers.append(w)
        for i, w in enumerate(new_walkers):
            w.data['log'] = logs[i].result()
        self.recorder.record(new_walkers)
        return new_walkers


class GAPCVProgressCoordinator(object):
    '''
    A WE simulation progress coordinate calculator

    Adds progress coordinate data to a list of walkers

    Initialised with a trajectory of the points that define the path,
    the indices of the atoms in the collective variable, and the PCV
    lambda parameter
    '''
    def __init__(self, pcv_traj, atom_indices, l):
        self.topology = pcv_traj.topology
        self.pcv_traj = pcv_traj.atom_slice(atom_indices)
        self.atom_indices = atom_indices
        self.l = l

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.topology)
        p = len(self.pcv_traj)
        s = []
        z = []
        x_l = walker_traj.atom_slice(self.atom_indices)
        for i, c in enumerate(x_l):
            rmsd = mdt.rmsd(self.pcv_traj, c)
            msd = rmsd * rmsd
            v = np.exp(msd * -self.l)
            vi = v * range(p)
            s = vi.sum() / ((p-1) * v.sum())
            z = -1/(self.l * np.log(v.sum()))
            walkers[i].pc = s
            walkers[i].data['z'] = z
        return walkers

class GASimpleDistanceProgressCoordinator(object):
    '''
    A WE simulation progress coordinate calculator
    
    Adds progress coordinate data to a list of walkers
    
    Initialised with an MDTraj Topology for the system and indices
    of the atoms to monitor the distance between.
    '''
    def __init__(self, topfile, atom_pair):
        self.topology = mdt.load_topology(topfile)
        self.atom_pair = atom_pair
        
    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.topology)
        pcs = mdt.compute_distances(walker_traj, [self.atom_pair])[:,0]
        for i, pc in enumerate(pcs):
            walkers[i].pc = pc
        return walkers
    

class GARMSDProgressCoordinator(object):
    '''
    A WE simulation progress coordinate generator

    Returns the RMSD of a set of atoms from a reference structure
    '''
    def __init__(self, ref, fit_sel):
        self.ref = ref
        self.fit_atoms = self.ref.topology.select(fit_sel)

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.ref.topology)
        pcs = mdt.rmsd(walker_traj, self.ref, atom_indices=self.fit_atoms)
        for i in range(len(walker_traj)):
            walkers[i].pc = pcs[i]
        return walkers

class GARMSD2ProgressCoordinator(object):
    '''
    A WE simulation progress coordinate generator

    Returns the fractional distance of a structure (1) between two
    reference points (2 & 3): f = r12/(r12+r13)
    '''
    def __init__(self, refstart, refend, fit_sel):
        self.refstart = refstart
        self.refend = refend
        self.fit_atoms = self.refstart.topology.select(fit_sel)

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        inps = [w.state for w in walkers]
        walker_traj = mdt.load(inps, top=self.refstart.topology)
        rms12 = mdt.rmsd(walker_traj, self.refstart, atom_indices=self.fit_atoms)
        rms13 = mdt.rmsd(walker_traj, self.refend, atom_indices=self.fit_atoms)
        for i in range(len(walker_traj)):
            pc = rms12[i]/(rms12[i] + rms13[i])
            walkers[i].pc = pc
        return walkers

    
class MinimalAdaptiveBinner(object):
    '''
    Implements the minimal adaptive binning strategy
    '''
    def __init__(self, n_bins, retrograde=False):
        self.n_bins = n_bins
        self.retrograde = retrograde

    def run(self, walkers):
        if not isinstance(walkers, list):
            walkers = [walkers]
        n_walkers = len(walkers)
        for i in range(n_walkers):
            if walkers[i].pc is None:
                raise TypeError('Error - missing progress coordinate')
        walkers.sort(key=lambda w: w.pc)
        bin_width = (walkers[-1].pc - walkers[0].pc) / self.n_bins
        pc_min = walkers[0].pc
        if self.retrograde:
            walkers.reverse()
        w = np.array([w.weight for w in walkers])
        zmax = -np.log(w.sum())
        bottleneck = 0
        for i in range(n_walkers):
            walkers[i].bin = int((walkers[i].pc - pc_min) / bin_width)
            if i < n_walkers - 1:
                z = np.log(w[i]) - np.log(w[i+1:].sum())
                if z > zmax:
                    zmax = z
                    bottleneck = i
        walkers[0].bin = 'lag'
        walkers[-1].bin = 'lead'
        walkers[bottleneck].bin = 'bottleneck'
        return walkers


class OMMStepper(object):
    """
    An OpenMM MD stepper

    """
    def __init__(self, simulation, nsteps):
        self.simulation = simulation
        self.nsteps = nsteps

    def run(self, walkers):
        new_walkers = []
        for w in walkers:
            self.simulation.context.setPositions(w.state.getPositions())
            self.simulation.context.setPeriodicBoxVectors(*w.state.getPeriodicBoxVectors())
            self.simulation.step(self.nsteps)
            state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
            w.update(state)
            new_walkers.append(w)
            
        return new_walkers

class OMMSimpleDistanceProgressCoordinator(object):
    '''
    A WE simulation progress coordinate calculator for OpenMM sims
    
    Adds progress coordinate data to a list of walkers
    
    Initialised with indices
    of the atoms to monitor the distance between.
    '''

    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2
        
    def run(self, walkers):
        import openmm.unit as unit
        if not isinstance(walkers, list):
            walkers = [walkers]
        for i, w in enumerate(walkers):
            crds = w.state.getPositions(asNumpy=True)
            dx = crds[self.a1] - crds[self.a2]
            r = dx * dx
            pc = r.sum().sqrt() / unit.nanometer
            walkers[i].pc = pc
        return walkers
