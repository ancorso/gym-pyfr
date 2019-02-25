#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
import itertools as it
import os
import numpy as np
import h5py
import json
import glob

import mpi4py.rc
mpi4py.rc.initialize = False

import h5py

from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler, get_comm_rank_root, get_mpi
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.progress_bar import ProgressBar
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_name, get_writer_by_extn

def _closest_upts_bf(etypes, eupts, pts):
    for p in pts:
        # Compute the distances between each point and p
        dists = [np.linalg.norm(e - p, axis=2) for e in eupts]

        # Get the index of the closest point to p for each element type
        amins = [np.unravel_index(np.argmin(d), d.shape) for d in dists]

        # Dereference to get the actual distances and locations
        dmins = [d[a] for d, a in zip(dists, amins)]
        plocs = [e[a] for e, a in zip(eupts, amins)]

        # Find the minimum across all element types
        yield min(zip(dmins, plocs, etypes, amins))


def _closest_upts_kd(etypes, eupts, pts):
    from scipy.spatial import cKDTree

    # Flatten the physical location arrays
    feupts = [e.reshape(-1, e.shape[-1]) for e in eupts]

    # For each element type construct a KD-tree of the upt locations
    trees = [cKDTree(f) for f in feupts]

    for p in pts:
        # Query the distance/index of the closest upt to p
        dmins, amins = zip(*[t.query(p) for t in trees])

        # Unravel the indices
        amins = [np.unravel_index(i, e.shape[:2])
                 for i, e in zip(amins, eupts)]

        # Dereference to obtain the precise locations
        plocs = [e[a] for e, a in zip(eupts, amins)]

        # Reduce across element types
        yield min(zip(dmins, plocs, etypes, amins))


def _closest_upts(etypes, eupts, pts):
    try:
        # Attempt to use a KD-tree based approach
        yield from _closest_upts_kd(etypes, eupts, pts)
    except ImportError:
        # Otherwise fall back to brute force
        yield from _closest_upts_bf(etypes, eupts, pts)

class PyFRObj:
    def __init__(self):
        self.ap = ArgumentParser(prog='pyfr')
        self.sp = self.ap.add_subparsers(dest='cmd', help='sub-command help')

        # Common options
        self.ap.add_argument('--verbose', '-v', action='count')

        # Run command
        self.ap_run = self.sp.add_parser('run', help='run --help')
        self.ap_run.add_argument('mesh', help='mesh file')
        self.ap_run.add_argument('cfg', type=FileType('r'), help='config file')
        self.ap_run.set_defaults(process=self.process_run)

        # Restart command
        self.ap_restart = self.sp.add_parser('restart', help='restart --help')
        self.ap_restart.add_argument('mesh', help='mesh file')
        self.ap_restart.add_argument('soln', help='solution file')
        self.ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                                help='new config file')
        self.ap_restart.set_defaults(process=self.process_restart)

        # Options common to run and restart
        backends = sorted(cls.name for cls in subclasses(BaseBackend))
        for p in [self.ap_run, self.ap_restart]:
            p.add_argument('--backend', '-b', choices=backends, required=True,
                           help='backend to use')
            p.add_argument('--progress', '-p', action='store_true',
                           help='show a progress bar')

    def parse(self, cmd_args):
        # Parse the arguments
        self.args = self.ap.parse_args(cmd_args)

    def process(self):
        # Invoke the process method
        self.args.process(self.args)


    def setup_dataframe(self):
        self.elementscls = self.solver.system.elementscls
        self.fmt = 'primitive' # all the configs had this as primitive

        # List of points to be sampled and format
        file = open('samp_pts.txt', 'r')
        self.pts = eval(file.read())

        # Define directory where solution snapshots should be saved
        self.save_dir = 'sol_data/'

        f = h5py.File('base.h5', 'r')
        self.goal_state = np.array(f['sol_data']).flatten()

        # Initial omega
        self.solver.system.omega = 0

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # MPI rank responsible for each point and rank-indexed info
        self._ptsrank = ptsrank = []
        self._ptsinfo = ptsinfo = [[] for i in range(comm.size)]

        # Physical location of the solution points
        plocs = [p.swapaxes(1, 2) for p in self.solver.system.ele_ploc_upts]

        # Load map from point to index
        with open('loc_to_idx.json') as loc_to_idx:
            loc_to_idx_str = json.load(loc_to_idx,)
            self.loc_to_idx = dict()
            for key in loc_to_idx_str:
                self.loc_to_idx[int(key)] = loc_to_idx_str[key]


        # Locate the closest solution points in our partition
        closest = _closest_upts(self.solver.system.ele_types, plocs, self.pts)

        # Process these points
        for cp in closest:
            # Reduce over the distance
            _, mrank = comm.allreduce((cp[0], rank), op=get_mpi('minloc'))

            # Store the rank responsible along with its info
            ptsrank.append(mrank)
            ptsinfo[mrank].append(
                comm.bcast(cp[1:] if rank == mrank else None, root=mrank)
            )

    def _process_samples(self, samps):
        samps = np.array(samps)

        # If necessary then convert to primitive form
        if self.fmt == 'primitive' and samps.size:
            samps = self.elementscls.con_to_pri(samps.T, self.solver.cfg)
            samps = np.array(samps).T

        return samps.tolist()

    def take_action(self, omega):
        comm, rank, root = get_comm_rank_root()
        self.solver.system.omega = float(comm.bcast(omega, root=root))


    def get_state(self):
        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(self.solver.system.ele_types, self.solver.soln))

        # Points we're responsible for sampling
        ourpts = self._ptsinfo[comm.rank]

        # Sample the solution matrices at these points
        samples = [solns[et][ui, :, ei] for _, et, (ui, ei) in ourpts]
        samples = self._process_samples(samples)

        # Gather to the root rank to give a list of points per rank
        samples = comm.gather(samples, root=root)

        # If we're the root rank process the data
        if rank == root:
            data = []

            # Collate
            iters = [zip(pi, sp) for pi, sp in zip(self._ptsinfo, samples)]

            for mrank in self._ptsrank:
                # Unpack
                (ploc, etype, idx), samp = next(iters[mrank])

                # Determine the physical mesh rank
                prank = self.solver.rallocs.mprankmap[mrank]

                # Prepare the output row [[x, y], [rho, rhou, rhouv, E]]
                row = [ploc, samp]

                # Append
                data.append(row)

            # Define info for saving to file
            list_of_files = glob.glob(self.save_dir + '/*')
            if len(list_of_files) == 0:
                file_num = 0
            else:
                latest_file = max(list_of_files, key=os.path.getctime)
                file_num = int(latest_file[-7:-3])

            # Save data in desired format
            # Define freestream values for to be used for cylinder
            rho = 1.0
            P = 1.0
            u = 0.236
            v = 0.0
            e = P/rho/0.4 + 0.5*(u**2 + v**2)
            freestream = np.array([rho, rho*u, rho*v, e])
            sol_data = np.zeros((128, 256, 4))
            sol_data[:, :] = freestream
            for i in range(len(self.loc_to_idx)):
                idx1, idx2 = self.loc_to_idx[i]
                sol_data[idx1, idx2] = data[i][1]

        print("sol data shape: ", sol_data.shape)
        return sol_data



    def step(self):
        if self.solver.tlist[0] == 0:
            self.solver.advance_to(self.solver.tlist[0])
            self.solver.tlist.popleft()


        # print("tlist: ", self.solver.tlist)
        # self.solver.advance_to(self.solver.tlist[3])

        print("\nadvancing to step ", self.solver.tlist[0])
        self.solver.advance_to(self.solver.tlist[0])
        print("\ndone and at time ", self.solver.tlist[0])

        self.solver.tlist.popleft()
        if self.solver.tlist:
            return False
        else:
            self.finalize()
            return True

    def get_reward(self, state):
        return - np.linalg.norm(self.goal_state - state.flatten())


    def finalize(self):
        # Finalise MPI
        MPI.Finalize()


    def _process_common(self, args, mesh, soln, cfg):
        # Prefork to allow us to exec processes after MPI is initialised
        if hasattr(os, 'fork'):
            from pytools.prefork import enable_prefork

            enable_prefork()

        # Import but do not initialise MPI
        from mpi4py import MPI

        # Manually initialise MPI
        MPI.Init()

        # Ensure MPI is suitably cleaned up
        register_finalize_handler()

        # Create a backend
        backend = get_backend(args.backend, cfg)

        # Get the mapping from physical ranks to MPI ranks
        rallocs = get_rank_allocation(mesh, cfg)

        # Construct the solver
        self.solver = get_solver(backend, rallocs, mesh, soln, cfg)


    def process_run(self, args):
        self._process_common(
            args, NativeReader(args.mesh), None, Inifile.load(args.cfg)
        )


    def process_restart(self, args):
        mesh = NativeReader(args.mesh)
        soln = NativeReader(args.soln)

        # Ensure the solution is from the mesh we are using
        if soln['mesh_uuid'] != mesh['mesh_uuid']:
            raise RuntimeError('Invalid solution for mesh.')

        # Process the config file
        if args.cfg:
            cfg = Inifile.load(args.cfg)
        else:
            cfg = Inifile(soln['config'])

        self._process_common(args, mesh, soln, cfg)

