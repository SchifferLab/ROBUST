from __future__ import print_function, division

import copy
import re
import os
import json
import multiprocessing
import psutil
import tarfile
import time
import itertools

import xml.etree.ElementTree as ET

import numpy as np
import scipy as sp
import scipy.optimize

from schrodinger.utils import sea
from schrodinger.job import jobcontrol

import schrodinger.application.desmond.packages.topo as topo
from schrodinger.application.desmond.cms import AtomGroup

import logging

logger = logging.getLogger(__name__)

try:
    import paramiko
except:
    logger.warning('Could not import paramiko, remote jobs will be submitted without checking gpu usage')

HOST = ['vif', 'tesla'] # What host to run /schrodinger/desmond on. Must be specified in host file.
USER = 'pldbuser'
NPROC = 1  # Number of cores, if less than 0 the fraction of available cores will be used (e.g. 0.25)
CPU = False # Run desmond CPU, this requires the deprecated DESMOND_MAIN license token
MAX_GROUPS = 999
ENERGY_COMPONENTS = ['nonbonded_elec', 'nonbonded_vdw']


class BAVERAGES(multiprocessing.Process):
    """
    Give any 1d timeseries this class calculates the block averaged standard error
    See:
        Flyvbjerg, Henrik, and Henrik Gordon Petersen.
        "Error estimates on averages of correlated data."
        The Journal of Chemical Physics 91.1 (1989): 461-466.

        Grossfield, Alan, and Daniel M. Zuckerman.
        "Quantifying uncertainty and sampling quality in biomolecular simulations."
         Annual reports in computational chemistry 5 (2009): 23-48.
    """

    def __init__(self, in_queue, out_queue, min_m=10):

        multiprocessing.Process.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.min_m = min_m

    def transform(self, data_array, block_length):
        """
        Provided with a series of values and (a) blocklength (l),
        this function returns an series of block averages.
        """

        # If the array (a) is not a multple of l drop the first x values so that it becomes one
        if len(data_array) % block_length != 0:
            data_array = data_array[int(len(data_array) % block_length):]

        o = []
        for i in range(len(data_array) // int(block_length)):
            o.append(np.mean(data_array[block_length * i:block_length + block_length * i]))

        return np.array(o)

    def run(self):

        while True:
            data = self.in_queue.get()
            # poison pill
            if data == 'STOP':
                break
            i, data_array = data
            # blocksize 1 => full run
            block_length = 2
            blocks = np.array([1, ])
            block_stderr = [np.std(data_array) / np.sqrt(len(data_array))]
            # Transform the series until nblocks = 2
            while len(data_array) // block_length >= self.min_m:
                b = self.transform(data_array, block_length)
                block_stderr.append(np.std(b) / np.sqrt(len(b)))
                block_length += 1
                if len(data_array) // block_length < self.min_m:
                    blocks = np.arange(block_length - 1) + 1

            # Simple exponential function
            def model_func(x, p0, p1):
                return p0 * (1 - np.exp(-p1 * x))

            # Fit curve
            opt_parms, parm_cov = sp.optimize.curve_fit(model_func, blocks, block_stderr, maxfev=2000)
            error_estimate = opt_parms[0]
            while True:
                if self.out_queue.full():
                    time.sleep(5)
                else:
                    self.out_queue.put([i, error_estimate])
                    break
        return


class VRUN(object):
    """
    Docstring
    """
    ENERGY_GROUP_PREFIX = 'i_ffio_grp_energy'

    def __init__(self, cms_model, trj, cfg_file, groups, cpu=False):
        self.cpu = cpu # Run desmond cpu (Deprecated)
        self.cms_model = cms_model
        self.trj = trj
        # parse input cfg
        self.cfg = self._parse_cfg(cfg_file)
        if 'ORIG_CFG' not in self.cfg:
            raise RuntimeError('ORIG_CFG is missing from {}'.format(cfg_file))
        if len(groups) < 2:
            raise RuntimeError('vrun requires two or more energy groups')
        elif len(groups) > MAX_GROUPS:
            raise RuntimeError('Too many groups specified! {}'.format(len(groups)))
        else:
            self.groups = groups

    @staticmethod
    def _parse_cfg(fn):
        try:
            with open(fn, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise RuntimeError('Fail to read cfg file: {}\n{}'.format(fn, e))
        try:
            cfg = sea.Map(''.join(lines))
        except Exception as e:
            raise RuntimeError('Failed to parse cfg file:  {}\n{}'.format(fn, e))
        return cfg

    def _write_cms(self, fn):
        """
        add energy groups to input cms file
        """
        out_cms = fn + '_vrun.cms'
        # Safe group assignments to prevent assigning the same set of atoms to more than one group
        atom_group_id = np.zeros(self.cms_model.atom_total, dtype=np.int)
        # group_id 0 is reserved
        for group_id, group in enumerate(self.groups, start=1):
            indices = group
            if atom_group_id[indices].any():
                raise RuntimeError('multiple groups contain same atom')
            atom_group_id[indices] = group_id

        # set energy group property

        # atom index begins from 1
        atom_groups = [
            AtomGroup(
                self.cms_model.select_atom_comp('(atom.num %s)' % ','.join(map(str, g))),
                'energy', group_id)
            for group_id, g in enumerate(self.groups, start=1)
            if g
        ]
        self.cms_model.set_atom_group(atom_groups)

        # set energy group property for component cts
        index = 0
        for atom in self.cms_model.atom:
            atom.property[self.ENERGY_GROUP_PREFIX] = atom_group_id[index]
            index = index + 1

        # Write updated cms file
        self.cms_model.write(out_cms)

        return out_cms

    def _write_cfg(self, fn, t_start=None, t_interval=None):
        """
        Prepare a desmond config file for vrun
        (See page 30 Desmond user guide)
        """
        out_cfg = fn + '_vrun.cfg'
        with open(out_cfg, 'w') as f:
            new_cfg = copy.copy(self.cfg['ORIG_CFG'])
            # use energy groups plugin
            new_cfg.update('energy_group = on')
            # frameset
            new_cfg.update('vrun_frameset="{}"'.format(os.path.abspath(self.trj)))
            new_cfg.update('time="inf"')
            new_cfg.update('trajectory.interval=0')
            new_cfg.update('backend.vrun.plugin.list = ["energy_groups" "status"]')
            new_cfg.update('backend.vrun.plugin.energy_groups.name="{}"'.format(fn + '.engrp'))
            if t_start:
                # When to start recording
                new_cfg.update('backend.vrun.plugin.energy_groups.first={}'.format(t_start))
            if t_interval:
                new_cfg.update('backend.vrun.plugin.energy_groups.interval={}'.format(t_interval))

            new_cfg.update('backend.vrun.plugin.energy_groups.options = [pressure_tensor corr_energy]')
            # input cms file
            new_cfg.update('backend.vrun.plugin.maeff_output.bootfile="{}"'.format(os.path.abspath(self.cmsfile)))
            # write cfg to file
            f.write(str(new_cfg))
        return out_cfg

    def _launch_vrun(self, jobname, nproc=1, host='localhost'):
        """
        Launch a desmond vrun job
        :param jobname: <str> jobname
        :param nproc: <int> number of cores to use
        :param host: <str> hostname
        :return: job a schrodinger.jobcontrol job
        """
        self.cmsfile = self._write_cms(jobname)
        self.cfgfile = self._write_cfg(jobname)

        cmd = ['desmond',
               '-JOBNAME', jobname,
               '-in', str(self.cmsfile),
               '-c', self.cfgfile]

        if not self.cpu:
            cmd.extend(['-PROCS', '1', '-gpu'])
        else:
            cmd.extend(['-PROCS', str(nproc)])

        if host is not None:
            logger.info('submittign job to: {}'.format(host))
            cmd.append('-HOST')
            cmd.append(host)
        else:
            cmd.extend(['-HOST', 'localhost'])

        logger.info('$SCHRODINGER/' + ' '.join(cmd))
        job = jobcontrol.launch_job(cmd)

        return job

    def calculate_energy(self, jobname, nproc=1, host=None):
        """
        Calculate energy components of a desmond MD Simulation usung Desmond vrun
        :param jobname:<str> The name of the desmond job
        :param nproc:<int> Number of processors to use
        :param host:<str> The name of the host machine
        :return:
        """
        if host is None:
            host = 'localhost'
        job = self._launch_vrun(jobname, nproc=nproc, host=host)
        # wait for job to finish
        job.wait()
        if job.succeeded():
            return True, job.getOutputFiles()
        else:
            return False, job.getOutputFiles()


def get_desmond_cpus(n_cpus):
    """

    Legacy code: This is no longer required since schrodinger/desmond no longer supports cpus

    Desmond distributes cpus accross the x, y and Z axis.
    It can only use 2, 3 or 5 or the powers fo these numbers.
    Example: X->2cpus, Y->cpus, Z->5cpus  Total: 30 cpus

    :param n_cpus:
    :return:
    """
    if n_cpus < 8:
        return 1
    allowed_n = [2, 4, 6, 8, 10, 12, 14, 3, 9, 15, 5]
    combinations = np.array(list(itertools.product(allowed_n, allowed_n, allowed_n)))  # Get all combinations
    products = np.array(list(map(np.product, combinations)))
    difference = n_cpus - products
    top_solution = products[difference >= 0][np.argmin(difference[difference >= 0])]
    return top_solution


def dynamic_cpu_assignment(n_cpus, desmond=False):
    """

    Legacy code, this is no longer required because schrodinger/desmond no longer supports cpus

    Return the number of CPUs to use.
    If n_cpus is less than zero it is treated as a fraction of the available CPUs
    If n_cpus is more than zero it will simply return n_cpus
    :param n_cpus:
    :param desmond:
    :return:
    """
    if n_cpus >= 1:
        return int(n_cpus)
    # get number of cpus
    total_cpus = psutil.cpu_count() * n_cpus  # Use at most x percent of the available cpus
    requested_cpus = total_cpus * n_cpus
    # get cpu usage off a 2s window
    cpu_usage = psutil.cpu_percent(interval=2)
    # NOTE available cpus is only an approximation of the available capacity
    free_cpus = int(total_cpus - (total_cpus * (cpu_usage / 100)))
    if free_cpus > requested_cpus:
        nproc = free_cpus
    else:
        nproc = int(total_cpus * n_cpus)
    if desmond:
        return get_desmond_cpus(nproc)

    elif nproc == 0:
        return 1
    else:
        return nproc


def _get_error(data, nproc):
    """
    Docstring
    :param data:
    :param nproc:
    :return:
    """
    combined_error = {}
    id2key = {}

    in_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    workers = [BAVERAGES(in_queue, out_queue) for _ in range(nproc)]
    for w in workers:
        w.start()

    for i, (key, value) in enumerate(data.items()):
        id2key[i] = key
        in_queue.put([i, value])

    for _ in range(len(data.keys())):
        i, error = out_queue.get()
        combined_error[id2key[i]] = error
    for _ in range(nproc):
        in_queue.put('STOP')
    for w in workers:
        w.join()
    return combined_error


def parse_output(filename, ngroups, energy_components, self_energy=False, correct_nb=True):
    """
    Add some documentation here
    """
    allowed_energy_terms = ['angle', 'dihedral', 'far_exclusion', 'far_terms', 'nonbonded_elec', 'nonbonded_vdw',
                            'pair_elec', 'pair_vdw', 'stretch', 'Total']

    # convert energy to list
    if type(energy_components) != list:
        energy_components = list(energy_components)

    # output dictionary
    component_dict = {}
    sim_time = []
    # Check if passed energies are correct
    for row in energy_components:
        if row not in allowed_energy_terms:
            logger.error('{} is not a recognized energy term.\nKnown energy terms are:'.format(row))
            for et in allowed_energy_terms:
                logger.error(et)
            raise KeyError('provided unkown energy term')
        # add energy term to component_dict dict
        component_dict[row] = {}

    # If correct_nb return difference between pair and nonbonded
    if correct_nb:
        if 'nonbonded_elec' in component_dict.keys():
            if 'pair_elec' not in component_dict.keys():
                component_dict['pair_elec'] = {}
        if 'nonbonded_vdw' in component_dict.keys():
            if 'pair_vdw' not in component_dict.keys():
                component_dict['pair_vdw'] = {}
    # create group indices
    column_indices = {}
    # with n groups specified by user, skip n+3 columns, which are
    # energy tag, (time), E00, E01 ... E0n since group 0 is reserved
    c = 3 + ngroups
    for i in range(ngroups):
        for j in range(i, ngroups):
            if self_energy:
                column_indices[c] = (i, j)
            else:
                if i != j:
                    column_indices[c] = (i, j)
            c += 1

    # populate component_dict dictionaries with pairs
    for comp_name in component_dict.keys():
        for pair in column_indices.values():
            component_dict[comp_name][pair] = []

    energy_term_pattern = r'(angle|dihedral|far_exclusion|far_terms|nonbonded_elec|nonbonded_vdw|pair_elec|pair_vdw' \
                          r'|stretch|Total) '
    floating_number_pattern = r'([-+]?\b(?:[0-9]*\.)?[0-9]+(?:[eE][-+]?[0-9]+)?\b)'

    pattern = energy_term_pattern + r'\s+' + r'\(' + floating_number_pattern + r'\)'
    # Regular expression for energy terms
    re_energy = re.compile(pattern)

    # Column positions of name & time
    energy_comp_name_column = 1
    time_column = 2

    with open(filename, 'r') as f:
        for line in f:
            energy_match = re_energy.match(line)
            if energy_match:
                t = float(energy_match.group(time_column))
                energy_comp = energy_match.group(energy_comp_name_column).strip()
                if energy_comp in component_dict.keys():
                    linestr = line.split()
                    for col, pair in column_indices.items():
                        component_dict[energy_comp][pair].append(float(linestr[col]))
                if t not in sim_time:
                    sim_time.append(t)
    # Calculated the corrected nb potential
    if correct_nb:
        if 'nonbonded_vdw' in component_dict.keys():
            for pair in component_dict['nonbonded_vdw'].keys():
                c = 0
                for nonbonded, pairwise in zip(component_dict['nonbonded_vdw'][pair], component_dict['pair_vdw'][pair]):
                    component_dict['nonbonded_vdw'][pair][c] = nonbonded - pairwise
                    c += 1
        if 'nonbonded_elec' in component_dict.keys():
            for pair in component_dict['nonbonded_elec'].keys():
                c = 0
                for nonbonded, pairwise in zip(component_dict['nonbonded_elec'][pair],
                                               component_dict['pair_elec'][pair]):
                    component_dict['nonbonded_elec'][pair][c] = nonbonded - pairwise

    return sim_time, component_dict


def _get_solute_by_res(cms_model):
    """
    Return a list of atom_groups and residue ids
    :param cms_model: <schrodinger.application.desmond.cms.Cms>
    :return atom_groups: <list> atom_ids of all residues with more than 3 atoms
    :return resids: <list> residue ids (resnum,chain) of all residues with more than 3 atoms
    """
    atom_groups = []
    resids = []
    for res in cms_model.residue:
        aids = res.getAtomIndices()
        # NOTE this definition of solute fails for membrane systems
        if len(aids) > 3:
            atom_groups.append(aids)
            resids.append((res.resnum, res.chain.strip()))
    return atom_groups, resids


def get_host(hosts, user):
    """
    Docstring
    :param hosts
    :param user
    :return:
    """

    if isinstance(hosts, str):
        return hosts
    elif isinstance(hosts, list):
        for host in hosts:
            # TODO
            return host


def _process(structure_dict):
    """
    DocString
    :param structure_dict:
    :return:
    """

    host = get_host(HOST, user=USER)
    fork = None
    # Check if transformers is called as part of a pipeline
    if 'pipeline' in structure_dict['custom']:
        pipeline = structure_dict['custom']['pipeline']
        fork = [pipeline[0], ]
        if len(pipeline) == 1:
            del(structure_dict['custom']['pipeline'])
        else:
            structure_dict['custom']['pipeline'] = pipeline[1:]

    structure_code = structure_dict['structure']['code']

    outname = '{}_trj_nonbonded'.format(structure_code)
    outfile = outname + '.json'
    outfile_raw = outname + '.tar.bz2'

    # desmond_cms file
    cmsfile = structure_dict['files']['desmond_cms']
    msys_model, cms_model = topo.read_cms(str(cmsfile))

    # desmond_cfg file
    cfgfile = structure_dict['files']['desmond_cfg']

    # desmond frame archive
    trjtar = structure_dict['files']['desmond_trjtar']

    # If run from command line it does not make sense to provide a tarfile
    if os.path.isdir(trjtar):
        trj_dir = trjtar
    elif tarfile.is_tarfile(trjtar):
        with tarfile.open(name=trjtar, mode='r:gz') as tfile:
            tfile.extractall()
            logger.info('extracting frameset')
            trj_dir = tfile.getnames()[0]
    else:
        raise RuntimeError('trjtar is neither a directory nor a tarfile')

    logger.info('creating atomgroups')

    if 'nonbonded_params' in structure_dict['files']:  # Custom calculation parameters
        logger.info('Found custom set of parameters')

        nonbonded_json = structure_dict['files'].get('desmond_nonbonded')
        with open(structure_dict['files']['nonbonded_params'], 'r') as fh:
            nonbonded_params = json.load(fh)

        mode = nonbonded_params.get('mode')

        # If append and no default, calculate default nonbonded interactions first
        if nonbonded_json is None and mode == 'a':
            nonbonded_dict = {}
            _id = 'default'
            atom_groups, group_ids = _get_solute_by_res(cms_model)
            if fork is None:
                fork = ['trj_nonbonded_pipeline', ]
            else:  # Update analysis pipeline
                if 'pipeline' in structure_dict['custom']:
                    structure_dict['custom']['pipeline'] = fork + structure_dict['custom']['pipeline']
                    fork = ['trj_nonbonded_pipeline', ]
                else:
                    structure_dict['custom']['pipeline'] = fork
                    fork = ['trj_nonbonded_pipeline', ]
        else:
            # Overwrite
            if mode is None or mode == 'w':
                nonbonded_dict = {}
                if structure_dict['files'].get('desmond_nonbonded') is not None:
                    logger.warning('Overwriting nonbonded calculation')
            # Append
            elif mode == 'a':
                with open(nonbonded_json, 'r') as fh:
                    nonbonded_dict = json.load(fh)
                if 'default' not in nonbonded_dict:  # Backward compatibility
                    logger.warning('Updating old nonbonded results to new version')
                    nonbonded_dict = {'default': nonbonded_dict, }
            # Create ID
            for i in range(999):
                _id = 'custom_{}'.format(i)
                if _id not in nonbonded_dict:
                    break
            # Create atom groups
            atom_groups = []
            for asl in nonbonded_params['asl']:
                atom_groups.append(list(map(int, topo.asl2gids(cms_model, asl))))  # asl2gids: returns np.int
            # Check if custom group ids are provided
            if 'group_ids' in nonbonded_params:
                group_ids = nonbonded_params['group_ids']
            else:
                group_ids = nonbonded_params['asl']
    else:  # Default (Ligand + Protein) calculation
        nonbonded_dict = {}
        _id = 'default'
        atom_groups, group_ids = _get_solute_by_res(cms_model)

    logger.info('number of groups: {}'.format(len(atom_groups)))

    # calculate energy terms
    nproc = dynamic_cpu_assignment(NPROC, desmond=True)

    vrun_obj = VRUN(cms_model, trj_dir, cfgfile, atom_groups, cpu=CPU)
    vrun_out = vrun_obj.calculate_energy(outname, nproc=nproc, host=host)

    # if vrun successful create output files
    if vrun_out[0]:
        energy_group_file = ''
        # Get energygroup file
        for f in vrun_out[1]:
            if f[-6:] == '.engrp':
                energy_group_file = '{}_{}'.format(_id, f)
                os.rename(f, energy_group_file)
        if not energy_group_file:
            raise RuntimeError('No energy group file returned by desmond vrun')

        # Write raw output
        with open('{}_atom_groups.json'.format(_id), 'w') as f:
            json.dump(atom_groups, f)
        nonbonded_raw = structure_dict['files'].get('desmond_nonbonded_raw')
        if nonbonded_raw is None:
            with tarfile.open(outfile_raw, 'w:bz2') as tar:
                for fn in [energy_group_file, '{}_atom_groups.json'.format(_id)]:
                    tar.add(fn)
        else:
            with tarfile.open(nonbonded_raw, 'r:bz2') as tar:
                tar.extractall()
                members = tar.getmembers()
            with tarfile.open(outfile_raw, 'w:bz2') as tar:
                for member in members:
                    filename = member.name
                    if filename.split('_')[0] not in ('custom', 'default'):  # Backward Compatibility
                        logger.warning('Updating file names to new version')
                        os.rename(filename, 'default_'+filename)
                        filename = 'default_'+filename
                    tar.add(filename)
                for fn in [energy_group_file, '{}_atom_groups.json'.format(_id)]:
                    tar.add(fn)

        # Get time and energy components
        sim_time, component_dict = parse_output(energy_group_file, len(atom_groups), ENERGY_COMPONENTS)
        # calculate mean and error
        results = {}
        for comp in ENERGY_COMPONENTS:
            pair_dict = component_dict[comp]
            data_dict = {}
            results[comp] = {}
            results[comp]['keys'] = []
            results[comp]['mean_potential'] = []
            results[comp]['error'] = []
            for pair, energy in pair_dict.items():
                # skip no interactions
                if np.mean(energy) == 0:
                    continue
                results[comp]['keys'].append(list(pair))
                results[comp]['mean_potential'].append(np.mean(energy))
                data_dict[pair] = energy  # Only store non-zero energies

            # Calculate the error separately over multiple processes
            nproc = dynamic_cpu_assignment(NPROC)
            error_dict = _get_error(data_dict, nproc)
            for k in results[comp]['keys']:
                results[comp]['error'].append(error_dict[tuple(k)])

        with open(outfile, 'w') as f:
            nonbonded_dict[_id] = {'energy_components': ENERGY_COMPONENTS,
                                   'group_ids': group_ids,
                                   'time': sim_time,
                                   'results': results}
            json.dump(nonbonded_dict, f)

        transformer_dict = {'structure': {
            'parent_structure_id':
                    structure_dict['structure']['structure_id']
            },
            'files': {'desmond_nonbonded': outfile,
                      'desmond_nonbonded_raw': outfile_raw},
            'custom': structure_dict['custom'],
        }
        if fork is not None:
            logger.info('Forking pipeline: ' + ' '.join(fork))
            transformer_dict['control'] = {'forks': fork}
        yield transformer_dict

    else:
        logger.error('vrun failed')  # TODO Ellaborate
        if fork is not None:
            logger.info('Forking pipeline: ' + ' '.join(fork))
            yield {
                'structure': {
                    'structure_id':
                        structure_dict['structure']['structure_id']
                },
                'control': {'flags': {'description': 'VRUN_FAILED', 'messages': 'TODO'},
                            'forks': fork},
                'custom': structure_dict['custom'],
            }


def run(structure_dict_list):
    for structure_dict in structure_dict_list:
        for new_structure_dict in _process(structure_dict):
            yield new_structure_dict


def parse_args():
    """
    Argument parser when script is run from commandline
    :return:
    """
    description='''
    Calculate nonbonded interactions between atom groups\n
    This transformer returns nonbonded interactions between distinct atom groups.
    It utilizes desmond vrun to do the calculation.
    All nonbonded interactions between groups are returned.\n
    By default the groups comprise all protein and ligand residues, however different group can be specified provided 
    that they do not share a subset of atoms.\n
    Results are returned in a complex json dictionary. The keys in the first layer identify the calculations, where 
    the calculation with default parameters is given the key "default" and all subsequent calculations called 
    custom_[0-999]. For each calculation group_id are provided. For the default calculation this is the
    residue identifier: (resnum, chain) for custom calculations, it is the group asl. The results are divided into 
    electrostatic and vdw protential, in addition to  the average pairwise interaction and ids the
    associated error calculated by block averaging is provided.
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infiles',
                        type=str,
                        nargs='+',
                        help='Simulation cmsfile and frames and cfg file')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='similarity_search',
                        help='Outfile prefix')
    parser.add_argument('--params',
                        type=str,
                        dest='params',
                        default=None,
                        help='custom group parameters.\n Example: {mode: "w", asl: {0: "protein",'
                             ' 1: "ligand"}, group_id: {0: "protein", 1: "ligand"}}')
    parser.add_argument('--cpu',
                        default=False,
                        action='store_true',
                        help='Run desmond cpu, this requires the deprecated DESMOND_MAIN license token'
                        )
    parser.add_argument('-n',
                        '--nproc',
                        type=int,
                        dest='nproc',
                        default=16,
                        help='Number of cores to use for calculation.\nDefault: 16')
    parser.add_argument('-h',
                        '--host',
                        type=int,
                        dest='host',
                        default='localhost',
                        help='Host to run schrodinger/desmond on')
    parser.add_argument('-u',
                        '--user',
                        type=str,
                        default='pldbuser',
                        help='Username to submit remote jobs under')

    return parser.parse_args()


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join('./', os.path.split(__file__)[-1][:-3] + '.log'), mode='w')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main(args):

    global NPROC
    global CPU
    global HOST
    global USER

    CPU = args.cpu
    NPROC = args.nproc
    HOST = args.host
    USER = args.user

    if CPU:
        logger.warning('Running Desmond_cpu, this requires the DESMOND_MAIN license token')
    else:
        logger.info('Running Desmond_gpu')
    prefix = args.prefix
    cmsfile, trjtar, cfgfile = args.infiles
    nonbonded_params = args.params

    structure_dict_list = [
        {'files': {
            'desmond_cms': cmsfile,
            'desmond_trjtar': trjtar,
            'desmond_cfg': cfgfile,
        },
            'structure': {'code': prefix,
                          'structure_id': 0},
            'custom': {}
        }
    ]
    if nonbonded_params is not None:
        structure_dict_list[0]['nonbonded_params'] = nonbonded_params
    for sd in run(structure_dict_list):
        with open('{}_trj_nonbonded_transformer.json'.format(prefix), 'w') as outfile:
            json.dump(sd, outfile)


if __name__ == '__main__':
    import argparse
    args = parse_args()
    logger = get_logger()
    main(args)