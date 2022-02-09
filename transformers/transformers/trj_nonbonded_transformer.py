from __future__ import print_function, division

import copy
import re
import os
import sys
import time
import json
import shutil
import psutil
import tarfile
import multiprocessing

import subprocess
from subprocess import Popen

from threading import Timer

import xml.etree.ElementTree as ET

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

from schrodinger.utils import sea

import schrodinger.application.desmond.packages.topo as topo
from schrodinger.application.desmond.cms import AtomGroup

import logging

logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)

IS_PLDB = True
PLDB_TMP = '/data/schiffer-pldb/tmp'

SCHRODINGER = '/opt/schrodinger/suite2020-4'  # Where to find schrodinger, important on the PLDB
HOSTS = ['tesla', 'harmony']  # What host to run /schrodinger/desmond on. Must be specified in host file.
NPROC = 1  # Number of cores, if less than 0 the fraction of available cores will be used (e.g. 0.25)
MAX_GROUPS = 999
ENERGY_COMPONENTS = ['nonbonded_elec', 'nonbonded_vdw']
DESMOND_TIMEOUT = 14400  # Timeout for the desmond job

MONITOR_TIME = 120  # Interval over which to monitor GPU usage in seconds
GPU_UTIL_THRESHOLD = 90  # Maximum average gpu usage over monitored time
MEM_UTIL_THRESHOLD = 90  # Maximum average gpu memory usage over monitored time
GPU_MEM_THRESHOLD = 2000  # Requires at least 2000 MBits of free GPU memory on the gpu (not tested)
TIMEOUT = 7200  # Timeout for host search, transformer exits if it can't find a host within time
SSH_TIMEOUT = 5  # Timeout for ssh
CMD_TIMEOUT = 5  # Timeout for remote nvidia-smi call

# execute command remotely using SSH
SSH_CMD = ('ssh -o "ConnectTimeout={ssh_timeout}" {server} '
           'timeout {cmd_timeout}')
# Command for running nvidia-smi locally
NVIDIASMI_CMD = 'nvidia-smi -q -x'
# Command for running nvidia-smi remotely
REMOTE_NVIDIASMI_CMD = '{} {}'.format(SSH_CMD, NVIDIASMI_CMD)

RAW = True  # Whether to save raw output files
DEBUG = False  # Log Debug messages


class VRUN(object):
    """
    Docstring
    """
    ENERGY_GROUP_PREFIX = 'i_ffio_grp_energy'

    def __init__(self, cms_model, trj, cfg_file, groups):
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

    def _launch_vrun(self, jobname, host):
        """
        Launch a desmond vrun job
        :param jobname: <str> jobname
        :param host: <str> hostname
        :return: job a schrodinger.jobcontrol job
        """
        self.cmsfile = self._write_cms(jobname)
        self.cfgfile = self._write_cfg(jobname)

        cmd = ['{}/desmond'.format(SCHRODINGER),
               '-JOBNAME', jobname,
               '-in', str(self.cmsfile),
               '-c', self.cfgfile,
               '-PROCS', '1',
               '-gpu',
               '-WAIT']

        logger.info('Running desomd/vrun')
        logger.debug('$SCHRODINGER_TEMPDIR: {}'.format(os.environ['SCHRODINGER_TMPDIR']))
        cmd.append('-HOST')
        cmd.append(host)

        logger.info(' '.join(cmd))
        p = run_cmd(' '.join(cmd), timeout=DESMOND_TIMEOUT)

        # Check if outputfile was created
        outfile = '{}.engrp'.format(jobname)

        if not os.path.isfile(outfile):
            logger.error('Could not find output energy file')
            logger.error(' '.join(os.listdir(os.getcwd())))
            stdout = ''.join(list(map(bytes2str, p.stdout.readlines())))
            stderr = ''.join(list(map(bytes2str, p.stderr.readlines())))
            logger.error('Desmond returned:\nStdout: {}\nStderr: {}'.format(stdout, stderr))
        else:
            logger.info('Desmond energy file: ' + outfile)

        return outfile

    def calculate_energy(self, jobname, host=None):
        """
        Calculate energy components of a desmond MD Simulation usung Desmond vrun
        :param jobname:<str> The name of the desmond job
        :param host:<str> The name of the host machine
        :return:
        """
        if host is None:
            host = 'localhost'
        return self._launch_vrun(jobname, host=host)


class MonitorGpuUsage(object):
    """
    Query gpu usage in x second intervals
    This class is inspired by:
    https://stackoverflow.com/questions/3393612/run-certain-code-every-n-seconds
    """

    def __init__(self, server, interval, ssh_timeout, cmd_timeout):
        self.server = server
        self.interval = interval
        self.ssh_timeout = ssh_timeout
        self.cmd_timeout = cmd_timeout
        self._timer = None
        self.is_running = False
        self.start()
        self.gpu_util = []
        self.mem_util = []
        self.mem_free = []

    def function(self):
        nvidiasmi = run_nvidiasmi_remote(self.server, self.ssh_timeout, self.cmd_timeout)
        gpu_util, mem_util, mem_free = get_gpu_util(nvidiasmi)
        self.gpu_util.append(gpu_util)
        self.mem_util.append(mem_util)
        self.mem_free.append(mem_free)

    def _run(self):
        self.is_running = False
        self.start()
        self.function()

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def dynamic_cpu_assignment(n_cpus):
    """
    Return the number of CPUs to use.
    If n_cpus is less than zero it is treated as a fraction of the available CPUs
    If n_cpus is more than zero it will simply return n_cpus
    :param n_cpus:
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

    if nproc == 0:
        return 1
    else:
        return nproc


def bytes2str(inp):
    """
    Convert bytes to string if not string already
    Required because Popen returns str in older versions but bytes in python 3+
    """
    if isinstance(inp, bytes):
        return inp.decode('utf-8')
    else:
        return inp


def run_cmd(cmd, timeout=10):
    """
    Run UNIX command with timeout
    """

    def kill(p):
        try:
            p.kill()
        except OSError:
            pass  # ignore

    stdout = subprocess.PIPE
    stderr = subprocess.PIPE
    p = Popen([cmd], shell=True,
              stdout=stdout,
              stderr=stderr)
    t = Timer(timeout, kill, [p])
    t.start()
    exit_code = p.wait()
    t.cancel()
    if exit_code:
        stdout = ''.join(list(map(bytes2str, p.stdout.readlines())))
        stderr = ''.join(list(map(bytes2str, p.stderr.readlines())))
        logger.error('Exit Code: {}'.format(exit_code))
        logger.error('Stdout: {}'.format(stdout))
        logger.error('Stderr: {}'.format(stderr))
        raise RuntimeError('Executing external command failed. \nExit Code: {}'.format(exit_code))
    else:
        return p


def run_nvidiasmi_remote(server, ssh_timeout, cmd_timeout):
    cmd = REMOTE_NVIDIASMI_CMD.format(server=server,
                                      ssh_timeout=ssh_timeout,
                                      cmd_timeout=cmd_timeout)
    p = run_cmd(cmd)
    res = ''.join(list(map(bytes2str, p.stdout.readlines())))
    return ET.fromstring(res) if res is not None else None


def get_gpu_util(nvidiasmi, gpuid=0):
    """
    Get gpu util from nvidiasmi command
    """
    gpu = nvidiasmi.findall('gpu')[gpuid]
    util = gpu.find('utilization')
    gpu_util = float(util.find('gpu_util').text.rstrip('%'))
    mem_util = float(util.find('memory_util').text.rstrip('%'))
    mem_usage = gpu.find('fb_memory_usage')  # This is the device memry (bar1 is used to map device mem)
    mem_free = float(mem_usage.find('free').text.rstrip('MiB'))
    return gpu_util, mem_util, mem_free


def get_average_gpu_util(server, interval=5):
    monitor = MonitorGpuUsage(server, interval, SSH_TIMEOUT, CMD_TIMEOUT)
    monitor.start()
    t = time.time()
    while time.time() - t < MONITOR_TIME:
        time.sleep(1)
    monitor.stop()
    avg_gpu_util = np.mean(monitor.gpu_util)
    avg_mem_util = np.mean(monitor.mem_util)
    avg_mem_free = np.mean(monitor.mem_free)
    return avg_gpu_util, avg_mem_util, avg_mem_free


def get_gpu_host(hosts):
    t = time.time()
    while time.time() - t < TIMEOUT:
        for host in hosts:
            gpu_util, mem_util, mem_free = get_average_gpu_util(host)
            logger.debug('{} average gpu util: {}%'.format(host, gpu_util))
            logger.debug('{} average memory util: {}%'.format(host, mem_util))
            logger.debug('{} average free memory: {}'.format(host, mem_free))
            if gpu_util < GPU_UTIL_THRESHOLD and mem_util < MEM_UTIL_THRESHOLD and mem_free > GPU_MEM_THRESHOLD:
                return host
    logger.error('Unable to find gpu host within {} seconds'.format(TIMEOUT))


def block_averages(x, l):
    """
    Given a vector x return a vector x' of the block averages .
    """

    if l == 1:
        return x

    # If the array x is not a multiple of l drop the first x values so that it becomes one
    if len(x) % l != 0:
        x = x[int(len(x) % l):]

    xp = []
    for i in range(len(x) // int(l)):
        xp.append(np.mean(x[l * i:l + l * i]))

    return np.array(xp)


def ste(x):
    return np.std(x) / np.sqrt(len(x))


def get_bse(x, min_blocks=3):
    steps = np.max((1, len(x) // 100))
    stop = len(x) // min_blocks + steps

    bse = []
    for l in range(1, stop, steps):
        xp = block_averages(x, l)
        bse.append(ste(xp))

    # Fit simple exponential to determine plateau
    def model_func(x, p0, p1):
        return p0 * (1 - np.exp(-p1 * x))

    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, np.arange(len(bse)), bse,
                                                (np.mean(bse), 0.1), maxfev=2000)

    return opt_parms[0]


def _get_error(data, nproc):
    pool = multiprocessing.Pool(processes=nproc)
    err = pool.map(get_bse, data.values())
    return dict(list(zip(data.keys(), err)))


def parse_output(filename, ngroups, energy_components, self_energy=False, correct_nb=True):
    """
    DocString
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


def assign_atomgroups(structure_dict, cms_model, fork):
    """

    """

    # Custom calculation parameters
    if 'nonbonded_params' in structure_dict['files'] or 'nonbonded_params' in structure_dict:

        logger.info('Found custom set of parameters')

        # Get nonbonded params; On the PLDB nonbonded params are provided as a json file
        params = structure_dict['files'].get('nonbonded_params')
        if params is not None:
            with open(structure_dict['files']['nonbonded_params'], 'r') as fh:
                nonbonded_params = json.load(fh)
        else:
            nonbonded_params = structure_dict['nonbonded_params']

        # Get mode: write or append, this is relevant only on the PLDB
        mode = nonbonded_params.get('mode')
        # Get prefious results if exist
        nonbonded_json = structure_dict['files'].get('desmond_nonbonded')

        # On the pldb we want to calculate the default vdw interactions unless specified otherwise
        # Consequently if mode is append, but the default calculation has not been run we need ot for another pipeline
        if nonbonded_json is None and mode == 'a':
            _id = 'default'
            atom_groups, group_ids = _get_solute_by_res(cms_model)
            nonbonded_dict = {_id: {'group_ids': group_ids}}
            if fork is None:
                structure_dict['custom']['pipeline'] = ['trj_nonbonded_pipeline', ]
                fork = ['trj_nonbonded_pipeline', ]
            else:  # Update analysis pipeline
                structure_dict['custom']['pipeline'] = fork + structure_dict['custom']['pipeline']
                fork = ['trj_nonbonded_pipeline', ]
        else:
            # Overwrite existing file
            if mode is None or mode == 'w':
                nonbonded_dict = {}
                if structure_dict['files'].get('desmond_nonbonded') is not None:
                    logger.warning('Overwriting nonbonded calculation')
            # Append to existing file
            elif mode == 'a':
                with open(nonbonded_json, 'r') as fh:
                    nonbonded_dict = json.load(fh)
                # Older versions did not support multiple nonbonded transformer runs
                # This will retroactively update older structures on the PLDB
                if 'default' not in nonbonded_dict:
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
            nonbonded_dict[_id] = {'group_ids': group_ids}
    else:  # Default (Ligand + Protein) calculation
        _id = 'default'
        atom_groups, group_ids = _get_solute_by_res(cms_model)
        nonbonded_dict = {_id: {'group_ids': group_ids}}

    logger.info('number of groups: {}'.format(len(atom_groups)))

    return _id, atom_groups, nonbonded_dict, structure_dict, fork


def clean_pldb_tmp(cwd, tmp):
    logger.info('Cleaning up tempdir')
    for f in os.listdir(tmp):
        shutil.move(os.path.join(tmp, f), os.path.join(cwd, f))
    os.chdir(cwd)
    os.rmdir(tmp)


def _process(structure_dict):
    """
    DocString
    :param structure_dict:
    :return:
    """

    fork = None
    # Check if transformers is called as part of a pipeline
    if 'pipeline' in structure_dict['custom']:
        pipeline = structure_dict['custom']['pipeline']
        fork = [pipeline[0], ]
        if len(pipeline) == 1:
            del (structure_dict['custom']['pipeline'])
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

    _id, atom_groups, nonbonded_dict, structure_dict, fork = assign_atomgroups(structure_dict, cms_model, fork)

    if len(HOSTS) == 1 and HOSTS[0] in ('localhost', '127.0.0.1'):
        host = HOSTS[0]
        logger.warning('Host is {}. Will not look for free gpu'.format(host))
    else:
        logger.info('Finding free host')
        logger.debug('Hosts: ' + ', '.join(HOSTS))
        host = get_gpu_host(HOSTS)

    logger.info('Running desmond job on: {}'.format(host))
    vrun_obj = VRUN(cms_model, trj_dir, cfgfile, atom_groups)
    energy_group_file = vrun_obj.calculate_energy(outname, host=host)

    if RAW:
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
                if not os.path.isdir('./tmp'):
                    os.mkdir('./tmp')
                tar.extractall(path='./tmp')
                members = tar.getmembers()
            with tarfile.open(outfile_raw, 'w:bz2') as tar:
                for member in members:
                    filename = member.name
                    if filename.split('_')[0] not in ('custom', 'default'):  # Backward Compatibility
                        logger.warning('Updating file names to new version')
                        os.rename(os.path.join('./tmp', filename), os.path.join('./tmp', 'default_' + filename))
                        filename = 'default_' + filename
                    tar.add(os.path.join('./tmp', filename),
                            arcname=filename)  # arcname to add the file without the path
                for fn in [energy_group_file, '{}_atom_groups.json'.format(_id)]:
                    tar.add(fn)

    # Get time and energy components
    sim_time, component_dict = parse_output(energy_group_file, len(atom_groups), ENERGY_COMPONENTS)
    # calculate mean and error
    results = {}
    logger.info('Calculating average potential')
    for comp in ENERGY_COMPONENTS:
        pair_dict = component_dict[comp]
        data_dict = {}
        results[comp] = {}
        results[comp]['keys'] = []
        results[comp]['mean_potential'] = []
        results[comp]['error'] = []
        for pair, energy in pair_dict.items():
            mean_potential = np.mean(energy)
            # skip zero and near 0 potential (The latter sometimes causes overflow issues during error calc.)
            if np.abs(mean_potential) < 1e-6:
                continue
            results[comp]['keys'].append(list(pair))
            results[comp]['mean_potential'].append(mean_potential)
            data_dict[pair] = energy  # Only store non-zero energies
        # Calculate the error separately over multiple processes
        nproc = dynamic_cpu_assignment(NPROC)
        logger.debug('Calculating error for {}'.format(comp))
        error_dict = _get_error(data_dict, nproc)
        for k in results[comp]['keys']:
            results[comp]['error'].append(error_dict[tuple(k)])

    nonbonded_dict[_id]['energy_components'] = ENERGY_COMPONENTS
    nonbonded_dict[_id]['time'] = sim_time
    nonbonded_dict[_id]['results'] = results

    with open(outfile, 'w') as f:
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


def run(structure_dict_list):
    for structure_dict in structure_dict_list:
        for new_structure_dict in _process(structure_dict):
            yield new_structure_dict


def parse_args():
    """
    Argument parser when script is run from commandline
    :return:
    """
    description = '''
    Calculate nonbonded interactions between atom groups\n
    In the background this calls desmond/vrun to recalculate the energies from a set of coordinates.\n
    Because Schrodinger/desmond only supports gpu jobs, you need access to at least on host with a gpu.\n
    All nonbonded interactions between groups are returned.\n
    Default groups comprise all protein and ligand residues, however different group can be specified; Provided 
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
                        help='custom group parameters.\n Example: {"mode": "w", "asl": ["protein", "ligand"],'
                             ' "group_id": ["protein", "ligand"]}')
    parser.add_argument('--raw',
                        dest='raw',
                        default=False,
                        action='store_true',
                        help='Safe raw output files, does not require a argumet. Default: false')
    parser.add_argument('-n',
                        '--nproc',
                        type=int,
                        dest='nproc',
                        default=8,
                        help='Number of cores to use (only used for calculating errors)\nDefault: 8')
    parser.add_argument('--host',
                        type=str,
                        nargs='+',
                        dest='host',
                        default=None,
                        help='Host(s) to run schrodinger/desmond on')
    parser.add_argument('--debug',
                        dest='debug',
                        default=False,
                        action='store_true',
                        help='Set lgo level to debug')

    return parser.parse_args()


def get_logger():
    logger = logging.getLogger(__name__)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join('./', os.path.split(__file__)[-1][:-3] + '.log'), mode='w')
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    if DEBUG:
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
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
    prefix = args.prefix
    cmsfile, trjtar, cfgfile = args.infiles

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

    if args.params is not None:
        if os.path.isfile(args.params):
            with open(args.params) as fh:
                structure_dict_list[0]['nonbonded_params'] = json.load(fh)
        else:
            nonbonded_params = json.loads(args.params)
            structure_dict_list[0]['nonbonded_params'] = nonbonded_params

    for sd in run(structure_dict_list):
        with open('{}_trj_nonbonded_transformer.json'.format(prefix), 'w') as outfile:
            json.dump(sd, outfile)


if __name__ == '__main__':
    import argparse

    args = parse_args()

    SCHRODINGER = os.environ['SCHRODINGER']
    NPROC = args.nproc
    HOSTS = args.host
    RAW = args.raw
    DEBUG = args.debug

    logger = get_logger()
    logger.info(' '.join(sys.argv))
    logger.debug('$SCHRODINGER = {}'.format(SCHRODINGER))

    main(args)
