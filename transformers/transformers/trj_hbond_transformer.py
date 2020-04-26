from __future__ import print_function, division

import os
import tarfile
import json
import psutil
import multiprocessing
import time

from schrodinger.infra import mm
import schrodinger.application.desmond.packages.topo as topo
import schrodinger.application.desmond.packages.traj as traj

import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize
from scipy import spatial

import logging

logger = logging.getLogger(__name__)

NPROC = 16
STEP = 2  # Process every nth frame
LIGAND_ASL = None
SOLVENT_ASL = 'solvent'  # solvent molecules in maestro asl


class BlockAverage(multiprocessing.Process):
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

    def __init__(self, in_queue, out_queue, min_blocks=5):
        """

        :param in_queue:
        :param out_queue:
        :param min_blocks:
        """

        multiprocessing.Process.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.min_blocks = min_blocks

    def transform(self, data_array, block_length):
        """

        :param data_array:
        :param block_length:
        :return:
        """

        # If the array (a) is not a multiple of l drop the first x values so that it becomes one
        if len(data_array) % block_length != 0:
            data_array = data_array[int(len(data_array) % block_length):]

        nblocks = len(data_array) // int(block_length)

        block_averages = np.zeros(nblocks)
        for i in range(nblocks):
            block_averages[i] = np.mean(data_array[block_length * i:block_length + block_length * i])

        return np.array(block_averages)

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
            while len(data_array) // block_length >= self.min_blocks:
                b = self.transform(data_array, block_length)
                block_stderr.append(np.std(b) / np.sqrt(len(b)))
                block_length += 1
                if len(data_array) // block_length < self.min_blocks:
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


class HydrongeBondAnalysis(multiprocessing.Process):
    """
    Python class for  calculating hydrogen bonds
    The main function assign_hbonds assigns hydrogen bonds according to a geometric criterion.

    H = hydrogen
    A = acceptor
    AA = atom bonded to acceptor
    D = atom bonded to hydrogen
    :: = potential hbond
    - = covalent bond

    Geometric criteria:
    1. the H::A distance must be less than or equal to 3.0 Angstrom
    2. the D-H::A angle must be at least 110 degree
    3. the H::A-AA angle must be at least 90 degree.
    """

    def __init__(self, _id, queue, cms_file, trj_dir, frames=None, asl='protein or ligand', ndx=None,
                 dmax=2.5, donor_angle=120.0, acceptor_angle=90.0):

        multiprocessing.Process.__init__(self)

        self._id = _id
        self.queue = queue

        self.dmax = dmax
        self.donor_angle = donor_angle
        self.acceptor_angle = acceptor_angle

        # Load cms_mode
        self.msys_model, self.cms_model = topo.read_cms(str(cms_file))

        # Get atom gids for which to get hbonds
        if ndx is None:
            self.ndx = topo.asl2gids(self.cms_model, asl)
        else:
            self.ndx = ndx

        # Get atom gids for solvent
        self.ndx_water = topo.asl2gids(self.cms_model, SOLVENT_ASL)

        # Load frame list
        if frames is not None:
            self.frame_list = [frame for (i, frame) in enumerate(traj.read_traj(str(trj_dir))) if i in frames]
        else:
            self.frame_list = traj.read_traj(str(trj_dir))

        # set donor & acceptor atoms
        self.donor = {}
        self.acceptor = {}

        self._set_donor_acceptor()

        # set bonded atoms
        self.bonded_atoms = {}
        self._set_bonded_atoms()

        self.can_form_hbond = np.append(list(self.acceptor.keys()), list(self.donor.keys())).astype(int)

        self.hbond_out = {}
        self.water_mediated_out = {}

    @staticmethod
    def dist(u, v):
        """
        Return euclidean distance
        :param u:
        :param v:
        :return:
        """
        return np.sqrt(np.sum((u - v) ** 2))

    @staticmethod
    def angle(r0, r1, r2):
        """
        Returns the angle r1,r0,r2
        r0 == angle vertex.
        :param r0:
        :param r1:
        :param r2:
        :return:
        """
        v1 = []
        v2 = []
        for i in range(len(r0)):
            v1.append(r1[i] - r0[i])
            v2.append(r2[i] - r0[i])
        magnitude_r1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        magnitude_r2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
        radians = np.arccos((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (magnitude_r1 * magnitude_r2))
        degrees = np.rad2deg(radians)
        return degrees

    def _set_donor_acceptor(self):
        """
        Get donor(H); Acceptor & bonded atoms
        """

        for a in self.cms_model.atom:
            # If a == hydrogen get heavy atom
            if a.element == 'H':
                da = a.bond[1].atom2
                # NOTE this class returns true for the donor heavy atom
                if mm.mmct_hbond_is_donor(self.cms_model, da.index):
                    # Convert back to pythonic indices
                    self.donor[a.index - 1] = da.index - 1
            else:
                if mm.mmct_hbond_is_acceptor(self.cms_model, a.index):
                    # If a == acceptor get list of bonded atoms
                    acc_att_list = []
                    for acceptor_neighbor in a.bonded_atoms:
                        if acceptor_neighbor.atom_type != 63:  # If not lone pair
                            acc_att_list.append(acceptor_neighbor.index - 1)
                    self.acceptor[a.index - 1] = acc_att_list

    def _set_bonded_atoms(self):
        """
        Set bond lookup dir
        :return:
        """
        for i, a in enumerate(self.cms_model.atom):
            self.bonded_atoms[i] = topo.aids2gids(self.cms_model, [ab.index for ab in a.bonded_atoms])

    def match_hbond(self, n1, n2, p):
        """
        Check whether atomic indices n1 & n2 for a hydrogen bond
        :param n1:
        :param n2:
        :param p:
        :return:
        """
        # Check wether n1::n2 are a donor-acceptor pair and get bonded atoms
        if n1 in self.acceptor:
            acceptor = n1
            aa = self.acceptor[n1]
        elif n2 in self.acceptor:
            acceptor = n2
            aa = self.acceptor[n2]
        else:
            return False

        if n1 in self.donor:
            donor = n1
            dd = self.donor[n1]
        elif n2 in self.donor:
            donor = n2
            dd = self.donor[n2]
        else:
            return False

        # measure D-H::A distance
        dist = self.dist(p[n1], p[n2])
        if dist > self.dmax:
            return False
        # measure D-H::A angle
        dangle = self.angle(p[donor], p[dd], p[acceptor])
        if dangle < self.donor_angle:
            return False
        # measure -H::A-AA angles
        for a in aa:
            aangle = self.angle(p[acceptor], p[donor], p[a])
            if aangle < self.acceptor_angle:
                return False
        # Congratulation, you are indeed a hydrogen bond
        return True

    def water_mediated_hbond(self, i, f, n1, n2, dist_tree):
        """
        Get water mediated hydrogen bonds
        :param i:
        :param f:
        :param n1:
        :param n2:
        :param dist_tree:
        :return:
        """
        # Get atomic coordinates
        pos = f.pos()
        # NOTE get pythonic indices water res
        r2_atoms = self.bonded_atoms[n2] + [n2]
        # Give some buffer to dmax
        dmax = self.dmax + 0.2
        for a2 in r2_atoms:
            ndx_nn = self.can_form_hbond[dist_tree.query_ball_point(pos[a2], dmax)]
            for n3 in ndx_nn:
                # NOTE skip ignore n1,self_res
                if n3 in [n1] + r2_atoms:
                    continue
                # Only check water mediated hydrogen bonds with heavy atoms
                if n3 not in self.ndx_water:
                    if self.match_hbond(a2, n3, pos):
                        # If interaction has been observed previously set frame f to 1
                        if (n1, n3) in self.water_mediated_out:
                            self.water_mediated_out[(n1, n3)][i] = 1.
                        elif (n3, n1) in self.water_mediated_out:
                            # Check in the opposite direction (shouldn't happen but better be safe)
                            self.water_mediated_out[(n3, n1)][i] = 1.
                        # If interaction hasn't been observed yet initialize new pairwise interactions
                        else:
                            self.water_mediated_out[(n1, n3)] = np.zeros(len(self.frame_list))
                            self.water_mediated_out[(n1, n3)][i] = 1.

    def run(self):
        """
        DocString
        :return:
        """
        for i, f in enumerate(self.frame_list):
            # print 'processing frame', f
            # Get atomic coordinates
            pos = f.pos()

            # Construct Kdist tree for donor acceptor list
            dist_tree = sp.spatial.cKDTree(pos[self.can_form_hbond], leafsize=100)

            # Give some buffer to dmax
            dmax = self.dmax + 0.2
            for n, (n1, p1) in enumerate(zip(self.can_form_hbond, pos[self.can_form_hbond])):

                # check if solute atom
                if n1 not in self.ndx:
                    continue
                # query kdist tree for nearest neighbour atoms
                nn_atoms = dist_tree.query_ball_point(p1, dmax)
                # Skip if only self
                if len(nn_atoms) <= 1:
                    continue
                ndx_nn = self.can_form_hbond[nn_atoms]
                for n2 in ndx_nn:
                    if n1 == n2 or n2 in self.bonded_atoms[n1]:
                        continue
                    if self.match_hbond(n1, n2, pos):
                        # if n2 not solute check for water mediated hbonds
                        if n2 in self.ndx_water:
                            self.water_mediated_hbond(i, f, n1, n2, dist_tree)
                        # for direct interactions add to frame results
                        else:
                            # If interaction has been pbserved previously set frame f to 1
                            if (n1, n2) in self.hbond_out:
                                self.hbond_out[(n1, n2)][i] = 1.
                            elif (n2, n1) in self.hbond_out:
                                # Check in the opposite direction (shouldn't happen but better be safe)
                                self.hbond_out[(n2, n1)][i] = 1.
                            # If interaction hasn't been observed yet initialize new pairwise interactions
                            else:
                                self.hbond_out[(n1, n2)] = np.zeros(len(self.frame_list))
                                self.hbond_out[(n1, n2)][i] = 1.
        self.queue.put([self._id, self.hbond_out, self.water_mediated_out])


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


def get_error(data, nproc):
    """
    DocString
    :param data:
    :param nproc:
    :return:
    """
    stddev = np.zeros(data.shape[0])

    in_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
    workers = [BlockAverage(in_queue, out_queue, min_blocks=20) for _ in range(nproc)]
    for w in workers:
        w.start()

    for i, row in enumerate(data):
        in_queue.put([i, row])

    for i in range(data.shape[0]):
        _id, error = out_queue.get()
        stddev[_id] = error

    for _ in range(nproc):
        in_queue.put('STOP')

    for w in workers:
        w.join()

    return stddev


def get_results(cms_model, frame_results, calculate_error=True, frequency_cutoff=0.1, is_water_mediated=False):
    """
    :param cms_model:
    :param frame_results:
    :param calculate_error:
    :param frequency_cutoff:
    :param is_water_mediated:
    :return df: Mean hydrogen bonds
    :rtype df: pd.DataFrame
    :return data_raw: Raw hydrogen bond data
    rtype data_raw: pd.DataFrame
    """

    atom_pair_id = []
    data_raw = []
    frequencies = []

    atom_dict = dict([(a.index, a) for a in cms_model.atom])

    # Make sure that there are actually results to return
    try:
        atom_pairs = frame_results.keys()
    except Exception as e:
        raise RuntimeError('There are no results to return! \n{}'.format(e))

    for p in atom_pairs:
        frequency = np.mean(frame_results[p])
        if frequency >= frequency_cutoff:
            data_raw.append(frame_results[p])
            frequencies.append(frequency)
            atom_pair_id.append([p[0] + 1, p[1] + 1])  # Convert from pythonic to atomic

    data_raw = np.asarray(data_raw)

    frequencies = np.asarray(frequencies).reshape(-1, 1)
    water_mediated = np.asarray([is_water_mediated] * len(frequencies)).reshape(-1, 1)

    if calculate_error:
        t = time.time()
        logger.info('Calculating standard deviation for {} hydrogen bonds'.format(frequencies.shape[0]))
        stddev = get_error(data_raw, nproc=dynamic_cpu_assignment(NPROC)).reshape(-1, 1)
        logger.info('Calculated hydrogen bond standard deviation in {:.0f} seconds'.format(time.time() - t))
    else:
        stddev = np.asarray([np.nan] * len(frequencies)).reshape(-1, 1)

    df = pd.DataFrame(np.hstack((frequencies, stddev, water_mediated)),
                      columns=['frequency', '$\\sigma$', 'water_mediated'])
    for i, (aid1, aid2) in enumerate(atom_pair_id):
        a1 = atom_dict[aid1]
        a2 = atom_dict[aid2]
        df.loc[i, 'atom index 1'] = aid1
        df.loc[i, 'atom index 2'] = aid2
        df.loc[i, 'chain 1'] = a1.chain.strip()
        df.loc[i, 'chain 2'] = a2.chain.strip()
        df.loc[i, 'resnum 1'] = a1.resnum
        df.loc[i, 'resnum 2'] = a2.resnum
        df.loc[i, 'resname 1'] = a1.pdbres.strip()
        df.loc[i, 'resname 2'] = a2.pdbres.strip()
        df.loc[i, 'atomname 1'] = a1.pdbname.strip()
        df.loc[i, 'atomname 2'] = a2.pdbname.strip()
    return df, data_raw


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
            del(structure_dict['custom']['pipeline'])
        else:
            structure_dict['custom']['pipeline'] = pipeline[1:]

    structure_code = structure_dict['structure']['code']
    outfile = '{}_trj_hbond.csv'.format(structure_code)
    outfile_raw = '{}_trj_hbonds_raw.tar.gz'.format(structure_code)

    # Load simulation files
    cms_file = structure_dict['files']['desmond_cms']

    msys_model, cms_model = topo.read_cms(str(cms_file))
    if LIGAND_ASL is None:
        logger.info('Calculating all intra- and intermolecular hydrogen bonds')
        ligand_ndx=None
    else:
        logger.info('Calculating hydrogen bonds between system and: {}'.format(LIGAND_ASL))
        ligand_ndx = topo.asl2gids(cms_model, LIGAND_ASL)
    logger.info('Unpacking trajectory frame set')
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

    combined_results = {}
    combined_water_results = {}

    nproc = dynamic_cpu_assignment(NPROC)
    t = time.time()
    logger.info('Calculating hydrogen bonds using {} workers'.format(nproc))

    frame_list = np.array_split(np.arange(0, len(traj.read_traj(str(trj_dir))), STEP, dtype=int), nproc)
    total_frames = sum(map(len, frame_list))

    # Start worker processes
    workers = []
    queue = multiprocessing.Queue()
    for i, frames in enumerate(frame_list):
        workers.append(HydrongeBondAnalysis(i, queue, cms_file, trj_dir, frames=frames, ndx=ligand_ndx))
        workers[i].start()

    # get results
    for i in range(nproc):
        _id, frame_results, water_frame_results = queue.get()
        for k in frame_results.keys():
            if k not in combined_results:
                combined_results[k] = np.zeros(total_frames)
                combined_results[k][frame_list[_id] // STEP] = frame_results[k].tolist()
            else:
                combined_results[k][frame_list[_id] // STEP] = frame_results[k].tolist()
        for k in water_frame_results:
            if k not in combined_water_results:
                combined_water_results[k] = np.zeros(total_frames)
                combined_water_results[k][frame_list[_id] // STEP] = water_frame_results[k].tolist()
            else:
                combined_water_results[k][frame_list[_id] // STEP] = water_frame_results[k].tolist()
    for w in workers:
        w.join()
    logger.info('Calculated hydrogen bonds in {:.0f} seconds'.format(time.time() - t))
    # Get mean hydrogen bonds and errors
    hbonds, hbonds_raw = get_results(cms_model, combined_results, is_water_mediated=False)
    water_mediated, water_mediated_raw = get_results(cms_model, combined_water_results, calculate_error=False,
                                                     is_water_mediated=True)
    mean_df = pd.concat([hbonds, water_mediated], ignore_index=False)
    mean_df.index = np.arange(mean_df.shape[0])
    mean_df.to_csv(outfile, sep=',')
    if not water_mediated_raw.shape[0]:
        raw_df = pd.concat((mean_df, pd.DataFrame(hbonds_raw)), axis=1)
    else:
        raw_df = pd.concat((mean_df, pd.DataFrame(np.vstack((hbonds_raw, water_mediated_raw)))), axis=1)
    raw_df.to_csv('trj_hbonds_raw.csv', sep=',')
    with tarfile.open(outfile_raw, 'w:gz') as tar:
        tar.add('trj_hbonds_raw.csv')

    transformer_dict = {
        'structure': {
            'parent_structure_id':
                structure_dict['structure']['structure_id'],
            'searchable': False
        },
        'files': {'trj_hbonds': outfile,
                  'trj_hbonds_raw': outfile_raw},
        'custom': structure_dict['custom']
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
        Calculate protein/ligand inter- and intramolecular hydrogen bond frequency.\n
        For eligible atomtypes hydrogen-bonds are defined by geometric criteria:
        
        H = hydrogen
        A = acceptor
        AA = atom bonded to acceptor
        D = atom bonded to hydrogen
        :: = potential hbond
        - = covalent bond
        
        1. the H::A distance must be less than or equal to 3.0 Angstrom.
        2. the D-H::A angle must be at least 110 degree.
        3. the H::A-AA angle must be at least 90 degree.
        
        Hydrogenbond frequency is calculated both for inter and intramolecular hydrogenbonds, error is estimated using 
        block averaging. The frequency of water mediated hydrogen bonds is also calculated. No error are calculated for
        water mediated hydrogen bonds, because not only can water mediated hydrogen bonds can exist in multiple unique 
        states but at each point in time there can potentialy be multiple water mediated hydrogen bonds between a pair
        of solute heavy atoms. Results are returned in a csv file. 
        
        '''
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('infiles',
                        type=str,
                        nargs='+',
                        help='Simulation cmsfile and trj')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='test',
                        help='Outfile prefix')
    parser.add_argument('-n',
                        '--nproc',
                        type=int,
                        dest='nproc',
                        default=16,
                        help='Number of cores to use for calculation.\nDefault: 16')
    parser.add_argument('-s',
                        '--step',
                        type=int,
                        dest='step',
                        default=2,
                        help='Process every X steps of the trajectory.\nDefault: 2 ')
    parser.add_argument('-l',
                        '--ligand_asl',
                        type=str,
                        dest='ligand_asl',
                        default=None,
                        help='Atom selection string specifying the ligand atoms. If provided only \
                             hydrogen bonds with the atoms identified as belonging to the ligand \
                              will be considered')

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
    global STEP
    global LIGAND_ASL

    NPROC = args.nproc
    STEP = args.step
    LIGAND_ASL = args.ligand_asl

    cms_file, trj = args.infiles
    prefix = args.prefix

    structure_dict_list = [
        {'structure': {'structure_id': 0, 'code': prefix},
         'files': {'desmond_cms': cms_file, 'desmond_trjtar': trj},
         'custom': []}]
    out_dict = [nsd for nsd in run(structure_dict_list)]
    with open('{}_trj_hbond_transformer.json'.format(prefix), 'w') as fout:
        json.dump(out_dict, fout)


if __name__ == '__main__':
    import argparse

    args = parse_args()
    logger = get_logger()
    main(args)
