from __future__ import print_function, division

import os
import json
import tarfile
import multiprocessing
import copy

import numpy as np

from schrodinger import structure
from schrodinger.structutils.analyze import find_ligands
import schrodinger.application.desmond.packages.topo as topo
import schrodinger.application.desmond.packages.traj as traj

import logging

logger = logging.getLogger(__name__)

NPROC = 3

CALCULATION_PARAM = [
    {'name': 'calpha rmsd', 'ref': None, 'align_asl': 'a. CA', 'calc_asl': None, 'calculation_type': 'rmsd'},
    {'name': 'calpha rmsf', 'ref': None, 'align_asl': 'a. CA', 'calc_asl': None, 'calculation_type': 'rmsf'}]


class Superimposer:
    """
    Python implementation of the Kabsch algorithm for structural alignment
    """

    def __init__(self):
        self._rot = None
        self._tran = None

    def fit(self, reference_coords, coords):
        if coords is None or reference_coords is None:
            raise Exception("Invalid coordinates set.")

        n = reference_coords.shape
        m = coords.shape
        if n != m or not (n[1] == m[1] == 3):
            raise Exception("Coordinate number/dimension mismatch.")

        self._calc_rot_tran(reference_coords, coords)

    def _calc_rot_tran(self, reference_coords, coords):
        """
        Superimpose the coordinate sets.
        :param reference_coords:
        :param coords:
        :return:
        """

        # center on centroid
        self.c1 = np.mean(coords, axis=0)
        self.c2 = np.mean(reference_coords, axis=0)

        coords = coords - self.c1
        reference_coords = reference_coords - self.c2

        # correlation matrix
        a = np.dot(np.transpose(coords), reference_coords)

        u, d, vt = np.linalg.svd(a)

        self._rot = np.dot(u, vt)

        # check if we have found a reflection
        if np.linalg.det(self._rot) < 0:
            vt[2] = -vt[2]
            self._rot = np.dot(u, vt)
        self._tran = self.c2 - np.dot(self.c1, self._rot)

    def get_rot_tran(self):
        """
        Return rotation matrix and translation vector.
        :return:
        """
        if self._rot is None:
            raise Exception("Nothing superimposed yet.")
        return self._rot, self._tran

    def transform(self, coords):
        """
        Apply rotation and translation matrix to  a set of coordinates
        :param coords:
        :return:
        """
        if self._rot is None:
            raise Exception("Nothing superimposed yet.")

        return np.dot(coords, self._rot) + self._tran

    def fit_transform(self, reference_coords, coords):
        """
        Calculate rotation and translation matrix and apply it to the reference coordinates
        :param reference_coords:
        :param coords:
        :return:
        """
        if coords is None or reference_coords is None:
            raise Exception("Invalid coordinates set.")

        n = reference_coords.shape
        m = coords.shape
        if n != m or not (n[1] == m[1] == 3):
            raise Exception("Coordinate number/dimension mismatch.")

        self._calc_rot_tran(reference_coords, coords)
        return np.dot(coords, self._rot) + self._tran


class RMS(multiprocessing.Process):
    """
    Docstring
    """

    def __init__(self, cms_file, trj_dir, queue, frames=None, params=None):
        """
        Docstring
        :param cms_file:
        :param trj_dir:
        :param frames:
        :param params:
        """
        multiprocessing.Process.__init__(self)

        self.queue = queue

        # load cms_model
        self.msys_model, self.cms_model = topo.read_cms(str(cms_file))

        # Get framelist
        if frames is not None:
            self.frame_list = [frame for (i, frame) in enumerate(traj.read_traj(str(trj_dir))) if i in frames]
        else:
            self.frame_list = traj.read_traj(str(trj_dir))

        self.total_frames = len(self.frame_list)

        self.align = Superimposer()

        # Calculation parameters
        self.calculation_parameters = []

        if params is not None:
            for param in params:
                self.add_param(**param)

    def add_param(self, name='', ref=None, align_asl='a. CA', calc_asl=None, calculation_type='rmsd'):
        """
        Docstring
        :param name:
        :param ref:
        :param align_asl:
        :param calc_asl:
        :param calculation_type:
        :return:
        """

        if calc_asl is None:
            calc_asl = align_asl

        if ref is None:
            ref_align = self.cms_model.getXYZ()[topo.asl2gids(self.cms_model, asl=align_asl, include_pseudoatoms=False)]
            ref_calc = self.cms_model.getXYZ()[topo.asl2gids(self.cms_model, asl=calc_asl, include_pseudoatoms=False)]
        else:
            ref_align = ref.getXYZ()[topo.asl2gids(ref, asl=align_asl, include_pseudoatoms=False)]
            ref_calc = ref.getXYZ()[topo.asl2gids(ref, asl=calc_asl, include_pseudoatoms=False)]

        mobile_align_ndx = topo.asl2gids(self.cms_model, asl=align_asl, include_pseudoatoms=False)
        mobile_calc_ndx = topo.asl2gids(self.cms_model, asl=calc_asl, include_pseudoatoms=False)

        # Check whether reference and mobile selctions have the same size
        if any([len(mobile_align_ndx) != len(ref_align), len(mobile_calc_ndx) != len(ref_calc)]):
            raise RuntimeWarning('Reference and Mobile atoms missmatch')
        else:
            self.calculation_parameters.append({'name': name, 'ref_align': ref_align, 'ref_calc': ref_calc,
                                                'mobile_align_ndx': mobile_align_ndx,
                                                'mobile_calc_ndx': mobile_calc_ndx,
                                                'mobile_calc': np.zeros((self.total_frames, len(mobile_calc_ndx), 3)),
                                                'type': calculation_type})

    @staticmethod
    def rmsd(reference, mobile):
        """
        Calculate root mean squared deviation
        :param reference: <numpy.ndarray> of reference atoms, shape: (natoms,3)
        :param mobile: <numpy.ndarray> of mobile atoms, shape: (nframes,natoms,3)
        :return:
        """
        rmsd_out = []
        for mobile_frame in mobile:
            rmsd_out.append(np.sqrt(np.mean((reference - mobile_frame) ** 2)))
        return rmsd_out

    @staticmethod
    def rmsf(reference, mobile):
        """
        calculate root mean squared fluctuation
        :param reference: <numpy.ndarray> of reference atoms, shape: (natoms,3)
        :param mobile: <numpy.ndarray> of mobile atoms, shape: (nframes,natoms,3)
        :return:
        """
        natoms = mobile.shape[1]
        rmsf_out = []
        for i in range(natoms):
            rmsf_out.append(np.sqrt(np.mean((mobile[::, i] - reference[i]) ** 2)))
        return rmsf_out

    def run(self):
        """
        Calculate Root mean square deviation and root mean square fluctuation
        """
        for i, frame in enumerate(self.frame_list):
            frame_coordinates = frame.pos()
            for param_dict in self.calculation_parameters:
                self.align.fit(param_dict['ref_align'], frame_coordinates[param_dict['mobile_align_ndx']])
                param_dict['mobile_calc'][i] = self.align.transform(frame_coordinates[param_dict['mobile_calc_ndx']])

        for param_dict in self.calculation_parameters:
            if param_dict['type'] == 'rmsd':
                self.queue.put({'type': param_dict['type'],
                                'name': param_dict['name'],
                                'results': self.rmsd(param_dict['ref_calc'], param_dict['mobile_calc'])})
            elif param_dict['type'] == 'rmsf':
                atom_ids = [(a.chain.strip(), a.resnum, a.pdbname.strip()) for (i, a) in enumerate(self.cms_model.atom)
                            if i in param_dict['mobile_calc_ndx']]
                self.queue.put({'type': param_dict['type'],
                                'name': param_dict['name'],
                                'results': self.rmsf(param_dict['ref_calc'], param_dict['mobile_calc']),
                                'atom_ids': atom_ids})


def _process(structure_dict):
    """
    Docstring
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

    outname = '{}_trj_rms'.format(structure_dict['structure']['code'])
    outfile = outname + '.json'
    results = []
    # Load simulation files
    cms_file = structure_dict['files']['desmond_cms']
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

    # Check whether a ligand exists and add ligand rmsd to set of calculations
    calculation_param = copy.copy(CALCULATION_PARAM)
    ligand_mae = structure_dict['files'].get('ligand_mae')
    if ligand_mae is None:
        msys_model, cms_model = topo.read_cms(str(cms_file))
        ligands = find_ligands(cms_model)
        if len(ligands) != 0:
            for ligand in ligands:
                ligand_asl = ' or '.join(['(r. {} and c. "{}" and not a.element H )'.format(res.resnum, res.chain)
                                          for res in ligand.st.residue])
                calculation_param.append({'name': 'ligand_rmsf',
                                          'ref': None,
                                          'align_asl': ligand_asl,
                                          'calc_asl': None,
                                          'calculation_type': 'rmsf'})
    else:
        ligand_st = structure.Structure.read(str(ligand_mae))
        ligand_asl = '( ' + ' or '.join(['(r.ptype {} and c. "{}")'.format(res.pdbres, res.chain) for res in
                                         ligand_st.residue]) + ' ) and not a.element H'
        calculation_param.append({'name': 'ligand_rmsf',
                                  'ref': None,
                                  'align_asl': ligand_asl,
                                  'calc_asl': None,
                                  'calculation_type': 'rmsf'})

    out_queue = multiprocessing.Queue()
    if NPROC != len(calculation_param):
        nproc = len(calculation_param)
    else:
        nproc = NPROC
    logger.info('Performing {} rms calculations, using {} cores.'.format(len(calculation_param), nproc))

    calculation_param = np.array_split(calculation_param, nproc)
    workers = []
    for i, params in enumerate(calculation_param):
        workers.append(RMS(cms_file, trj_dir, out_queue, params=params))
        logger.info('starting subprocess {}'.format(i))
        workers[-1].start()
    for i in range(nproc):
        results.append(out_queue.get())
    for w in workers:
        w.join()
    with open(outfile, 'w') as f:
        json.dump(results, f)

    transformer_dict = {
        'structure': {
            'parent_structure_id': structure_dict['structure']['structure_id']
        },
        'files': {'trj_rms': outfile},
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
    Calculate protein rmsd and protein/ligand rmsf.
    Results are returned in json dictionary.
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infiles',
                        type=str,
                        nargs='+',
                        help='cms file, trajectory and optional ligand structure')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='similarity_search',
                        help='Outfile prefix')
    parser.add_argument('-n',
                        '--nproc',
                        type=int,
                        dest='nproc',
                        default=3,
                        help='Number of cores to use for calculation.\nDefault: 3')
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


def main():

    global NPROC

    args = parse_args()

    NPROC = args.nproc
    prefix = args.prefix
    if len(args.infiles) == 2:
        cmsfile, trjtar = args.infiles
        ligand_mae = None
    else:
        cmsfile, trjtar, ligand_mae = args.infiles

    structure_dict_list = [{'structure': {'code': prefix,
                                          'structure_id': 0},
                            'files': {'desmond_cms': cmsfile,
                                      'desmond_trjtar': trjtar,
                                      'ligand_mae': ligand_mae},
                            'custom': {}
                            }]
    outp = [sd for sd in run(structure_dict_list)]

    with open('{}_trj_rms_transformer.json'.format(prefix), 'w') as fout:
        json.dump(outp, fout)


if __name__ == '__main__':
    import argparse
    logger = get_logger()
    main()
