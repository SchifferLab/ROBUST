from __future__ import division, print_function

import os

import json
import tarfile
import time
import functools

import pandas as pd
import numpy as np

from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.application.desmond.packages import topo
from schrodinger.application.desmond.packages import traj

from schrodinger.application.desmond.packages.analysis import analyze
from schrodinger.application.desmond.packages.analysis import Torsion

import logging

logger = logging.getLogger(__name__)
NPROC = 0.25  # Use X% of the available cores
STEP = 1  # Use every X frames

ELEMENT_PRIORITY = {'C': 0, 'O': 1, 'N': 1, 'F': 1, 'S': 2, 'P': 2, 'Cl': 2, 'Br': 2}

BACKBONE = {'PHI': [(-1, 'C'), (0, 'N'), (0, 'CA'), (0, 'C')],
            'PSI': [(0, 'N'), (0, 'CA'), (0, 'C'), (1, 'N')]}

CHI_DEFINITIONS = {'A': [],
                   'G': [],
                   'V': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG1')], ],
                   'L': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD1')]],
                   'I': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG1')], [(0, 'CA'), (0, 'CB'), (0, 'CG1'), (0, 'CD1')]],
                   'P': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], ],
                   'F': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD1')]],
                   'Y': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD1')]],
                   'W': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD1')]],
                   'M': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'SD')],
                         [(0, 'CB'), (0, 'CG'), (0, 'SD'), (0, 'CE')]],
                   'C': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'SG')], ],
                   'S': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'OG')], ],
                   'T': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'OG1')]],
                   'D': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'OD1')]],
                   'E': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD')],
                         [(0, 'CB'), (0, 'CG'), (0, 'CD'), (0, 'OE1')]],
                   'H': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'ND1')]],
                   'K': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD')],
                         [(0, 'CB'), (0, 'CG'), (0, 'CD'), (0, 'CE')], [(0, 'CG'), (0, 'CD'), (0, 'CE'), (0, 'NZ')]],
                   'R': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD')],
                         [(0, 'CB'), (0, 'CG'), (0, 'CD'), (0, 'NE')], [(0, 'CG'), (0, 'CD'), (0, 'NE'), (0, 'CZ')]],
                   'N': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'OD1')]],
                   'Q': [[(0, 'N'), (0, 'CA'), (0, 'CB'), (0, 'CG')], [(0, 'CA'), (0, 'CB'), (0, 'CG'), (0, 'CD')],
                         [(0, 'CB'), (0, 'CG'), (0, 'CD'), (0, 'OE1')]]}



def set_original_atom_index(st):
    """
    Set orig_atom_index property
    :param st:
    :type st: schrodinger.structure.Structure
    :return:
    """
    for atm in st.atom:
        atm.property['i_m_orig_atom_index'] = atm.index
    return st


def get_original_atom_index(a):
    oaid = a.property.get('i_m_original_index')
    if oaid is None:
        raise ValueError('Original atom index not defined')
    else:
        return oaid


def print_iframe(i, fr, tr, n=1000, logger=None):
    if logger is None:
        if not i % n:
            print('Processed {} frames'.format(i))
    else:
        if not i % n:
            logger.info('Processed {} frames'.format(i))


def get_adjacent_atom(a, exclusion_list=None, element_priority=None):
    """
    Return the heavy atom covalently bound to atom a
    If atom a is making covalent bonds to more than one heavy atom return the atom with the highest priority.
    The priority rules are:
    1. Bond order
    2. Element type
    2.1 Unique element (i.e. only one atom of type E is bonded to atom a)
    2.2 Element priority (If element priority is undefined alphabetical order will be used)
    :param a:
    :param exclusion_list:
    :param element_priority:
    :return:
    """

    def get_priority(e):
        priority = element_priority.get(e)
        if priority is None:
            return 0
        else:
            return priority

    adjacent_atoms = []
    for aa in a.bonded_atoms:
        if exclusion_list is not None and aa.index in exclusion_list:
            continue
        elif a.element == 'H':
            continue
        else:
            adjacent_atoms.append(aa)
    # Return None if all adjacent atoms are hydrogens
    if len(adjacent_atoms) == 0:
        return
    # If there is only one adjacent atom return it
    elif len(adjacent_atoms) == 1:
        return adjacent_atoms[0]
    # If there are 2 bonded non hydrogen atoms
    elif len(adjacent_atoms) == 2:
        # Check if bond order is different
        bond_order = [a._ct.getBond(a, aa).order for aa in adjacent_atoms]
        # if bond order is invariant check element types
        if len(set(bond_order)) == 1:
            elements = [aa.element for aa in adjacent_atoms]
            # If element types are invariant return the first atom in the list
            if len(set(elements)) == 1:
                return adjacent_atoms[0]
            # Else return the first atom based on element priority
            else:
                if element_priority is not None and any([e in element_priority for e in elements]):
                    return sorted([(get_priority(e), aa) for e, aa in zip(elements, adjacent_atoms)],
                                  key=lambda x: x[0], reverse=True)[0][1]
                else:
                    return sorted(list(zip(elements, adjacent_atoms)), key=lambda x: x[0])[0][1]
        # Else return the atom with the highest bond order
        else:
            return sorted(list(zip(bond_order, adjacent_atoms)), key=lambda x: x[0], reverse=True)[0][1]
    # If there are 3 bonded non hydrogen atoms
    else:
        elements = [aa.element for aa in adjacent_atoms]
        # If all 3 atoms are of different element types return the first in alphabetical order
        if len(set(elements)) == 3:
            if element_priority is not None and any([e in element_priority for e in elements]):
                return sorted([(get_priority(e), aa) for e, aa in zip(elements, adjacent_atoms)],
                              key=lambda x: x[0], reverse=True)[0][1]
            else:
                return sorted(list(zip(elements, adjacent_atoms)), key=lambda x: x[0])[0][1]
        # If one atom is of a different element type than the others return that atom
        elif len(set(elements)) == 2:
            for aa, element in zip(adjacent_atoms, elements):
                if elements.count(element) == 1:
                    return aa
        # If all 3 atoms are off the same type return the first atom in the list
        else:
            return adjacent_atoms[0]


def get_hetero_torsion_atoms(st, element_priority=None):
    torsion_list = []
    rings = []
    for ring in st.ring:
        rings.append([a.index for a in ring.atom])

    for b in st.bond:
        a1 = b.atom1
        a2 = b.atom2

        if any([a1.element == 'H', a2.element == 'H']):
            continue

        if any([a1.bond_total < 2, a2.bond_total < 2]):
            continue

        # Check if a1 and a2 are part of the same ring system
        if any([all([a1.index in ring, a2.index in ring]) for ring in rings]):
            continue

        a3 = get_adjacent_atom(a1, exclusion_list=[a2.index, ], element_priority=element_priority)
        a4 = get_adjacent_atom(a2, exclusion_list=[a1.index, ], element_priority=element_priority)

        if any([a3 is None, a4 is None]):
            continue
        else:
            torsion_list.append([a3, a1, a2, a4])
    return torsion_list


def get_protein_torsion_atoms(st, asl='protein', bb=True, chi=True):
    """
    return torsion angles for all residues defined by asl.
    By default only PHI and PSI will be returned.
    Optional a list of Chi angles can be provided.
    !!!CHI list must start from 0!!!
    PHI and PSI will not be returned if no_bb==True
    :param st:
    :param asl:
    :param chi:
    :param bb:
    :return:
    """

    def get_atm(residue, pdbname):
        """
        Return pythonic index of the first atom in residue that has the pdb atom name atm_pdbname
        Raise RuntimeError if no atom was found.
        :param residue:
        :param pdbname:
        :return:
        """
        for atm in residue.atom:
            if atm.pdbname.strip() == pdbname:  # check whether atomname equals pdbname
                return atm

    torsion_list = []
    # sanity checks
    if not chi and not bb:
        raise RuntimeError("Not torsion angles specified")

    atom_ids = evaluate_asl(st, asl)  # get atom ids
    reslist = [r for r in st.residue]  # get list of residue objects
    for i, r0 in enumerate(reslist):
        if r0.isStandardResidue() and r0.getCode().strip() in CHI_DEFINITIONS.keys():
            if any([aid.index in atom_ids for aid in r0.atom]):  # check whether asl includes r0
                if bb:  # fetch phi and psi angles
                    bb_topo = []  # determine backbone topology
                    if i - 1 >= 0:
                        if reslist[i - 1].isStandardResidue() and reslist[i - 1].isConnectedToResidue(
                                r0):  # if r-1 is a standard residue connected to r0 store phi
                            bb_topo.append('PHI')
                    if i + 1 < len(reslist):
                        if reslist[i + 1].isStandardResidue() and r0.isConnectedToResidue(
                                reslist[i + 1]):  # if r+1 is standard residue connected to r0 store psi
                            bb_topo.append('PSI')
                    if not bb_topo:  # sanity check
                        raise RuntimeError(
                            'Unable to determine backbone topology for residue {}:{}'.format(r0.resnum, r0.chain))
                    for angle in bb_topo:
                        torsion = []
                        for res_pos, atm_name in BACKBONE[angle]:
                            torsion.append(get_atm(reslist[i + res_pos], atm_name))
                        torsion_list.append(torsion)
                if chi:  # if chi_angle list is provided
                    r0_pdbcode = r0.getCode().strip()  # get oneletter code
                    for chii in range(4):  # At most there can be four chi torsion angles
                        if len(CHI_DEFINITIONS[r0_pdbcode]) > chii:  # check wether r0 has a chiX angle
                            torsion = []
                            for res_pos, atm_name in CHI_DEFINITIONS[r0_pdbcode][chii]:
                                torsion.append(get_atm(reslist[i + res_pos], atm_name))
                            torsion_list.append(torsion)
    return torsion_list


def _process(structure_dict):
    """

    :param structure_dict:
    :return:
    """
    t = time.time()

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

    logger.info('Load simulation files')

    cms_file = structure_dict['files']['desmond_cms']
    msys_model, cms_model = topo.read_cms(str(cms_file))

    trjtar = structure_dict['files']['desmond_trjtar']

    # If run from command line it does not make sense to provide a tarfile
    if os.path.isdir(trjtar):
        trj_dir = trjtar
    elif tarfile.is_tarfile(trjtar):
        with tarfile.open(name=trjtar, mode='r:gz') as tfile:
            tfile.extractall()
            trj_dir = tfile.getnames()[0]
    else:
        raise RuntimeError('trjtar is neither a directory nor a tarfile')

    frame_list = traj.read_traj(str(trj_dir))
    frame_list = [frame_list[i] for i in range(0, len(frame_list), STEP)]

    logger.info('Calculating torsion angles')

    cms_model = set_original_atom_index(cms_model)
    ligand_ct = cms_model.extract(evaluate_asl(cms_model, 'ligand'), copy_props=True)

    torsion_list = get_hetero_torsion_atoms(ligand_ct, element_priority=ELEMENT_PRIORITY)
    torsion_list.extend(get_protein_torsion_atoms(cms_model))

    analyzers = []

    torsion_ids = pd.DataFrame(columns=['index', 'aid1', 'aid2', 'aid3', 'aid4'])
    torsion_ids.set_index('index', inplace=True)

    for i, atom_set in enumerate(torsion_list):
        atom_set = list(map(get_original_atom_index, atom_set))
        analyzers.append(Torsion(msys_model, cms_model, *atom_set))
        torsion_ids.loc[i, ['aid1', 'aid2', 'aid3', 'aid4']] = atom_set

    results = analyze(frame_list, *analyzers,
                      **{"progress_feedback": functools.partial(print_iframe, logger=logger)})

    out_arch = '{}_torsion.tar.gz'.format(structure_code)
    with tarfile.open(out_arch, 'w:gz') as tar:
        torsion_ids.to_csv('torsion_ids.csv', sep=',')
        tar.add('torsion_ids.csv')
        for i, timeseries in enumerate(results):
            fname = 'torsion_{}.csv'.format(i)
            np.savetxt(fname, timeseries, delimiter=',')
            tar.add(fname)

    logger.info('Calculated torsion angles in {:.0f} seconds'.format(time.time() - t))
    # Return structure dict
    transformer_dict = {
        'structure': {
            'parent_structure_id':
                structure_dict['structure']['structure_id'],
            'searchable': False
        },
        'files': {'trj_torsion': out_arch},
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
    Record protein and all small organic molecules torsion angles.\n
    Torsion angle time series will be written to a tar archive in separate csv file. 
    Torsion angles cann be identified using torsion_ids.csv.
    Each angle is identified by 4 atom ids where atoms 2 and 3 form the central bond.
    The index in torsion_ids.csv corresponds to the time series: torsion_#.csv. '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infiles',
                        type=str,
                        nargs='+',
                        help='Simulation cmsfile and trj')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='similarity_search',
                        help='Outfile prefix')
    parser.add_argument('-n',
                        '--nproc',
                        type=int,
                        dest='nproc',
                        default=16,
                        help='Number of cores to use for calculation.\nDefault: 16')

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

    cwd = os.getcwd()

    prefix = args.prefix
    desmond_cms, trjtar = args.infiles
    desmond_cms = os.path.abspath(str(desmond_cms))
    trjtar = os.path.abspath(str(trjtar))
    NPROC = args.nproc

    # work in tempdir
    tempdir = tempfile.TemporaryDirectory()
    # makesure that tempdir gets closed properly
    atexit.register(tempdir.cleanup)
    # cd to tempdir
    os.chdir(tempdir.name)

    structure_dict_list = [
        {'structure': {'structure_id': 0, 'code': prefix},
         'files': {'desmond_cms': desmond_cms, 'desmond_trjtar': trjtar},
         'custom': {}}]
    out_dict = [nsd for nsd in run(structure_dict_list)]
    with open(os.path.join(cwd, '{}_trj_torsion.json'.format(prefix)), 'w') as fout:
        json.dump(out_dict, fout)
    # Move outputfiles to cwd
    for structure_dict in out_dict:
        for outfile in structure_dict['files'].values():
            shutil.move(outfile, os.path.join(cwd, outfile.split()[-1]))


if __name__ == '__main__':
    import shutil
    import argparse
    import tempfile
    import atexit

    args = parse_args()
    logger = get_logger()
    main(args)
