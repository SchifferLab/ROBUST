from __future__ import print_function, division

import os
import json
import copy
import tarfile

import pandas as pd
import numpy as np

from schrodinger import structure
from schrodinger.job import jobcontrol
from schrodinger.application.desmond.packages import topo, traj
from schrodinger.structutils.analyze import find_ligands, evaluate_asl


import logging

logger = logging.getLogger(__name__)

NPROC = 12  # Split into at most X subjobs
NFRAMES = 100  # Number of frames use for mmgbsa calculation
LIGAND_ASL = None  # Set to overwrite automatic ligand detection

MMGBSA_CMD = 'prime_mmgbsa'

MMGBSA_OPTIONS = {'-report_prime_log': 'yes',
                  '-csv_output': 'yes',
                  '-jobname': 'trj_mmgbsa',
                  '-ligand': 'ligand',
                  '-out_type': 'COMPLEX',
                  '-job_type': 'ENERGY',
                  '-NJOBS': '1'}


def get_ligand_asl(structure_dict):
    """
    Get ligand asl from ligand_mae if available or base_mae.
    Will raise a ValueError if no ligand is found
    :param structure_dict:
    :return:
    """

    ligand_mae = structure_dict['files'].get('ligand_mae')


    if ligand_mae is None:
        cms = str(structure_dict['files'].get('desmond_cms'))
        st = structure.Structure.read(cms)
        ligands = find_ligands(st)
        if len(ligands) != 0:
            for ligand in ligands:
                ligand_asl = ' or '.join(['(r. {} and c. {})'.format(res.resnum, res.chain)
                                          for res in ligand.st.residue])
        else:
            raise ValueError('No ligand found')
    else:
        ligand_st = structure.Structure.read(str(ligand_mae))
        ligand_asl = '( ' + ' or '.join(['(r.ptype {} and c. {})'.format(res.pdbres, res.chain) for res in
                                         ligand_st.residue]) + ' )'
    return ligand_asl


def extract_st(cms_file, trj_dir, asl='all', frames=1):
    """
    Extract N frames from the trajectory and save them in structure.Structure

    :param cms_file:
    :param trj_dir:
    :param asl:
    :param frames: if int: number of frames uniformly distributed, if iterable: list fo frames
    :return structures: list of schrodinger.structure.Structure
    :type structures: list
    """
    structures = []
    msys_model, cms_model = topo.read_cms(cms_file)
    frame_arch = traj.read_traj(trj_dir)
    if type(frames) == int:
        for f in np.linspace(0, len(frame_arch)-1, frames).astype(int):
            st = topo.update_cms(cms_model, frame_arch[f])
            st = st.extract(evaluate_asl(cms_model,asl))
            st.title = 'Frame {}'.format(f)
            structures.append(st)
    else:
        try:
            for f in frames:
                st = topo.update_cms(cms_model, frame_arch[f])
                st = st.extract(asl)
                st.title = 'Frame {}'.format(f)
                structures.append(st)
        except Exception as e:
            raise RuntimeError(e)

    return structures


def run_mmgbsa(structures, ligand_asl, njobs=1):
    """
    Run a prime mmgbsa calculations on a series of trajectory frames
    :param structures:
    :param ligand_asl:
    :param njobs:
    :return:
    """
    logger.info('Launching {} subjobs'.format(njobs))
    # Run MMGBSA calculation
    inp = []
    jobs = []
    for i, chunk in enumerate(np.array_split(np.arange(len(structures)), njobs)):
        jobname = 'mmgbsa_input{}'.format(i)
        infile = 'mmgbsa_input{}.mae'.format(i)
        for j in chunk:
            st = structures[j]
            st.append(infile)
        inp.append(jobname)

        args = [MMGBSA_CMD, infile,]

        mmgbsa_options = copy.copy(MMGBSA_OPTIONS)

        mmgbsa_options['-jobname'] = jobname

        mmgbsa_options['-ligand'] = ligand_asl

        for kv in mmgbsa_options.items():
            args.extend(kv)


        logger.info('Running Prime MMGBSA:')
        logger.info('$SCHRODINGER/'+' '.join(args))

        job = jobcontrol.launch_job(args)
        jobs.append(job)
    outfiles = []
    for job in jobs:
        job.wait()
        if job.succeeded():
            outfiles.append(job.getOutputFiles())
        else:
            raise  RuntimeError('ProteinPreparationWizard failed with {}'.format(job.ExitStatus))

    return outfiles

def _process(structure_dict):
    """
    Docstring
    :param structure_dict:
    :return:
    """

    outname = '{}_trj_mmgbsa'.format(structure_dict['structure']['code'])
    outfile = outname + '.tar.gz'

    fork = None
    # Check if transformers is called as part of a pipeline
    if 'pipeline' in structure_dict['custom']:
        pipeline = structure_dict['custom']['pipeline']
        fork = [pipeline[0], ]
        if len(pipeline) == 1:
            del(structure_dict['custom']['pipeline'])
        else:
            structure_dict['custom']['pipeline'] = pipeline[1:]

    # Get simulation files

    cms_file = str(structure_dict['files']['desmond_cms'])
    trjtar = structure_dict['files']['desmond_trjtar']

    # When run from cmdline accept frame dir
    if os.path.isdir(trjtar):
        trj_dir = trjtar
    elif tarfile.is_tarfile(trjtar):
        with tarfile.open(name=trjtar, mode='r:gz') as tfile:
            tfile.extractall()
            logger.info('extracting frameset')
            trj_dir = tfile.getnames()[0]
    else:
        raise RuntimeError('trjtar is neither a directory nor a tarfile')

    # Get Ligand ASL
    if LIGAND_ASL is None:
        logger.info('No ligand specified, automatically determining ligand')
        ligand_asl = get_ligand_asl(structure_dict)
        logger.info('Ligand ASL:\n{}'.format(ligand_asl))
    else:
        ligand_asl = LIGAND_ASL

    # Get ligand-receptor complex from trj

    structures = extract_st(cms_file, trj_dir, asl='protein or ({})'.format(ligand_asl), frames=NFRAMES)

    if NFRAMES<NPROC:
        njobs = NFRAMES
    else:
        njobs = NPROC

    outfiles = run_mmgbsa(structures, ligand_asl=ligand_asl, njobs=njobs)

    # merge outfiles

    df = pd.DataFrame()
    mae_file = '{}_mmgbsa_output-out.maegz'.format(outname)
    csv_file = '{}_mmgbsa_output-out.csv'.format(outname)
    for of in outfiles:
        for f in of:
            suffix = f.split('.')[-1]
            if suffix == 'maegz':
                st = structure.Structure.read(f)
                st.append(mae_file)
            elif suffix == 'csv':
                df = pd.concat([df, pd.read_csv(f, sep=',', index_col=0)], axis=0)
    df.to_csv(csv_file, sep=',')


    with tarfile.open(outfile, mode='w:gz') as tar:
        tar.add(mae_file)
        tar.add(csv_file)


    transformer_dict = {
        'structure': {
            'parent_structure_id': structure_dict['structure']['structure_id']
        },
        'files': {'trj_mmgbsa': outfile},
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
                        help='cms file, trajaectory and optional ligand structure')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='similarity_search',
                        help='Outfile prefix')
    parser.add_argument('-n',
                        '--nframes',
                        type=int,
                        dest='nframes',
                        default=20,
                        help='Number of snapshots to use in calculation')
    parser.add_argument('-l',
                        '--ligand_asl',
                        type=str,
                        dest='ligand_asl',
                        default=None,
                        help='Atom selection string specifying the ligand atoms.')

    return parser.parse_args()


def get_logger():
    cwd = os.getcwd()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(cwd, os.path.split(__file__)[-1][:-3] + '.log'), mode='w')
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

    global NFRAMES
    global LIGAND_ASL

    args = parse_args()

    NFRAMES = args.nframes
    LIGAND_ASL = args.ligand_asl
    prefix = args.prefix
    if len(args.infiles) == 2:
        cmsfile, trjtar = args.infiles
        ligand_mae = None
    else:
        cmsfile, trjtar, ligand_mae = args.infiles

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(os.getcwd(), ''.join(__file__.split('.')[:-1]) + '.log'))
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

    structure_dict_list = [{'structure': {'code': prefix,
                                          'structure_id': 0},
                            'files': {'desmond_cms': cmsfile,
                                      'desmond_trjtar': trjtar,
                                      'ligand_mae': ligand_mae},
                            'custom': {}
                            }]
    outp = [sd for sd in run(structure_dict_list)]

    with open('{}_trj_mmgbsa_transformer.json'.format(prefix), 'w') as fout:
        json.dump(outp, fout)


if __name__ == '__main__':
    import argparse
    logger = get_logger()
    main()