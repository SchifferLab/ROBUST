from __future__ import print_function, division

import functools
import logging
import os
import subprocess
import tarfile
from multiprocessing import Pool
import json

import numpy as np
import pandas as pd
from schrodinger.application.desmond.packages import topo, traj
from schrodinger.structutils.analyze import evaluate_asl

logger = logging.getLogger(__name__)

NPROC = 12
MAX_FRAMES = 6000


def worker(frames_ind, cms_fn="", trj_fn=""):
    ### Streamlining the generation of a balls/contacts file for a given structure
    def voronota_commands(st, balls_txt='balls.txt', contacts_txt='contacts.txt', input_pdb="input_pdb.pdb"):

        st.write(input_pdb)

        # ./voronota get-balls-from-atoms-file < input.pdb > balls.txt
        cmd = ['voronota', 'get-balls-from-atoms-file', '--include-heteroatoms', '<', input_pdb, '>', balls_txt]

        process = subprocess.Popen(' '.join(cmd), shell=True, stdout=subprocess.PIPE)
        process.wait()
        # print(process.returncode)

        # ./voronota calculate-contacts < balls.txt > contacts.txt
        cmd = ['voronota', 'calculate-contacts', '<', balls_txt, '>', contacts_txt]

        process = subprocess.Popen(' '.join(cmd), shell=True, stdout=subprocess.PIPE)
        process.wait()
        # print(process.returncode)

        balls_df = pd.read_csv(balls_txt, sep=" ", header=None)

        contacts_df = pd.read_csv(contacts_txt, sep=" ", header=None)

        return balls_df, contacts_df

    ### Creating dataframe that only includes contacts between water-protein and water-ligand
    def parsing_voronoi_contacts(balls_df,
                                 contacts_df,
                                 solvent):

        # Format of resulting dataframe: Chain ID | Residue Number | Residue Name | # Water Contacts | Total Contact Area
        water_contacts_df = pd.DataFrame().reindex_like(contacts_df)
        water_contacts_df.columns = ["Residue Number", "Residue Name", "Chain ID"]
        water_contacts_df["Residue Number"] = int()
        water_contacts_df["Residue Name"] = str()
        water_contacts_df["Chain ID"] = str()
        water_contacts_df["# Water Contacts"] = int()
        water_contacts_df["Total Contact Area"] = np.nan

        # going through the contacts_df generated by voronota
        for i in range(0, len(contacts_df), 1):

            # Format of contacts_df: Atom 1 | Atom 2 | Contact Area
            current_atom1 = contacts_df.iat[i, 0]
            current_atom2 = contacts_df.iat[i, 1]

            # Format of balls_df: x | y | z | Radius of Sphere
            #                   | # | Atom # | Chain ID | Residue # | Residue Name | Atom Name | . | .
            atom1_resname = str(balls_df.iat[current_atom1, 8]).strip()
            atom2_resname = str(balls_df.iat[current_atom2, 8]).strip()

            # add atom information into final dataframe if atom1 is a protein or ligand that is in contact with water
            if atom1_resname not in solvent and atom2_resname in solvent:
                # collecting information from balls_df and contacts_df that will be added to final dataframe
                resnum = balls_df.iat[current_atom1, 7]
                resname = balls_df.iat[current_atom1, 8]
                chain_ID = balls_df.iat[current_atom1, 6]
                # all water contacts are set to one so that groupby function can add them
                num_water_contacts = 1
                contact_area = contacts_df.iat[i, 2]

                # placing the information into final dataframe: water_contacts
                water_contacts_df.iat[i, 0] = resnum
                water_contacts_df.iat[i, 1] = resname
                water_contacts_df.iat[i, 2] = chain_ID
                water_contacts_df.iat[i, 3] = num_water_contacts
                water_contacts_df.iat[i, 4] = contact_area


            elif atom1_resname in solvent and atom2_resname not in solvent:
                # collecting information from balls_df and contacts_df that will be added to final dataframe
                resnum = balls_df.iat[current_atom2, 7]
                resname = balls_df.iat[current_atom2, 8]
                chain_ID = balls_df.iat[current_atom2, 6]
                # all water contacts are set to one so that groupby function can add them
                num_water_contacts = 1
                contact_area = contacts_df.iat[i, 2]

                # placing the information into final dataframe: water_contacts
                water_contacts_df.iat[i, 0] = resnum
                water_contacts_df.iat[i, 1] = resname
                water_contacts_df.iat[i, 2] = chain_ID
                water_contacts_df.iat[i, 3] = num_water_contacts
                water_contacts_df.iat[i, 4] = contact_area

        # remove all contacts that are not water-protein and water-ligand
        water_contacts_df = water_contacts_df.dropna(how="any")

        # reindex water_contacts so that they are sequential and start from 0
        wat_cont_index = list(range(0, len(water_contacts_df)))
        water_contacts_df.index = wat_cont_index

        # adding up the water contacts and contact areas for the atoms that have
        # the same chain ID, residue number, and residue name
        water_contacts_df = water_contacts_df.groupby(by=["Chain ID", "Residue Number", "Residue Name"],
                                                      as_index=False).agg(
            {"# Water Contacts": sum, "Total Contact Area": sum})

        return water_contacts_df

    ## Finding the water contacts for the frames assigned to a given processor
    water_contacts_list = []

    process_ID = os.getpid()
    balls_txt = "balls_{}.txt".format(process_ID)
    contacts_txt = "contacts_{}.txt".format(process_ID)
    input_pdb = "input_pdb{}.pdb".format(process_ID)

    msys_model, cms_model = topo.read_cms(cms_fn)
    traj_list = traj.read_traj(trj_fn)

    for i in frames_ind:
        reduced_cms = topo.update_cms(cms_model, traj_list[i], update_pseudoatoms=False)
        reduced_st_aids = evaluate_asl(reduced_cms, "(fillres within 6 (protein or ligand)) and not ions")
        reduced_st = reduced_cms.extract(reduced_st_aids)

        solvent_ind = evaluate_asl(cms_model, "solvent")

        res_names = []
        for i in solvent_ind:
            res_names.append(cms_model.atom[i].pdbres.strip())

        solvent_name = np.unique(res_names)

        # Reassign solvent residue numbers
        solvent_resnum = 1
        for res in reduced_st.residue:

            if res.pdbres.strip() in solvent_name:
                res.resnum = solvent_resnum

                solvent_resnum += 1

        # Calling the functions within the worker function
        balls_df, contacts_df = voronota_commands(st=reduced_st, balls_txt=balls_txt,
                                                  contacts_txt=contacts_txt, input_pdb=input_pdb)
        water_contacts_df = parsing_voronoi_contacts(balls_df=balls_df, contacts_df=contacts_df, solvent = solvent_name)

        water_contacts_list.append(water_contacts_df)

    # Concatenates all the water contacts df into one df
    # The output from all processors will generate a list of these df's
    water_contacts_all = pd.concat(water_contacts_list)

    return water_contacts_all


def parsing_worker(out, n_frames):
    """

    :param out:
    :type out: list
    :param n_frames:
    :type n_frames: int
    :return:
    """
    water_contacts_final = pd.concat(out)
    water_contacts_final.insert(4, "WC_STD", water_contacts_final["# Water Contacts"])
    water_contacts_final.insert(6, "TCA_STD", water_contacts_final["Total Contact Area"])

    water_contacts_final = water_contacts_final.groupby(by=["Chain ID", "Residue Number", "Residue Name"],
                                                        as_index=False).agg(
        {"# Water Contacts": sum, "WC_STD": np.std,
         "Total Contact Area": sum, "TCA_STD": np.std})

    for i in range(0, len(water_contacts_final), 1):
        n_wc_sum = water_contacts_final.iat[i, 3]
        n_wc_mean = n_wc_sum / n_frames
        water_contacts_final.iat[i, 3] = n_wc_mean

        tca_sum = water_contacts_final.iat[i, 5]
        tca_mean = tca_sum / n_frames
        water_contacts_final.iat[i, 5] = tca_mean

    return water_contacts_final


def _process(structure_dict):
    fork = None
    # Check if transformers is called as part of a pipeline
    if 'pipeline' in structure_dict['custom']:
        pipeline = structure_dict['custom']['pipeline']
        fork = [pipeline[0], ]
        if len(pipeline) == 1:
            del (structure_dict['custom']['pipeline'])
        else:
            structure_dict['custom']['pipeline'] = pipeline[1:]

    cms_fn = str(structure_dict["files"]["desmond_cms"])
    trjtar = str(structure_dict["files"]["desmond_trjtar"])

    # If run from command line it does not make sense to provide a tarfile
    if os.path.isdir(trjtar):
        trj_dir = trjtar
    elif tarfile.is_tarfile(trjtar):
        with tarfile.open(name=trjtar, mode='r:gz') as tfile:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tfile)
            logger.info('extracting frameset')
            trj_dir = tfile.getnames()[0]
    else:
        raise RuntimeError('trjtar is neither a directory nor a tarfile')

    traj_list = traj.read_traj(trj_dir)
    n_frames_total = len(traj_list)

    if n_frames_total < MAX_FRAMES:
        frames_ind = list(range(0, n_frames_total, 1))
    else:
        frames_ind = np.linspace(0, n_frames_total, MAX_FRAMES, endpoint=False).astype(int)

    del (traj_list)

    n_frames_used = len(frames_ind)

    workload = np.array_split(frames_ind, NPROC)

    pool = Pool(processes=NPROC)

    func = functools.partial(worker, cms_fn=cms_fn, trj_fn=trj_dir)

    out = pool.map(func, workload)

    water_contacts_final = parsing_worker(out=out, n_frames=n_frames_used)

    water_contacts_final.to_csv("Water_Contacts_Final.csv")

    transformer_dict = {
        'structure': {
            'parent_structure_id':
                structure_dict['structure']['structure_id'],
            'searchable': False
        },
        'files': {'trj_water_contacts': "Water_Contacts_Final.csv"},
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
    # TODO change description
    description = '''
        Calculates the number of contacts between water-protein and water-ligand residues\n
        as well as their collective contact areas.\n
        
        Voronota software is used to generate the water-protein and water-ligand contacts,\n
        which is based on voronoi diagrams for each atom. Additional data parsing generates\n
         
        
        

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
    parser.add_argument('--max_frames',
                        type=int,
                        dest='max_frames',
                        default=2000,
                        help='Process at most [max_frames] frames equally spaced across trajectory.\nDefault: 2000')

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
    global MAX_FRAMES

    NPROC = args.nproc
    MAX_FRAMES = args.max_frames

    cms_file, trj = args.infiles
    prefix = args.prefix

    structure_dict_list = [
        {'structure': {'structure_id': 0, 'code': prefix},
         'files': {'desmond_cms': cms_file, 'desmond_trjtar': trj},
         'custom': []}]
    out_dict = [nsd for nsd in run(structure_dict_list)]
    with open('{}_trajectory_water_contacts.json'.format(prefix), 'w') as fout:
        json.dump(out_dict, fout)


if __name__ == '__main__':
    import argparse

    args = parse_args()
    logger = get_logger()
    main(args)

