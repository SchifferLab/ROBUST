import os
import sys
import getpass
import logging
import tempfile
import warnings
import argparse

import numpy as np
import pandas as pd

#from utils.pldb import *


# Add path to $PATH to make sure you can import modules
try:
    sys.path.append(os.path.split(__file__)[0])
    from pldb import *
except:
    warnings.warn('Could not import pldb helper functions.\n Some functions might not be available',
                  category=ImportWarning)

from schrodinger import structure

from schrodinger.application.desmond.packages import topo

from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils.analyze import find_ligands
from schrodinger.structutils.analyze import center_of_mass
from schrodinger.structutils.analyze import find_common_substructure


def parse_args():
    description='''
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data',
                        type=str,
                        nargs='+',
                        help='A list of data files to  merge')
    parser.add_argument('-t',
                        '--type',
                        type=str,
                        required=True,
                        help='data type, must be one of:\nrms\nvdw\nelec\nhbond\ntorsion')
    parser.add_argument('-i',
                        dest='dataset',                
                        type=str,
                        default=None,
                        help='the dataset in csv format, merging data files requires either the dataset or a pldb api '
                             'endpoint')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='trj_data',
                        help='Outfile prefix')
    parser.add_argument('--merge_replicates',
                        dest='merge_replicates',
                        action='store_true',
                        default=False,
                        help='In addition to merging data, merge replicates. This requires a "name" column')
    parser.add_argument('--endpoint',
                        type=str,
                        dest='endpoint',
                        default=None,
                        help='PLDB URL')
    parser.add_argument('--username',
                        type=str,
                        dest='username',
                        help='PLDB username')
    parser.add_argument('--password',
                        type=str,
                        dest='password',
                        default=None,
                        help='PLDB password')
    return parser.parse_args()


def get_logger(prefix):
    """

    :param prefix:
    :type prefix: str
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(os.getcwd(), prefix + '.log'), mode='w')
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


def get_ligand_atoms(st):
    """

    :param st:
    :type st: schrodinger.structure.Structure
    :return:
    """
    ligand_atoms = find_ligands(st)
    if len(ligand_atoms) == 0:
        raise ValueError('No Ligand found for {}'.format(st.property['s_m_title']))
    elif len(ligand_atoms) > 1:
        raise ValueError('Multiple Ligands found for {}'.format(st.property['s_m_title']))
    else:
        return ligand_atoms[0].atom_indexes


def com_distance(st, atoms1, st2=None, atoms2=None):
    """
    Return Center of Mass distance for two sets of atoms.
    If one set of atoms and one structure is provided,
    the distance between the subset and the full structure is returned

    If a second stucture (st2) is provided atoms2 is a subset of st2.
    :param st:
    :type st: schrodinger.structure.Structure
    :param atoms1:
    :type atoms1: list
    :param st2:
    :type st2: schrodinger.structure.Structure
    :param atoms2:
    :type atoms2: list
    :return:
    """
    if st2 is None:
        st2 = st
    if atoms2 is None:
        atoms2 = [atm.index for atm in st2.atom]

    com1 = center_of_mass(st, atoms1)
    com2 = center_of_mass(st2, atoms2)
    return np.sqrt(np.sum((com1 - com2) ** 2))


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


def get_original_atom_index(st, atom_ids):
    """
    Get orig_atom_index property
    :param st:
    :type st: schrodinger.structure.Structure
    :param atom_ids:
    :type atom_ids: list
    :return:
    """
    original_atom_indices = []
    for aid in atom_ids:
        oaid = st.atom[aid].property.get('i_m_orig_atom_index')
        if oaid is None:
            raise ValueError('i_m_orig_atom_index not set for atom_id: {}'.format(aid))
        else:
            original_atom_indices.append(oaid)
    return original_atom_indices


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


def get_min_common_substructure(structures, ligand_ids=None, return_st=True, mcs_color=(139, 0, 139),
                                return_common_atoms=False):
    """
    Find the minimum common substructure
    :param structures: List of schrodinger.structure.Structure to calculate the mcs from
    :type structures: [schrodinger.structure.Structure]
    :param return_st: Boolean If True the substructure will be returned
    :param mcs_color: Maximum common substructure will be colored according to RGB values
    :param return_common_atoms: If True return only the list of common atoms
    :return:
    """
    for st in structures:
        set_original_atom_index(st)

    ligand_structures = []
    if ligand_ids is None:
        for st in structures:
            ligand_atoms = get_ligand_atoms(st)
            ligand_structures.append(st.extract(ligand_atoms, copy_props=True))
    else:
        for st, r, c in zip(structures, ligand_ids['ligand_resnum'], ligand_ids['ligand_chain']):
            ligand_structures.append(st.extract(evaluate_asl(st, 'r. {} and c. {} and not a.element H'.format(int(r), c)),
                                                copy_props=True))

    default_color = (211, 211, 211)  # light grey
    substructures = find_common_substructure(ligand_structures, atomTyping=12, allow_broken_rings=True)
    common_atoms = []
    for j, subst in enumerate(substructures):
        if not len(subst) == 1:
            st = ligand_structures[j]
            ligand_atoms = [a.index for a in st.atom]
            com_distances = []
            for match in subst:
                com_distances.append(com_distance(st, ligand_atoms, atoms2=match))
            common_atoms.append(subst[np.argmin(com_distance)])
        else:
            common_atoms.append(subst[0])

    orig_atoms = [get_original_atom_index(st, atom_ids) for st, atom_ids in zip(ligand_structures, common_atoms)]
    if return_common_atoms and not return_st:
        return orig_atoms
    elif return_common_atoms and return_st:
        superimposer = Superimposer()
        for i, (catm, st) in enumerate(zip(common_atoms, ligand_structures)):
            if i == 0:
                reference_coord = np.array([st.atom[x].xyz for x in catm])
            else:
                mobile_coord = np.array([st.atom[x].xyz for x in catm])
                superimposer.fit(reference_coord, mobile_coord)
                st.setXYZ(superimposer.transform(st.getXYZ()))
            for atm in st.atom:  # Color all atoms in base gray
                atm.setColorRGB(*default_color)
                atm.temperature_factor = 0.
                atm.label_user_text = ''
                atm.label_color = 1
                atm.label_format = ''
            for i, aid in enumerate(catm):
                atm = st.atom[aid]  # type: structure._StructureAtom
                atm.setColorRGB(*mcs_color)
                atm.temperature_factor = 1.
                atm.label_user_text = str(i + 1)
                atm.label_color = 10
                atm.label_format = '%UT'
        return orig_atoms, ligand_structures

    mcs = np.min(list(map(len, common_atoms)))
    if not return_st:
        return mcs
    else:
        for catm, st in zip(common_atoms, ligand_structures):
            for atm in st.atom:  # Color all atoms in base gray
                atm.setColorRGB(*default_color)
                atm.temperature_factor = 0.
                atm.label_user_text = ''
                atm.label_color = 1
                atm.label_format = ''
            for i, aid in enumerate(catm):
                atm = st.atom[aid]  # type: structure._StructureAtom
                atm.setColorRGB(*mcs_color)
                atm.temperature_factor = 1.
                atm.label_user_text = str(i + 1)
                atm.label_color = 10
                atm.label_format = '%UT'
        return mcs, ligand_structures


def merge_rms(data, dataset=None, api=None):
    """
    Merge the trj_rms data.
    Merger can also be called only one data file.
    :param data:
    :type data: pd.DataFrame
    :param dataset:
    :type dataset: pd.DataFrame
    :param api:
    :type api: pldbclient.api_client.Api
    :return:
    """
    if len(data) == 1:
        df_raw = data[0]
    else:
        df_raw = pd.concat(data, sort=False)

    if api is None:
        dataset.dropna(axis=0, subset=['desmond_cms', 'trj_rms'], inplace=True)
        ligand_ids = dataset.loc[:, ['ligand_resnum', 'ligand_chain']]
    else:
        ligand_ids = None

    ligand_col = [c for c in df_raw.columns if c[0] == '#']  # ligand_rmsf columns are indicated by '#<atom_id>'

    # Get structures
    cms_models = []
    for index in df_raw.index:
        if api is None:
            msys_model, cms_model = topo.read_cms(dataset.loc[index, 'desmond_cms'])
        else:
            msys_model, cms_model = get_desmond_cms(api, index)
        cms_models.append(cms_model)

    # Get a list (of lists) of common atoms
    common_atoms, ligand_st = get_min_common_substructure(cms_models, ligand_ids=ligand_ids, return_st=True,
                                                          return_common_atoms=True)
    if not all(common_atoms):  # translate: if common atoms is a list of empty lists
        logger.warning('Ligands do not share a common substructure')
        df_raw.drop(ligand_col, axis=1, inplace=True)
        return df_raw, None
    else:
        df_merged = df_raw.drop(ligand_col, axis=1)
        # Assign common atom ids
        cid_map = dict([(stid, {}) for stid in df_merged.index])
        for i, atom_array in enumerate(zip(*common_atoms)):
            for stid, aid in zip(df_merged.index, atom_array):
                cid_map[stid][aid] = i + 1  # Because atom indices start at 1
        for stid in df_merged.index:
            for aid in ligand_col:
                if not pd.isna(df_raw.loc[stid, aid]):
                    x = df_raw.loc[stid, aid]
                    aid = int(aid[1:])  # Drop the preceeding '#'
                    if aid in cid_map[stid]:  # If aid has a common atom id
                        label = 'LIG:{}'.format(cid_map[stid][aid])
                        df_merged.loc[stid, label] = x
        return df_merged, ligand_st


def merge_vdw(data, dataset=None, api=None):
    """

    :param data:
    :type data: pd.DataFrame
    :param dataset: Placeholder
    :param api: Placeholder
    :return:
    """
    if len(data) == 1:
        return data[0], None
    df_merged = pd.concat(data, sort=False)  # type: pd.DataFrame
    df_merged.fillna(0, inplace=True)
    return df_merged, None


def merge_elec(data, dataset=None, api=None):
    """

    :param data:
    :type data: pd.DataFrame
    :param dataset: Placeholder
    :param api: Placeholder
    :return:
    """
    if len(data) == 1:
        return data[0], None
    df_merged = pd.concat(data, sort=False)  # type: pd.DataFrame
    df_merged.fillna(0, inplace=True)
    return df_merged, None


def merge_hbond(data, dataset=None, api=None):
    """

    :param data:
    :type data: pd.Dataframe
    :param dataset: Placeholder
    :param api: Placeholder
    :return:
    """
    if len(data) == 1:
        return data[0], None
    df_merged = pd.concat(data, sort=False)  # type: pd.DataFrame
    df_merged.fillna(0, inplace=True)
    return df_merged, None


def merge_torsion(data, dataset=None, api=None):
    """

    :param data:
    :type data: pd.DataFrame
    :param dataset:
    :type dataset: pd.DataFrame
    :param api:
    :type api: pldbclient.api_client.Api
    :return:
    """
    if len(data) == 1:
        df_raw = data[0]  # type: pd.DataFrame
    else:
        df_raw = pd.concat(data, sort=False)  # type: pd.DataFrame

    ligand_col = [c for c in df_raw.columns if c[0] == '#']  # ligand_torsion columns are indicated by '#<a1:a2:a3:a4>'

    # Get structures
    cms_models = []
    for stid in df_raw.index:
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                resp = api.get_structure_file(structure_id=stid, file_type='desmond_cms')
                resp.raise_for_status()
            except HTTPError as e:
                resp.close()
                logger.error(e)

            with open(os.path.join(tempdir, 'desmond.cms'), 'wb') as fh:
                fh.write(resp.content)
            resp.close()
            msys_model, cms_model = topo.read_cms(os.path.join(tempdir, 'desmond.cms'))
        cms_models.append(cms_model)

    common_atoms, ligand_st = get_min_common_substructure(cms_models, return_st=True, return_common_atoms=True)
    if not all(common_atoms):  # translate: if common atoms is a list of empty lists
        logger.warning('Ligands do not share a common substructure')
        df_raw.drop(ligand_col, axis=1, inplace=True)
        return df_raw, None
    else:
        df_merged = df_raw.drop(ligand_col, axis=1)  # type: pd.DataFrame
        # Assign common atom ids
        cid_map = dict([(stid, {}) for stid in df_merged.index])
        for i, atom_array in enumerate(zip(*common_atoms)):
            for stid, aid in zip(df_merged.index, atom_array):
                cid_map[stid][aid] = i + 1  # Because atom indices start at 1
        for stid in df_merged.index:
            for torsion_atoms in ligand_col:
                if not pd.isna(df_raw.loc[stid, torsion_atoms]):
                    x = df_raw.loc[stid, torsion_atoms]
                    torsion_atoms = list(map(int, torsion_atoms[1:].split(':')))  # torsion ids == '#a1:a2:a3:a4'
                    if all([aid in cid_map[stid] for aid in torsion_atoms]):  # If aid has a common atom id
                        torsion_bond = sorted([cid_map[stid][torsion_atoms[1]], cid_map[stid][torsion_atoms[2]]])
                        label = 'LIG:{}:{}'.format(*torsion_bond)
                        df_merged.loc[stid, label] = x
        return df_merged, ligand_st


def main(args):
    merger = {'rms': merge_rms,
              'vdw': merge_vdw,
              'elec': merge_elec,
              'hbond': merge_hbond,
              'torsion': merge_torsion}

    if args.type not in merger:
        logger.error('{} is not a valid datatype'.format(args.type))
        logger.info('Valid datatypes:\n' + '\n'.join(list(merger.keys())))
        raise ValueError
    if args.dataset is None:
        dataset = None
    else:
        dataset = pd.read_csv(args.dataset, sep=',', index_col=0)

    merger = merger[args.type]
    outfile = '{}_{}_merged.csv'.format(args.prefix, args.type)
    outst = '{}_{}_atom_labels.maegz'.format(args.prefix, args.type)

    # Load data

    data = []
    for f in args.data:
        data.append(pd.read_csv(f, index_col=0, sep=','))

    if len(data) == 1 and args.type in ('vdw', 'elec', 'hbond') and not args.merge_replicates:
        logger.warning('Attempting to merge a single {} dataset'.format(args.type))
        logger.warning('Merging a single dataset only makes sense for rms and torsion')
        logger.warning('Nothing to do here.')
        sys.exit(0)

    if args.merge_replicates:  # Check if all data have a name column
        names = pd.DataFrame()
        for i, df in enumerate(data):
            if 'name' not in df.columns:
                logger.warning('Dataset {} does not contain a "name" column.'.format(i))
                logger.warning('A "name" column is required to merge replicates.')
                for index in df.index:
                    names.loc[index, 'name'] = index
            else:
                for index, name in zip(df.index, df.name):
                    names.loc[index, 'name'] = name
        names['name'] = names['name'].astype(str)

    logger.info('Merging {} {} data'.format(len(data), args.type))
    n_structures = np.sum(list(map(len, data)))
    structure_ids = None
    for i, df in enumerate(data):
        if 'name' in df.columns:
            df.drop(['name', ], axis=1)  # We don't need the "name" column anymore
        if i == 0:
            structure_ids = df.index
        else:
            structure_ids = structure_ids.union(df.index)
    if len(structure_ids) != n_structures:
        raise ValueError('Duplicate structures in data')

    endpoint = args.endpoint
    username = args.username
    password = args.password
    if endpoint is None:
        api = None
    else:
        if password is None:
            password = getpass.getpass()
        logger.info('Connecting to PLDB endpoint: {}'.format(endpoint))
        api = pldb_client(endpoint, username, password)

    df, ligand_structures = merger(data, dataset, api)

    if args.merge_replicates:
        df_merged = pd.DataFrame(columns=df.columns)
        for name, count in zip(*np.unique(names['name'], return_counts=True)):
            logger.info('Merging {} replicates for {}'.format(count, name))
            replicates = df.index[names['name'] == name]
            df_merged.loc[name] = df.loc[replicates].mean(axis=0)
        if ligand_structures is not None:
            if os.path.isfile(outst):
                logger.warning('Overwriting atom labels structure: {}'.format(outst))
                os.remove(outst)
            for name, index in zip(*np.unique(names['name'], return_index=True)):
                st = ligand_structures[index]
                st.property['s_m_title'] = '{}_{}_atom_labels'.format(name, args.type)
                st.append(outst)
        df_merged.drop('name', axis=1, inplace=True)
        df_merged.to_csv(outfile, sep=',')
    else:
        df.drop('name', axis=1, inplace=True)
        df.to_csv(outfile, sep=',')
        if ligand_structures is not None:
            if os.path.isfile(outst):
                logger.warning('Overwriting atom labels structure: {}'.format(outst))
                os.remove(outst)
            for index, st in zip(data.index, ligand_structures):
                st.property['s_m_title'] = '{}_{}_atom_labels'.format(index, args.type)
                st.append(outst)
    logger.info('All Done!')
    return


if __name__ == '__main__':
    a = parse_args()
    logger = get_logger('merge_data')
    logger.info('python ' + ' '.join(sys.argv))
    if a.endpoint is None and a.dataset is None:
        raise ValueError('Merging data files requires either the dataset or a pldb api endpoint')
    main(a)
