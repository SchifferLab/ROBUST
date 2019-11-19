import os
import sys
import time
import tarfile
import getpass
import logging
import warnings
import argparse


from scipy.stats import entropy

#from utils.pldb import *

# Add path to $PATH to make sure you can import modules
try:
    sys.path.append(os.path.split(__file__)[0])
    from pldb import *
except:
    warnings.warn('Could not import pldb helper functions.\n Some functions might not be available',
                  category=ImportWarning)


from schrodinger.structutils.analyze import find_ligands, evaluate_asl
from schrodinger.application.desmond.packages import topo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('descriptors',
                        type=str,
                        nargs='+',
                        help='Descriptors to calculate, valid arguments are:\nrms\nvdw\nelec\nhbond\ntorsion')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='table of input structures in csv format. "structure_id" column required')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='similarity_search',
                        help='Outfile prefix')
    parser.add_argument('--keep_names',
                        dest='keep_names',
                        action='store_true',
                        default=False,
                        help='Add "name" column to the preprocessed dataset, required for merging replicates')
    parser.add_argument('-t',
                        '--type',
                        dest='data_types',
                        default={'vdw': 'default', 'elec': 'default'},
                        help='Some datasets contain multiple data_types (e.g. cutom nonbonded data).'
                             'The data_type can be specified by passing a dictionary with {<descriptor>: <datatype>}.'
                             'Currently data_types are only relevant if descriptors were calculated on the PLDB',
                        type=json.loads),
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



def preprocess_rms(api, dataset, data_type=None):
    """

    :param api:
    :type api: pldbclient.api_client.Api
    :param dataset:
    :type dataset: pandas.Dataframe
    :param data_type: Placeholder
    :return:
    """

    if api is None:
        dataset.dropna(axis=0, subset=['desmond_cms', 'trj_rms'], inplace=True)

    df_rms = pd.DataFrame(index=dataset.structure_id)

    for index, stid in zip(dataset.index, dataset.structure_id):
        with tempfile.TemporaryDirectory() as tempdir:
            if api is None:
                msys_model, cms_model = topo.read_cms(dataset.loc[index, 'desmond_cms'])
                with open(dataset.loc[index, 'trj_rms'], 'r') as fh:
                    trj_rms = json.load(fh)
            else:
                # Load desomd cms
                msys_model, cms_model = get_desmond_cms(api, stid, dir=tempdir)
                # Load trj_rms
                trj_rms = get_trj_rms(api, stid, dir=tempdir)

        # Preprocess data
        for data in trj_rms:
            if data['name'] == 'calpha rmsd':
                df_rms.loc[stid, 'mean_rmsd'] = np.mean(data['results'])
            if data['name'] == 'calpha rmsf':
                for aid, x in zip(data['atom_ids'], data['results']):
                    df_rms.loc[stid, '{}:{}'.format(*aid[:-1])] = x
            if data['name'] == 'ligand_rmsf':
                    aid2an = {}
                    for n in evaluate_asl(cms_model, 'ligand and not a.element H'):
                        aid2an['{}:{}'.format(str(cms_model.atom[n].resnum),
                                              cms_model.atom[n].pdbname.strip())] = n
                    for aid, x in zip(data['atom_ids'], data['results']):
                        an = aid2an.get('{}:{}'.format(*aid[1:]))
                        if an is None:  # Legacy check, this is true if calculation included pseudoatoms
                            continue
                        label = '#{}'.format(an)
                        df_rms.loc[stid, label] = x
    df_rms.index = dataset.structure_id
    return df_rms


def preprocess_vdw(api, dataset, data_type='default'):
    """
    Get pairwise vdw interactions
    :param api:
    :param dataset:
    :param data_type:
    :return:
    """

    if data_type is None:
        raise ValueError('vdw datatype not set. Specify vdw data_type or use default data_types')

    energy_component = 'nonbonded_vdw'
    if api is None:  # Only process structures which contain all input files
        dataset.dropna(axis=0, subset=['desmond_cms', 'desmond_nonbonded'], inplace=True)

    vdw_dict = dict((index, {}) for index in dataset.index)

    for index, stid in zip(dataset.index, dataset.structure_id):
        # Load data
        if api is None:
            msys_model, cms_model = topo.read_cms(dataset.loc[index, 'desmond_cms'])
            with open(dataset.loc[index, 'desmond_nonbonded'], 'r') as fh:
                nonbonded_dict = json.load(fh)
        else:
            # Load desmond_cms
            msys_model, cms_model = get_desmond_cms(api, stid)
            # Load desmond_nonbonded
            nonbonded_dict = get_desmond_nonbonded(api, stid)

        # Check if data_type in nonbonded_dict
        if data_type not in nonbonded_dict:
            if data_type != 'default':
                raise ValueError('{} data type not claculated for {}'.format(data_type, stid))
            elif 'group_ids' in nonbonded_dict:
                logger.warning('{} uses outdated file format'.format(stid))
            else:
                raise ValueError('{} data type not claculated for {}'.format(data_type, stid))
        else:
            nonbonded_dict = nonbonded_dict[data_type]

        if 'ligand_resnum' not in dataset.columns and 'ligand_chain' not in dataset.columns:
            ligands = find_ligands(st=cms_model)
            assert (len(ligands) == 1)
            ligand_resid = None
            for atm in ligands[0].st.atom:
                if ligand_resid is None:
                    ligand_resid = (atm.resnum, atm.chain.strip())
                else:
                    if (atm.resnum, atm.chain.strip()) != ligand_resid:
                        raise ValueError('Structure {}\nFound multiple ligand residues!'.format(stid))
        else:
            ligand_resid = (int(dataset.loc[index, 'ligand_resnum']), dataset.loc[index, 'ligand_chain'])

        # Map group ids
        group_ids = nonbonded_dict['group_ids']
        id2resid = {}

        if all([type(gid) == list for gid in group_ids]):
            for i, resid in enumerate(map(tuple, group_ids)):
                if resid == ligand_resid:
                    id2resid[i] = 'LIG'
                else:
                    id2resid[i] = '{}:{}'.format(*resid)
        else:
            logger.warning('Custom group_ids found')
            for i, group_id in enumerate(group_ids):
                id2resid[i] = group_id

        vdw_keys = list(map(tuple, nonbonded_dict['results'][energy_component]['keys']))
        vdw_means = nonbonded_dict['results'][energy_component]['mean_potential']
        for key, mean in zip(vdw_keys, vdw_means):
            ri, rj = list(map(int, key))
            pair = sorted([id2resid[ri], id2resid[rj]])  # Sort ensure common pairs are always assigned the same pair_id
            pair_id = '{} - {}'.format(*pair)
            if pair_id not in vdw_dict[index]:
                vdw_dict[index][pair_id] = mean

    df_vdw = pd.DataFrame(vdw_dict).T
    # replace NaNs with 0
    df_vdw.fillna(0, inplace=True)
    df_vdw.index = dataset.structure_id
    return df_vdw


def preprocess_elec(api, dataset, data_type='default'):
    """
    Get pairwise electrostatic interactions
    :param api:
    :param dataset:
    :param data_type:
    :return:
    """

    if data_type is None:
        raise ValueError('elec data_type not set. Specify elec data_type or use default data_types')

    energy_component = 'nonbonded_elec'

    if api is None:
        dataset.dropna(axis=0, subset=['desmond_cms', 'desmond_nonbonded'], inplace=True)

    elec_dict = dict((index, {}) for index in dataset.index)

    for index, stid in zip(dataset.index, dataset.structure_id):
        # Load data
        if api is None:
            msys_model, cms_model = topo.read_cms(dataset.loc[index, 'desmond_cms'])
            with open(dataset.loc[index, 'desmond_nonbonded'], 'r') as fh:
                nonbonded_dict = json.load(fh)
        else:
            # Load desmond_cms
            msys_model, cms_model = get_desmond_cms(api, stid)
            # Load desmond_nonbonded
            nonbonded_dict = get_desmond_nonbonded(api, stid)

        # Check if data_type in nonbonded_dict
        if data_type not in nonbonded_dict:
            if data_type != 'default':
                raise ValueError('{} data type not claculated for {}'.format(data_type, stid))
            elif 'group_ids' in nonbonded_dict:
                logger.warning('{} uses outdated file format'.format(stid))
            else:
                raise ValueError('{} data type not claculated for {}'.format(data_type, stid))
        else:
            nonbonded_dict = nonbonded_dict[data_type]

        # Determine Ligand residue
        if 'ligand_resnum' not in dataset.columns and 'ligand_chain' not in dataset.columns:
            ligands = find_ligands(st=cms_model)
            assert (len(ligands) == 1)
            ligand_resid = None
            for atm in ligands[0].st.atom:
                if ligand_resid is None:
                    ligand_resid = (atm.resnum, atm.chain.strip())
                else:
                    if (atm.resnum, atm.chain.strip()) != ligand_resid:
                        raise ValueError('Structure {}\nFound multiple ligand residues.'.format(stid))
        else:
            ligand_resid = (int(dataset.loc[index, 'ligand_resnum']), dataset.loc[index, 'ligand_chain'])

        # Map group ids
        group_ids = nonbonded_dict['group_ids']
        id2resid = {}

        if all([type(gid) == list for gid in group_ids]):
            for i, resid in enumerate(map(tuple, group_ids)):
                if resid == ligand_resid:
                    id2resid[i] = 'LIG'
                else:
                    id2resid[i] = '{}:{}'.format(*resid)
        else:
            logger.warning('Custom group_ids found')
            for i, group_id in enumerate(group_ids):
                id2resid[i] = group_id

        elec_keys = list(map(tuple, nonbonded_dict['results'][energy_component]['keys']))
        elec_means = nonbonded_dict['results'][energy_component]['mean_potential']
        for key, mean in zip(elec_keys, elec_means):
            ri, rj = list(map(int, key))
            pair = sorted([id2resid[ri], id2resid[rj]])  # Sort ensure common pairs are always assigned the same pair_id
            pair_id = '{} - {}'.format(*pair)
            if pair_id not in elec_dict[index]:
                elec_dict[index][pair_id] = mean

    df_elec = pd.DataFrame(elec_dict).T
    # replace NaNs with 0
    df_elec.fillna(0, inplace=True)
    df_elec.index = dataset.structure_id
    return df_elec


def _load_hbonds(cms_model, data, ligand_resid=None):
    """

    :param cms_model:
    :param data:
    :param ligand_resid:
    :return:
    """
    # TODO update dict for general use
    # Includes PRO-1 and PHE-99
    equivalent_atoms_dict = {'GLU': [[('OE1', 'OE2'), 'OS']],
                             'ASP': [[('OD1', 'OD2'), 'OS']],
                             'GLN': [[('HE21', 'HE22'), 'HS'], [('OE1',), 'OS']],
                             'ASN': [[('HD21', 'HD22'), 'HS'], [('OD1',), 'OS']],
                             'SER': [[('HG',), 'HS'], [('OG',), 'OS']],
                             'THR': [[('HG1',), 'HS'], [('OG1',), 'OS']],
                             'TYR': [[('OH',), 'OS'], [('HH',), 'HS']],
                             'TRP': [[('HE1',), 'HS']],
                             'HIS': [[('HD1',), 'HS']],
                             'ARG': [[('HH11', 'HH12', 'HH21', 'HH22', 'HE'), 'HS']],
                             'LYS': [[('HZ1', 'HZ2', 'HZ3'), 'HS']],
                             'PRO': [[('H1', 'H2'), 'H']],
                             'PHE': [[('OXT',), 'O']]}

    categorical_columns = ['water_mediated', 'chain 1', 'chain 2', 'resnum 1', 'resnum 2',
                           'resname 1', 'resname 2', 'atomname 1', 'atomname 2']
    resid_columns = ['water_mediated', 'chain 1', 'chain 2', 'resnum 1', 'resnum 2', 'atomname 1', 'atomname 2']

    aggregate_rules = {'frequency': 'sum', 'water_mediated': 'first', 'chain 1': 'first',
                       'chain 2': 'first', 'resnum 1': 'first', 'resnum 2': 'first',
                       'resname 1': 'first', 'resname 2': 'first', 'atomname 1': 'first',
                       'atomname 2': 'first'}

    # Replace ligand resname with "LIG"
    for index in data.index:
        if (data.loc[index, 'resnum 1'], data.loc[index, 'chain 1']) == ligand_resid:
            data.loc[index, 'resname 1'] = 'LIG'
            data.loc[index, 'resnum 1'] = 1
            data.loc[index, 'chain 1'] = 'L'
            # Replace atomname with element
            data.loc[index, 'atomname 1'] = cms_model.atom[data.loc[index, 'atom index 1']].element
        elif (data.loc[index, 'resnum 2'], data.loc[index, 'chain 2']) == ligand_resid:
            data.loc[index, 'resname 2'] = 'LIG'
            data.loc[index, 'resnum 2'] = 1
            data.loc[index, 'chain 2'] = 'L'
            # Replace atomname with element
            data.loc[index, 'atomname 2'] = cms_model.atom[data.loc[index, 'atom index 2']].element

    data = data.drop(['$\\sigma$', 'atom index 1', 'atom index 2'], axis=1)

    # Replace pdbnames of equivalent atoms

    for res, equivalent_atoms in equivalent_atoms_dict.items():
        for pdbnames, symbol in equivalent_atoms:
            tmp_df = data.loc[np.logical_or((data['resname 1'] == res), (data['resname 2'] == res))]
            for name in pdbnames:
                tmp_df.loc[(tmp_df['resname 1'] == res) & (tmp_df['atomname 1'] == name), 'atomname 1'] = symbol
                tmp_df.loc[(tmp_df['resname 2'] == res) & (tmp_df['atomname 2'] == name), 'atomname 2'] = symbol
            data.update(tmp_df)

    data = data.groupby(categorical_columns).aggregate(aggregate_rules)
    data.index = np.arange(data.shape[0])

    # Collapse categorical columns

    data['water_mediated'] = data['water_mediated'].astype(int)
    data['resnum 1'] = data['resnum 1'].astype(int)
    data['resnum 2'] = data['resnum 2'].astype(int)
    hbond_index = []
    for i in data.index:
        hbond_index.append(':'.join(list(map(str, data.loc[i, resid_columns]))))
    data.index = hbond_index
    data.drop(categorical_columns, axis=1, inplace=True)
    return data


def preprocess_hbond(api, dataset, data_type=None):
    """

    :param api:
    :param dataset:
    :param data_type: Placeholder
    :return:
    """
    hbonds_df = pd.DataFrame()

    if api is None:
        dataset.dropna(axis=0, subset=['desmond_cms', 'trj_hbonds'], inplace=True)

    for index, stid in zip(dataset.index, dataset.loc[:, 'structure_id']):
        # Get data
        if api is None:
            msys_model, cms_model = topo.read_cms(dataset.loc[index, 'desmond_cms'])
            data = pd.read_csv(dataset.loc[index, 'trj_hbonds'], sep=',', index_col=0)
            data.index = np.arange(data.shape[0])
        else:
            msys_model, cms_model = get_desmond_cms(api, stid,)
            data = get_trj_hbonds(api, stid)

        # Get ligand
        if 'ligand_resnum' not in dataset.columns and 'ligand_chain' not in dataset.columns:
            ligands = find_ligands(st=cms_model)
            assert (len(ligands) == 1)
            ligand_resid = None
            for atm in ligands[0].st.atom:
                if ligand_resid is None:
                    ligand_resid = (atm.resnum, atm.chain.strip())
                else:
                    if (atm.resnum, atm.chain.strip()) != ligand_resid:
                        raise ValueError('Structure {}\nFound multiple ligand residues!'.format(stid))
        elif 'ligand_resnum' in dataset.columns and 'ligand_chain' in dataset.columns:
            ligand_resid = (int(dataset.loc[index, 'ligand_resnum']), dataset.loc[index, 'ligand_chain'])

        frequency_df = _load_hbonds(cms_model, data, ligand_resid=ligand_resid)
        for fi in frequency_df.index:
            hbonds_df.loc[index, fi] = frequency_df.loc[fi, 'frequency']
    hbonds_df.fillna(0, inplace=True)
    hbonds_df.index = dataset.structure_id
    return hbonds_df


def label_torsion(cms_model, aid1, aid2, aid3, aid4):
    """
    Label torsion angles according to common nomenclature.
    If no common labe exists return the atom indices separated by : and preceeded by #
    :param cms_model:
    :param aid1:
    :param aid2:
    :param aid3:
    :param aid4:
    :return:
    """
    atoms = list(cms_model.atom)
    gids = topo.aids2gids(cms_model, [aid1, aid2, aid3, aid4])
    atom_names = [atoms[gid].pdbname.strip() for gid in gids]
    backbone_torsion = {'C', 'N', 'CA'}
    chi_torsion = {'Chi1': {'N', 'CA', 'CB', 'CG', 'CG1', 'SG', 'OG', 'OG1'},
                   'Chi2': {'CA', 'CB', 'CG', 'CG1', 'CD', 'CD1', 'SD', 'OD1', 'ND1'},
                   'Chi3': {'CB', 'CG', 'CD', 'SD', 'CE', 'OE1', 'NE'},
                   'Chi4': {'CG', 'CD', 'CE', 'NE', 'CZ', 'NZ'}}
    if len(set(atom_names).difference(backbone_torsion)) == 0:
        # Here we use the fact that the phi angle includes 2 C atoms whereas the psi angle includes 2 N atoms
        bb_dict = dict(zip(*np.unique(atom_names, return_counts=True)))
        calpha = [atoms[gid] for gid in gids if atoms[gid].pdbname.strip() == 'CA'][0]
        chain = calpha.chain.strip()
        resnum = calpha.resnum
        if bb_dict['C'] == 2:
            return '{}:{}:Phi'.format(resnum, chain)
        elif bb_dict['N'] == 2:
            return '{}:{}:Psi'.format(resnum, chain)
        else:
            raise ValueError('Missassigned angle: {} {} {} {}'.format(*gids))
    elif all([atoms[gid].getResidue().isStandardResidue() for gid in gids]):
        for angle, atom_set in chi_torsion.items():
            if len(set(atom_names).difference(atom_set)) == 0:
                atm = [atoms[gid] for gid in gids][0]
                chain = atm.chain.strip()
                resnum = atm.resnum
                return '{}:{}:{}'.format(resnum, chain, angle)
    else:
        return '#' + ':'.join(list(map(str, [aid1, aid2, aid3, aid4])))


def preprocess_torsion(api, dataset, data_type=None):
    """

    :param api:
    :param dataset:
    :param data_type: Placeholder
    :return:
    """

    if api is None: # Only process structures which contain all input files
        dataset.dropna(axis=0, subset=['desmond_cms', 'trj_torsion'], inplace=True)

    torsion_entropy = pd.DataFrame(index=dataset.structure_id)

    bins = np.arange(-180, 198, 18)  # Bin size for calculating entropy

    cwd = os.path.abspath('./')

    for index, stid in zip(dataset.index, dataset.structure_id):
        with tempfile.TemporaryDirectory() as tempdir:

            os.chdir(tempdir)
            
            if api is None:
                msys_model, cms_model = topo.read_cms(dataset.loc[index, 'desmond_cms'])
                trj_torsion = dataset.loc[index, 'trj_torsion']
            else:
                # Load cms_model
                msys_model, cms_model = get_desmond_cms(api, stid, dir=tempdir)
                trj_torsion = get_trj_torsion(api, stid, dir=tempdir)

            with tarfile.open(trj_torsion, 'r:gz') as tar:
                tar.extractall(path=tempdir)

            torsion_ids = pd.read_csv('torsion_ids.csv', sep=',', index_col=0)
            for tid in torsion_ids.index:
                torsion_label = label_torsion(cms_model, *list(map(int, torsion_ids.loc[tid, :])))
                series = np.genfromtxt('torsion_{}.csv'.format(int(tid)), delimiter=',')
                counts, bins = np.histogram(series, bins=bins)
                freq = counts / np.sum(counts)
                torsion_entropy.loc[index, torsion_label] = entropy(freq)
        os.chdir(cwd)
    torsion_entropy.index = dataset.structure_id
    return torsion_entropy


def main(args):
    preprocessor = {'rms': preprocess_rms,
                    'vdw': preprocess_vdw,
                    'elec': preprocess_elec,
                    'hbond': preprocess_hbond,
                    'torsion': preprocess_torsion}

    logger.info('Loading dataset')
    dataset = pd.read_csv(args.input, sep=',')

    if 'structure_id' not in dataset.columns:
        raise ValueError('Structures need to be defined in "structure_id" column')
    if 'processed' in dataset.columns:
        logger.info(
            'Dropping {} structures, not flagged as processed'.format(dataset.shape[0] - dataset.processed.sum()))
        dataset = dataset[dataset.processed != 0]

    if args.keep_names and 'name' not in dataset.columns:
        raise ValueError('No "name" column in dataset.')

    dataset.structure_id = dataset.structure_id.astype(int)

    data_types = args.data_types

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

    descriptors = args.descriptors
    logger.info('Descriptors:' + '\n'.join(descriptors))
    for d in descriptors:
        outfile = '{}_{}.csv'.format(args.prefix, d)
        if d not in preprocessor:
            logger.error(d + ' is not a valid descriptor.\nValid options are:\n' + '\n'.join(list(preprocessor.keys())))
            continue
        else:
            logger.info('Preprocessing: {}'.format(d))
            func = preprocessor[d]
            data_type = data_types.get(d)  # Get data_type
            t = time.time()
            df = func(api, dataset, data_type=data_type)
            if args.keep_names:
                df.loc[:, 'name'] = dataset['name'].values
            df.to_csv(outfile, sep=',')
            logger.info('Preprocessed {} in {:.1f} seconds'.format(d, time.time() - t))
    logger.info('All Done!')
    return


if __name__ == '__main__':
    a = parse_args()
    logger = get_logger(a.prefix)
    logger.info('python ' + ' '.join(sys.argv))
    main(a)
