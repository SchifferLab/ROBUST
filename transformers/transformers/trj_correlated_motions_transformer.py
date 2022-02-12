from __future__ import print_function

import os

import tarfile
import json

import numpy as np
from numpyencoder import NumpyEncoder
import schrodinger.application.desmond.packages.topo as topo
import schrodinger.application.desmond.packages.traj as traj

import logging

logger = logging.getLogger(__name__)

ASL_CALC = '(protein and a.ptype CA) or (ligand and not a.element H)'


class SUPERIMPOSER:
    """
    Python implementation of the Kabsch algorithm for structural alignment
    """

    def __init__(self):
        self._rot = None
        self._tran = None

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

    def fit(self, reference_coords, coords):
        if coords is None or reference_coords is None:
            raise Exception("Invalid coordinates set.")

        n = reference_coords.shape
        m = coords.shape
        if n != m or not (n[1] == m[1] == 3):
            raise Exception("Coordinate number/dimension mismatch.")

        self._calc_rot_tran(reference_coords, coords)

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


class LinearCorrelation:
    """
    Compute the correlation matrix
    Class supports:

    Pearson Correlation
    Mutual Information
    General Correlation

    Pearson Correlation:

    Hunenberger, P. H., A. E. Mark, and W. F. Van Gunsteren.
    "Fluctuation and cross-correlation analysis of protein motions observed in nanosecond molecular dynamics simulations."
    Journal of molecular biology 252.4 (1995): 492-503.

    General Correlation:

    Lange, Oliver F., and Helmut Grubmuller.
    "Generalized correlation for biomolecular dynamics."
    Proteins: Structure, Function, and Bioinformatics 62.4 (2006): 1053-1061.
    """

    DIM = 3  # Dimensionality

    def __init__(self, cms_file, trj_dir, align_asl='a. CA', calc_asl=None, frames=None):
        """
        Docstring
        :param cms_file:
        :param trj_dir:
        :param align_asl:
        :param calc_asl:
        :param frames:
        """

        # Set alignment algorith
        align = SUPERIMPOSER()

        # load cms_model
        msys_model, cms_model = topo.read_cms(str(cms_file))

        ndx_align = topo.asl2gids(cms_model, align_asl)  # atoms for structural alignment

        if calc_asl is None:
            self.ndx_calc = topo.asl2gids(cms_model, align_asl)  # atoms to calculate the covariance
        else:
            self.ndx_calc = topo.asl2gids(cms_model, calc_asl)

        # Get atom total
        self.natoms = len(self.ndx_calc)
        # Get range of the triangular matrix [(natoms*DIM)*(natoms*DIM)]
        self.index_range = np.cumsum(np.arange(self.natoms * self.DIM))

        # Load framelist
        if frames is not None:
            frame_list = [frame for (i, frame) in enumerate(traj.read_traj(str(trj_dir))) if i in frames]
        else:
            frame_list = traj.read_traj(str(trj_dir))

        ref_coordinates = frame_list[0].pos()

        total_frames = len(frame_list)

        # Align trajectory and get coordinates
        traj_crds = np.zeros((total_frames, len(self.ndx_calc), self.DIM))
        logger.info('Aligning trajectory')
        for i, frame in enumerate(frame_list):
            frame_coordinates = frame.pos()
            # Calculate rotation & translation matrix
            align.fit(ref_coordinates[ndx_align], frame_coordinates[ndx_align])
            # Align all coordinates
            frame_coordinates_aligned = align.transform(frame_coordinates)
            # Safe coordinates
            traj_crds[i] = frame_coordinates_aligned[self.ndx_calc]

        logger.info('Calculating covariance matrix')
        self.covar_matrix = self.covariance(traj_crds)

    def covariance(self, traj_crds):
        """
        Compute the covariance matrix from a set of trajectory coordinates
        To safe time we only compute the lower half of the matrix
        By default np.tril_indices assumes a n*n matrix
        k is the diagonal offset
        """

        matrix_indices = np.vstack(
            (np.tril_indices(self.natoms * self.DIM, k=0)[0],
             np.tril_indices(self.natoms * self.DIM, k=0)[1])).T.astype(int)
        # Get covarianve array
        covar_matrix = np.zeros(len(matrix_indices))

        logger.info('compute covarinace matrix for {} atoms'.format(self.natoms))
        for n, ndx in enumerate(matrix_indices):
            # Get matrix indices
            i, j = ndx
            # Get atom indices
            a1 = int(i / self.DIM)
            a2 = int(j / self.DIM)
            # Get dimension indices
            d1 = i - a1 * self.DIM
            d2 = j - a2 * self.DIM
            # Get coordinate array (traj_crds = [nframe,natoms,3]
            crd1 = traj_crds[::, a1][::, d1]
            crd2 = traj_crds[::, a2][::, d2]
            # Get fluctuation r-<r>
            dev1 = np.subtract(crd1, np.mean(crd1))
            dev2 = np.subtract(crd2, np.mean(crd2))
            # Safe covariance
            covar_matrix[n] = np.mean(dev1 * dev2)
        return covar_matrix

    @property
    def pearson_correlation(self):
        """
        Calculate the pearson correlation coefficient.
        r = <xi,xj>/sqrt(<xi>**2*<xj>**2)
        """
        matrix_indices = np.vstack(
            (np.tril_indices(self.natoms, k=0)[0],
             np.tril_indices(self.natoms, k=0)[1])).T.astype(int)

        p_correlation = np.zeros(len(matrix_indices))

        for n, ndx in enumerate(matrix_indices):
            dai, daj = ndx * self.DIM
            xi = xj = xij = 0
            for d in range(self.DIM):
                xi += self.covar_matrix[self.index_range[dai + d] + dai + d]
                xj += self.covar_matrix[self.index_range[daj + d] + daj + d]
                xij += self.covar_matrix[self.index_range[dai + d] + daj + d]
            p_correlation[n] = xij / (np.sqrt(xi * xj))
        return p_correlation

    def _entropy(self, covar_matrix):
        """
        Calculate the differential entropy based on the Covariance
        This assumes that the data is normal distributed
        For proof of below formula see e.g:
        https://arxiv.org/pdf/1309.0482.pdf
        :param covar_matrix:
        :return:
        """

        dim = covar_matrix.shape[0]
        return (0.5 * dim) * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.linalg.det(covar_matrix))

    def _get_3x3(self, n1, n2):
        """
        Return a 3*3 Matrix from source
        Matrix origin: n1|n2

        Source is assumed to be the lower triangle of a matrix
        """
        matrix_out = np.zeros((3, 3))
        # Basic 3*3 matrix indices
        ndx_rows, ndx_columns = np.indices((3, 3))
        # Actual matrix indices
        ndx_rows += n1
        ndx_columns += n2

        for r, c in zip(ndx_rows, ndx_columns):
            for rc in zip(r, c):
                # Since we extract values from the lower half of a triangular matrix
                # i is always greater j
                if rc[0] < rc[1]:
                    i = rc[1]
                    j = rc[0]
                else:
                    i = rc[0]
                    j = rc[1]
                # get the 1D transform
                transform_1d = self.index_range[i] + j
                matrix_out[rc[0] - n1, rc[1] - n2] = self.covar_matrix[transform_1d]
        return matrix_out

    @property
    def mutual_information(self):
        """
        Calculate the Mutual Information for all natom*natom pairs
        (Or rather the lower half of the symmetric matrix)
        """

        matrix_indices = np.vstack(
            (np.tril_indices(self.natoms, k=0)[0],
             np.tril_indices(self.natoms, k=0)[1])).T.astype(int)

        mutual_information = np.zeros(len(matrix_indices))

        for n, ndx in enumerate(matrix_indices):
            ai, aj = ndx
            dai = ai * self.DIM
            daj = aj * self.DIM
            # Get sub matrices
            ci = self._get_3x3(dai, dai)
            cj = self._get_3x3(daj, daj)
            ij = self._get_3x3(dai, daj)
            # Get the 6x6 correlation matrix cij
            cij = np.vstack((np.hstack((ci, ij)), np.hstack((ij.T, cj))))
            # Calculate differential entropy
            e1 = self._entropy(ci)
            e2 = self._entropy(cj)
            e3 = self._entropy(cij)
            # Get Mutual Information
            mutual_information[n] += (e1 + e2 - e3)
        return mutual_information

    @property
    def general_correlation(self):
        """
        Transform results from Information into the generalized correlation coefficient
        """
        mutual_information = self.mutual_information  # Create local instance of mutual information
        general_correlation = np.zeros(mutual_information.shape)
        for n, mi in enumerate(mutual_information):
            if mi > 0:
                general_correlation[n] = np.sqrt(1 - np.exp(-2 / self.DIM * mi))
            else:
                general_correlation[n] = 0
        return general_correlation


def get_atom_ids(cms_model, index):
    """
    Convert gids to atom_ids: "chain resnum resname atom name"
    :param cms_model:
    :param index:
    :return:
    """
    atom_ids = []
    atoms = [a for a in cms_model.atom]
    for gid in index:
        a = atoms[gid]
        chain = a.chain.strip()
        resnum = a.resnum
        resname = a.pdbres.strip()
        atom_name = a.pdbname.strip()
        atom_ids.append('{} {} {} {}'.format(chain, resnum, resname, atom_name))
    return atom_ids


def generate_matrix(trimat, nrows):
    """
    Create matrix from the lower triangle of a square matrix
    :param trimat: triangle of a square matrix
    :param nrows: number of rows/columns)
    :return:
    """
    matrix_indices = np.vstack((np.tril_indices(nrows)[0],
                                np.tril_indices(nrows)[1])).T.astype(int)
    matrix = np.zeros((nrows, nrows))
    for n, ij in enumerate(matrix_indices):
        i, j = ij
        if i == j:
            matrix[(i, j)] = trimat[n]
        else:
            matrix[(i, j)] = trimat[n]
            matrix[(j, i)] = trimat[n]
    return matrix


def process(structure_dict):
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

    outname = '{}_trj_correlated_motions'.format(structure_dict['structure']['code'])
    outfile = outname + '.json'

    cms_file = structure_dict['files']['desmond_cms']

    msys_model, cms_model = topo.read_cms(str(cms_file))

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
    lc = LinearCorrelation(cms_file, trj_dir, ASL_CALC)
    gc_matrix = generate_matrix(lc.general_correlation, lc.natoms)
    pearson_matrix = generate_matrix(lc.pearson_correlation, lc.natoms)

    results = {'atom selection': ASL_CALC,
               'gids': lc.ndx_calc,
               'atom_ids': get_atom_ids(cms_model, lc.ndx_calc),
               'mutual_information': gc_matrix.tolist(),
               'pearson_correlation': pearson_matrix.tolist()}

    with open(outfile, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)

    transformer_dict = {
        "structure": {
            "parent_structure_id":
                structure_dict["structure"]["structure_id"]
        },
        "files": {"trj_correlated_motions": outfile},
        'custom': structure_dict['custom'],
    }
    if fork is not None:
        logger.info('Forking pipeline: ' + ' '.join(fork))
        transformer_dict['control'] = {'forks': fork}
    yield transformer_dict


def run(structure_dict_list):
    for structure_dict in structure_dict_list:
        for new_structure_dict in process(structure_dict):
            yield new_structure_dict


def parse_args():
    """
    Argument parser when script is run from commandline
    :return:
    """
    description = '''
    Calculates generalized correlated motion, mutual information, and pearson correlation between sets of carbon alpha atoms within a protein complex and returns results in a json dictionary
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

    return parser.parse_args()


def main(args):
    cms_file, trj= args.infiles
    prefix = args.prefix
    structure_dict_list = [
        {'structure': {'structure_id': 0, 'code': prefix},
         'files': {'desmond_cms': cms_file, 'desmond_trjtar': trj},
         'custom': []}]
    out_dict = [nsd for nsd in run(structure_dict_list)]
#    with open('{}_trj_hbond_transformer.json'.format(prefix), 'w') as fout:
 #       json.dump(out_dict, fout)

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

    structure_dict_list = [{'structure': {'structure_id': 0, 'code':'test'},
                            'files': {'desmond_cms': cms_file, 'desmond_trjtar': trj},'custom': []}]
    outp = []
    for sd in run(structure_dict_list):
        outp.append(sd)
    with open('testrun.json', 'w') as fout:
        json.dump(outp, fout)


if __name__ == '__main__':
    import argparse
    args = parse_args()
    main(args)

