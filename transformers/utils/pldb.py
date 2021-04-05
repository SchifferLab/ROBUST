import os
import json
import warnings
import tempfile
import atexit

import pandas as pd
import numpy as np

from requests import HTTPError
try:
    from pldbclient.api_client import Api
except ImportWarning as w:
    warnings.warn('Failed to import pldbclient, some functunality will not be available', ImportWarning)

from schrodinger.application.desmond.packages import topo


def pldb_client(endpoint, username, password, logger):
    if password is None:
        password = getpass.getpass()
    api = Api(endpoint, username, password)
    try:
        resp = api.get_structure_codes()
        resp.raise_for_status()
    except Exception as e:
        logger.error(e)
        raise HTTPError(e)
    finally:
        resp.close()
    return api


def get_desmond_cms(api, stid, dir=None):
    """

    :param api: 
    :param stid: 
    :param dir: 
    :return: 
    """

    if dir is None:
        tempdir = tempfile.TemporaryDirectory()
        atexit.register(tempdir.cleanup)
        dir = tempdir.name

    filename = os.path.join(dir, 'desmond.cms')
    try:
        resp = api.get_structure_file(stid, file_type='desmond_cms')
        resp.raise_for_status()
    except Exception as e:
        resp.close()
        raise HTTPError(e)
    with open(filename, 'wb') as fh:
        fh.write(resp.content)
    resp.close()
    return topo.read_cms(filename)


def get_trj_rms(api, stid, dir=None):
    """

    :param api: 
    :param stid: 
    :param dir: 
    :return: 
    """

    if dir is None:
        tempdir = tempfile.TemporaryDirectory()
        atexit.register(tempdir.cleanup)
        dir = tempdir.name

    filename = os.path.join(dir, 'trj_rms.json')
    try:
        resp = api.get_structure_file(stid, file_type='trj_rms')
        resp.raise_for_status()
    except Exception as e:
        resp.close()
        raise HTTPError(e)
    with open(filename, 'wb') as fh:
        fh.write(resp.content)
    resp.close()
    with open(filename, 'r') as fh:
        trj_rms = json.load(fh)
    return trj_rms


def get_desmond_nonbonded(api, stid, dir=None):
    """

    :param api: 
    :param sitd: 
    :param dir: 
    :return: 
    """

    if dir is None:
        tempdir = tempfile.TemporaryDirectory()
        atexit.register(tempdir.cleanup)
        dir = tempdir.name
    try:
        resp = api.get_structure_file(structure_id=stid, file_type='desmond_nonbonded')
        resp.raise_for_status()
        with open(os.path.join(dir, 'desmond_nonbonded.json'), 'wb') as fh:
            fh.write(resp.content)
    except HTTPError as e:
        raise HTTPError(e)
    finally:
        resp.close()
    with open(os.path.join(dir, 'desmond_nonbonded.json'), 'r') as fh:
        nonbonded_dict = json.load(fh)
    return nonbonded_dict


def get_trj_hbonds(api, stid, dir=None):
    """
    
    :param api: 
    :param stid: 
    :param dir: 
    :return: 
    """
    if dir is None:
        tempdir = tempfile.TemporaryDirectory()
        atexit.register(tempdir.cleanup)
        dir = tempdir.name

    try:
        resp = api.get_structure_file(structure_id=stid, file_type='trj_hbonds')
        resp.raise_for_status()
    except HTTPError as e:
        resp.close()
        raise HTTPError(e)
    filename = os.path.join(dir, '{}_trj_hbonds.csv'.format(stid))
    with open(filename, 'wb') as fh:
        fh.write(resp.content)
    resp.close()
    
    data = pd.read_csv(filename, sep=',', index_col=0)
    data.index = np.arange(data.shape[0])
    
    return data
    
def get_trj_torsion(api, stid, dir='./'):
    """

    :param api: 
    :param stid: 
    :param dir: 
    :return: 
    """
    filename = os.path.join(dir, 'trj_torsion.tar.gz')
    try:
        resp = api.get_structure_file(stid, file_type='trj_torsion')
        resp.raise_for_status()
    except HTTPError as e:
        resp.close()
        raise HTTPError(e)

    with open(filename, 'wb') as fh:
        fh.write(resp.content)
    resp.close()
    return filename
