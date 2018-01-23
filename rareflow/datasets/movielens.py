from __future__ import absolute_import, division, print_function
import os
import itertools
import zipfile
import numpy as np
import scipy.sparse as sp
from rareflow.datasets.core import maybe_download_data, Dataset

_MOVIELENS_CONFIG = {
    'URL_PREFIX': 'http://files.grouplens.org/datasets/movielens/',
    'URL_100K': 'ml-100k.zip',
    'URL_1M': 'ml-1m.zip',
    'URL_10M': 'ml-10m.zip',
    'URL_20M': 'ml-20m.zip',
    'CORPUS_100K': 'movielens_100K',
    'CORPUS_1M': 'movielens_1M',
    'CORPUS_10M': 'movielens_10M',
    'CORPUS_20M': 'movielens_20M'
}


def _read_archive_data(path, archive_path):
    """
    Read the archive file line by line
    :param path: str
        The archive file path
    :param archive_path: str
        The internal path inside archive
    :return: A generator that emit each line at a time
    """
    with zipfile.ZipFile(path) as zf:
        with zf.open(archive_path) as file:
            for line in file:
                yield line.decode('utf-8')


def _parse_line(line, separator='::'):
    """
    Parse a line in dataset
    :param line: str
        A line in dataset (utf-8)
    :param separator: str | '::'
        Seperator between fields
    :return: A generator of 4-tuple (user_id, item_id, rating, timestamp)
    """
    uid, iid, rating, timestamp = line.split(separator)
    return int(uid) - 1, int(iid) - 1, int(rating), int(timestamp)


def _make_contiguous(raw_data, separator):
    """
    Mapping original user, item id to contiguous ids space
    :param raw_data: Generator
        The raw data generator
    :param separator: str
        The separator between fields
    :return: Generator
        Generator of 4-tuple (uid, iid, rating, timestamp) with
        contiguous user, item ids
    """
    user_map = {}
    item_map = {}

    for line in raw_data:
        uid, iid, rating, timestamp = _parse_line(line, separator)

        uid = user_map.setdefault(uid, len(user_map))
        iid = item_map.setdefault(iid, len(item_map))

        yield uid, iid, rating, timestamp


def _movielens_100K_generator():
    """
    Get movielens 100K dataset generator

    The original file has a tab separated list of: user id | item id | rating | timestamp.
    :return: Generator
        A generator that yield 4-tuple (uid, iid, rating, timestamp)
    """
    zip_path = maybe_download_data(_MOVIELENS_CONFIG['URL_PREFIX'] + _MOVIELENS_CONFIG['URL_100K'])
    archive_path = os.path.join('ml-100k', 'u.data')

    for line in _read_archive_data(zip_path, archive_path):
        yield _parse_line(line, separator='\t')


def _movielens_1M_generator():
    """
    Get movielens 1M dataset generator
    The original file has a :: separator: user id::item id::rating::timestamp.
    :return: Generator
        A generator that yield 4-tuple (uid, iid, rating, timestamp)
    """
    zip_path = maybe_download_data(_MOVIELENS_CONFIG['URL_PREFIX'] + _MOVIELENS_CONFIG['URL_1M'])
    archive_path = os.path.join('ml-1m', 'ratings.dat')

    data = _read_archive_data(zip_path, archive_path)

    for line in _make_contiguous(data, separator='::'):
        yield line


def _movielens_10M_generator():
    """
    Get movielens 10M dataset generator
    The original file has a :: separator: user id::item id::rating::timestamp.
    :return: Generator
        A generator that yield 4-tuple (uid, iid, rating, timestamp)
    """
    zip_path = maybe_download_data(_MOVIELENS_CONFIG['URL_PREFIX'] + _MOVIELENS_CONFIG['URL_10M'])
    archive_path = os.path.join('ml-10M100K', 'ratings.dat')

    data = _read_archive_data(zip_path, archive_path)

    for line in _make_contiguous(data, separator='::'):
        yield line


def _movielens_20M_generator():
    """
    Get movielens 20M dataset generator
    The original file has a :: separator: user id::item id::rating::timestamp.
    :return: Generator
        A generator that yield 4-tuple (uid, iid, rating, timestamp)
    """
    zip_path = maybe_download_data(_MOVIELENS_CONFIG['URL_PREFIX'] + _MOVIELENS_CONFIG['URL_20M'])
    archive_path = os.path.join('ml-20m', 'ratings.csv')

    data = itertools.islice(_read_archive_data(zip_path, archive_path), 1, None)

    for line in _make_contiguous(data, separator='::'):
        yield line


def _get_users_items(data):
    """
    Get user ids, item ids and the shape of interaction matrix
    :param data: Generator
        the raw data
    :return: 3-tuple: (uids, iids, (nrows, ncols))
    """
    uids = set()
    iids = set()

    # convert user_id and item_id to zero-based
    for uid, iid, _, _ in data:
        uids.add(uid)
        iids.add(iid)

    uids = np.array(list(uids), dtype=np.int32)
    iids = np.array(list(iids), dtype=np.int32)

    return uids, iids, (max(uids) + 1, max(iids) + 1)


def _build_interactions_matrix(rows, cols, data, min_rating):
    """
    Build the interaction matrix from ratings
    :param rows: int
        Number of rows
    :param cols: int
        Number of columns
    :param data: 4-tuple
        Tuple contains uid, iid, rating, timestamp
    :param min_rating: int
        The minimum rating to consider a positive/negative interaction
    :return: The interaction matrix in COO format
    """
    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, timestamp in data:
        if rating >= min_rating:
            mat[uid, iid] = 1.0

    return mat.tocoo()


_MOVIELENS_GENERATORS = {
    _MOVIELENS_CONFIG['CORPUS_100K']: _movielens_100K_generator,
    _MOVIELENS_CONFIG['CORPUS_1M']: _movielens_1M_generator,
    _MOVIELENS_CONFIG['CORPUS_10M']: _movielens_10M_generator,
    _MOVIELENS_CONFIG['CORPUS_20M']: _movielens_20M_generator
}


def fetch_data(name, min_rating=4):
    """
    Fetch movielens data
    :param name: str
        Name of the dataset to be fetched
    :param min_rating: int
        The minimum rating to consider a positive/negative interaction
    :return: The dataset object
    """
    if name not in _MOVIELENS_GENERATORS:
        raise ValueError('RaReFlow does not support %s dataset' % name)

    # get number of users and items
    uids, iids, (num_users, num_items) = _get_users_items(_MOVIELENS_GENERATORS[name]())

    # get interaction matrix
    interactions = _build_interactions_matrix(num_users, num_items, _MOVIELENS_GENERATORS[name](), min_rating)

    return Dataset(name, uids, iids, interactions)
