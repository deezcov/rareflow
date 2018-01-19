from __future__ import absolute_import, division, print_function
import os
import itertools
import zipfile
import numpy as np
import scipy.sparse as sp
from rareflow.datasets.core import get_data_home, maybe_download_data, Dataset

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
    return int(uid), int(iid), int(rating), int(timestamp)


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

        uid = user_map.setdefault(uid, len(user_map) + 1)
        iid = item_map.setdefault(iid, len(item_map) + 1)

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
    pass


def _get_users_items(data):
    """
    Get user ids, item ids and the shape of interaction matrix
    :param data: Generator
        the raw data
    :return: 3-tuple: (uids, iids, (nrows, ncols))
    """
    uids = set()
    iids = set()

    for uid, iid, _, _ in data:
        uids.add(uid)
        iids.add(iid)

    uids = np.array(list(uids), dtype=np.int32)
    iids = np.array(list(iids), dtype=np.int32)

    return uids, iids, (max(uids), max(iids))


def _build_interactions_matrix(rows, cols, data, min_rating):
    """
    
    :param rows:
    :param cols:
    :param data:
    :param min_rating:
    :return:
    """