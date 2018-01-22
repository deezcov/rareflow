from __future__ import absolute_import, division, print_function

import os
import logging
import requests
from tqdm import tqdm
import sys


class Dataset(object):
    """
    The Dataset object contains (at a minimum) pair of user-item interactions.
    It can be enriched by other informations:
    - For user: user infos
    - For item: features coming from different modalities (e.g. audio, text)
    """

    def __init__(self, name, user_ids, item_ids, interactions):
        super(Dataset, self).__init__()
        self.name = name
        self._user_ids = user_ids
        self._item_ids = item_ids
        self._interactions = interactions

    @property
    def user_ids(self):
        return self._user_ids

    @property
    def item_ids(self):
        return self._item_ids

    @property
    def interactions(self):
        return self._interactions

    @property
    def num_users(self):
        return len(self._user_ids)

    @property
    def num_items(self):
        return len(self._item_ids)

    @property
    def num_interactions(self):
        return self._interactions.count_nonzero()

    def __repr__(self):
        return '<Dataset %s has %d users x %d items x %d interactions>' % \
               (self.name, self.num_users, self.num_items, self.num_interactions)

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """
        return self._interactions.tocsr()


def get_data_home(data_home=None):
    """
    Return home path of RaReFlow. In the case where the directory
    data_home does not exist, it is created automatically.
    :param data_home: str | None
        The home path of DoReMi.
    :return: str
        The data home path
    """
    if data_home is None:
        data_home = os.path.join(os.path.expanduser('~'), '.rareflow')

    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home


def maybe_download_data(url, data_home=None, download_if_missing=True, chunk_size=1024*1024):
    """
    Download data from url to the local destination
    :param url: str
        The url where the dataset can be found.
    :param data_home: str
        The data home path
    :param download_if_missing: boolean
        Should download file if it does not exist in the data home
    :param chunk_size: int
        The downloaded chunk size
    :return: str
        The path point to downloaded data
    """
    if data_home is None:
        data_home = get_data_home()

    file_path = os.path.join(data_home, url[url.rfind('/') + 1:])
    if not os.path.exists(file_path):
        if download_if_missing:
            logging.info('Download %s from url %s' % (file_path, url))
            req = requests.get(url, stream=True)
            req.raise_for_status()

            # total size in bytes.
            total_size = int(req.headers.get('content-length', 0))

            with open(file_path, 'wb') as f:
                for chunk in tqdm(req.iter_content(chunk_size=chunk_size),
                                  total=int(total_size / chunk_size),
                                  unit='M',
                                  unit_scale=True,
                                  file=sys.stdout):
                    if chunk:
                        f.write(chunk)
            logging.info('Finish downloading %s' % file_path)
        else:
            raise IOError('Missing dataset %s' % file_path)
    else:
        logging.info('Dataset %s already exists!' % file_path)

    return file_path


