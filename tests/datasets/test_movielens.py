import pytest

from rareflow.datasets.movielens import fetch_data


def test_fetch_movielens():
    corpus_infos = {
        'movielens_100K': {'num_users': 943, 'num_items': 1682},
        'movielens_1M': {'num_users': 6040, 'num_items': 3706}
    }

    for name, infos in corpus_infos.iteritems():
        dataset = fetch_data(name=name)
        assert type(dataset).__name__ == 'Dataset'
        assert dataset.num_users == infos['num_users']
        assert dataset.num_items == infos['num_items']