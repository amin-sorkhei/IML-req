__author__ = 'sorkhei'

import os
import numpy as np


def read_movie_lens_data():
    """
    :return:
    ratings: a numpy array where each rows is (userID, itemID, rating, timestamp)
    items: a python dictionary where keys are integer ids and values are name of the movie
    userids: numpy array 1:943
    itemids: numpy array 1:1682
    """
    if 'data' not in os.listdir(os.getcwd()):
        exit('data directory not found')

    data_dir = os.path.join(os.getcwd(), 'data')
    data_dir_content = os.listdir(data_dir)

    if 'u.data' not in data_dir_content or 'u.item' not in data_dir_content:
        exit('required files not found')

    ratings = np.loadtxt(os.path.join(data_dir, 'u.data'), dtype=np.int)

    items = open(os.path.join(data_dir, 'u.item')).readlines()
    items_dictionary = dict([(int(item.split('|')[0]), item.split('|')[1]) for item in items])

    user_ids = np.arange(1, 943)
    item_ids = np.arange(1, 1682)

    return ratings, items_dictionary, user_ids, item_ids