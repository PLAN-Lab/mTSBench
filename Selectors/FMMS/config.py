import os
import torch
import utils
# Xiaona 
dataset = 'ts'#'pmf'
random_state = 0
modelnum = 200


def get_rate():
    if dataset == 'pmf':
        rate = 0.1
    else:
        rate = 0.2
    return rate, dataset


def get_path():
    FEATURE_FILE = os.path.join("meta_features_FMMS.csv")
    TARGET_FILE = os.path.join("../results_val", "performance_matrix_vus_pr.csv")
    return FEATURE_FILE, TARGET_FILE


def get_para(train_params, params):
    return '%s_%s_%s_%s_%s_%s' % (train_params['optname'],
                                  train_params['lossname'],
                                  str(params['embedding_size']),
                                  str(train_params['batch']),
                                  str(train_params['epoch']),
                                  str(train_params['lr']))