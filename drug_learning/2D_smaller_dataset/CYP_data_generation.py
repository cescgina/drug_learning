import os
import numpy as np
import pandas as pd
from small_datasets_utils import col_to_array, get_features


def get_fingerprint(FINGERPRINT, PATH_DATA, PATH_FEAT):
    """Loads or generates (if the npy file doesn't exist in PATH_FEAT) fingerprints for each molecule in csv in PATH_DATA"""
    if FINGERPRINT == 'Morgan':
        dict = {'shared_set_npy': "shared_set_features.npy", 'only_2c9_npy': 'only_2c9_set_features.npy',
                'only_3a4_npy': 'only_3a4_set_features.npy'}
    elif FINGERPRINT == 'MACCS':
        dict = {'shared_set_npy': "shared_set_features_MACCS.npy", 'only_2c9_npy': 'only_2c9_set_features_MACCS.npy',
                'only_3a4_npy': 'only_3a4_set_features_MACCS.npy'}
    elif FINGERPRINT == 'RDKit':
        dict = {'shared_set_npy': "shared_set_features_RDKIT.npy", 'only_2c9_npy': 'only_2c9_set_features_RDKIT.npy',
                'only_3a4_npy': 'only_3a4_set_features_RDKIT.npy'}

    if os.path.exists(os.path.join(PATH_FEAT,  dict['shared_set_npy'])):
        features_shared = np.load(os.path.join(PATH_FEAT, dict['shared_set_npy']))
    else:
        features_shared = get_features(os.path.join(PATH_DATA, 'shared_set_cyp.sdf'), FINGERPRINT)
        np.save(os.path.join(PATH_FEAT, dict['shared_set_npy']), features_shared)

    if os.path.exists(os.path.join(PATH_FEAT, dict['only_2c9_npy'])):
        features_only_2c9 = np.load(os.path.join(PATH_FEAT, dict['only_2c9_npy']))
    else:
        features_only_2c9 = get_features(os.path.join(PATH_DATA, 'only_2c9_set_cyp.sdf'), FINGERPRINT)
        np.save(os.path.join(PATH_FEAT, dict['only_2c9_npy']), features_only_2c9)

    if os.path.exists(os.path.join(PATH_FEAT, dict['only_3a4_npy'])):
        features_only_3a4 = np.load(os.path.join(PATH_FEAT, dict['only_3a4_npy']))
    else:
        features_only_3a4 = get_features(os.path.join(PATH_DATA, 'only_3a4_set_cyp.sdf'), FINGERPRINT)
        np.save(os.path.join(PATH_FEAT, dict['only_3a4_npy']), features_only_3a4)

    return features_shared, features_only_2c9, features_only_3a4
