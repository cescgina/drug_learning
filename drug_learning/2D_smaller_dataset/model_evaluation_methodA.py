from small_datasets_utils import *
import CYP_data_generation as CYP
import numpy as np
import numpy.testing as npt
import pandas as pd
from itertools import product
import os
from imblearn.under_sampling import RandomUnderSampler

FINGERPRINT = 'MACCS'
DESCRIPTOR = 'Mordred'
use_fingerprints = True  # if true -> fingerpornts are used.
use_descriptors = False  # if true -> descriptors are used.
#TODO caldrà posar opció per fer balanced o no
balance_dataset = False   # if true -> it randomly remove molecules from the biggest class with RandomUnderSampler()

do_combination = True # False to use percentil_com to choose the combinations. True to do all the combinations with percetile_descriptors_ls and percetile_fingerprint_ls

# Percentage to be kept
percetile_descriptors_ls = [10,20,40,60]
percetile_fingerprint_ls = [10,20,40,60]
# Select percentile combination. Used if do_combination=False
percentil_com = [(10,0)] # In each tuple it is indicated (percetile_fp, percentile_des)

dataset_size = 500  # training + validation sets
train_size = int(0.75 * dataset_size)
val_size = int(0.25 * dataset_size)

plot_distribution = True #TODO No cal?
seed = 1
num_train_val_splits = 2 # Number of splits of the whole dataset
folds = 2 # Number of folds in cross-validation

load_clean_data = True  # Otherwise it will load data with NaN
remove_outliers = True
normalize_descriptors = True

#Select the model to be evaluated
# TODO list of dicts containing all the models with all the hyperparameters. Then we could loop over the entries of list of models
# TODO hyperparameters could be list of lists, so then we could generate a list of dicts (each dict a model) -> Problem: both models must use the same fingerprint and descriptors.
layers_dimensions = [130, 130, 130, 130, 130, 130, 1]  # [256,256,256,256,256,256,256,256,1] # excluding the input layer
lr = 0.001  # 0.1
dropout = 0.2
optimizer = 'adam'  # 'sgd'
L2 = 0.001

np.random.seed(seed)  # set a numpy seed
rand_num = np.random.randint(10000, size=num_train_val_splits)

PATH = '../'#"/gpfs/scratch/bsc72/bsc72665/" # For Power9
PATH_DATA = "../datasets/CYP/" #f"{PATH}}datasets/CYP/"
PATH_FEAT = 'features/' # f"{PATH}}2D_smaller_dataset/features"
if balance_dataset:
    if use_fingerprints and use_descriptors:
        PATH_SAVE = f"{PATH}2D_smaller_dataset/method_A/{FINGERPRINT}_{DESCRIPTOR}/balanced_dataset/{dataset_size}molec/"
    elif use_fingerprints:
        PATH_SAVE = f"{PATH}2D_smaller_dataset/method_A/{FINGERPRINT}/balanced_dataset/{dataset_size}molec/"
    else:
        PATH_SAVE = f"{PATH}2D_smaller_dataset/method_A/{DESCRIPTOR}/balanced_dataset/{dataset_size}molec/"
elif not balance_dataset:
    if use_fingerprints and use_descriptors:
        PATH_SAVE = f"{PATH}2D_smaller_dataset/method_A/{FINGERPRINT}/unbalanced_dataset/{dataset_size}molec/"
    elif use_fingerprints:
        PATH_SAVE = f"{PATH}2D_smaller_dataset/method_A/{FINGERPRINT}/unbalanced_dataset/{dataset_size}molec/"
    else:
        PATH_SAVE = f"{PATH}2D_smaller_dataset/method_A/{DESCRIPTOR}/unbalanced_dataset/{dataset_size}molec/"
PATH_CV_results = f"{PATH_SAVE}CV_results/"

if not os.path.exists(PATH_CV_results):
    os.makedirs(PATH_CV_results)

if use_fingerprints and not use_descriptors:
    print(f"Only {FINGERPRINT} fingerprints are going to be used.")
elif use_descriptors and not use_fingerprints:
    print(f"Only {DESCRIPTOR} descriptors are going to be used.")
elif use_fingerprints and use_descriptors:
    print("Fingerprints and descriptors are going to be used (concatenated).")


shared_data = pd.read_csv(os.path.join(PATH_DATA, "shared_set_cyp.csv"))
labels_2c9 = (shared_data["p450-cyp2c9 Activity Outcome"] == "Active").values.astype(int)
labels_3a4 = (shared_data["p450-cyp3a4 Activity Outcome"] == "Active").values.astype(int)
testing_2c9_data = pd.read_csv(os.path.join(PATH_DATA, "only_2c9_set_cyp.csv"))
labels_testing_2c9 = (testing_2c9_data["p450-cyp2c9 Activity Outcome"] == "Active").values.astype(int)
testing_3a4_data = pd.read_csv(os.path.join(PATH_DATA, "only_3a4_set_cyp.csv"))
labels_testing_3a4 = (testing_3a4_data["p450-cyp3a4 Activity Outcome"] == "Active").values.astype(int)

smi_col_shared = col_to_array(shared_data, 'CanonicalSMILES')
smi_col_only2c9 = col_to_array(testing_2c9_data, 'CanonicalSMILES')
smi_col_only3a4 = col_to_array(testing_3a4_data, 'CanonicalSMILES')

# get_fingerprints is a function that has to be changed if we change the dataset
features_shared, features_only_2c9, features_only_3a4 = CYP.get_fingerprint(FINGERPRINT, PATH_DATA, PATH_FEAT)

if load_clean_data:
    if os.path.exists(os.path.join(PATH_FEAT, "shared_set_features_mordred_clean.npy")):
        descriptors_shared = pd.read_csv(os.path.join(PATH_FEAT, "shared_set_features_mordred_clean.npy")).drop(
            ['Unnamed: 0'], axis=1)
    else:
        descriptors_shared = get_descriptors(smi_col_shared, labels_2c9, clean_dataset=True,
                                             filename='shared_set_features_mordred_clean')
    if os.path.exists(os.path.join(PATH_FEAT, "only2c9_features_mordred_clean.npy")):
        descriptors_only2c9 = pd.read_csv(os.path.join(PATH_FEAT, "only2c9_features_mordred_clean.npy")).drop(
            ['Unnamed: 0'], axis=1)
    else:
        descriptors_only2c9 = get_descriptors(smi_col_only2c9, labels_testing_2c9, clean_dataset=True,
                                              filename='only2c9_features_mordred_clean')
    #if os.path.exists(os.path.join(PATH_FEAT, "only3a4_features_mordred_clean.npy")):
    #    descriptors_only3a4 = pd.read_csv(os.path.join(PATH_FEAT, "only3a4_features_mordred_clean.npy")).drop(
    #        ['Unnamed: 0'], axis=1)
    #else:
    #    descriptors_only3a4 = get_descriptors(smi_col_only2c9, labels_testing_3a4, clean_dataset=True,
    #                                          filename='only3a4_features_mordred_clean')

else:
    if os.path.exists(os.path.join(PATH_FEAT, "shared_set_features_mordred.npy")):
        descriptors_shared = pd.read_csv(os.path.join(PATH_FEAT, "shared_set_features_mordred.npy")).drop(
            ['Unnamed: 0'], axis=1)
    else:
        descriptors_shared = get_descriptors(smi_col_shared, labels_2c9, clean_dataset=False,
                                             filename='shared_set_features_mordred')
    if os.path.exists(os.path.join(PATH_FEAT, "only2c9_features_mordred.npy")):
        descriptors_only2c9 = pd.read_csv(os.path.join("features", "only2c9_features_mordred.npy")).drop(['Unnamed: 0'],
                                                                                                         axis=1)
    else:
        descriptors_only2c9 = get_descriptors(smi_col_only2c9, labels_testing_2c9, clean_dataset=False,filename='only2c9_features_mordred')
    #if os.path.exists(os.path.join(PATH_FEAT, "only3a4_features_mordred.npy")):
    #    descriptors_only3a4 = pd.read_csv(os.path.join("features", "only3a4_features_mordred.npy")).drop(['Unnamed: 0'],
    #                                                                                                     axis=1)
    #else:
    #    descriptors_only3a4 = get_descriptors(smi_col_only3a4, labels_testing_3a4, clean_dataset=False,filename='only3a4_features_mordred')

if load_clean_data:  # To get data with the same number of descriptors (columns in this case) for the shared_dataset and the only_2c9_dataset
    lst_shared_clean = list(descriptors_shared.columns.values)
    lst_only2c9_clean = list(descriptors_only2c9.columns.values)
    #lst_only3a4_clean = list(descriptors_only3a4.columns.values)
    common_elements = list(set(lst_shared_clean) & set(lst_only2c9_clean))

    lst_shared_clean = list(set(lst_shared_clean) - set(common_elements))
    lst_only2c9_clean = list(set(lst_only2c9_clean) - set(common_elements))
    #lst_only3a4_clean = list(set(lst_only3a4_clean) - set(common_elements))

    descriptors_shared = descriptors_shared.drop(lst_shared_clean, axis=1)
    descriptors_only2c9 = descriptors_only2c9.drop(lst_only2c9_clean, axis=1)
    #descriptors_only3a4 = descriptors_only3a4.drop(lst_only3a4_clean, axis=1)

assert descriptors_shared.shape[1] == descriptors_only2c9.shape[1]

if remove_outliers:
    threshold = 3
    if not os.path.exists(os.path.join(PATH_FEAT, "shared_set_features_mordred_clean_no_outliers.npy")):
        descriptors_shared = drop_outliers(descriptors_shared, threshold=threshold)
        descriptors_shared.to_csv(os.path.join(PATH_FEAT, 'shared_set_features_mordred_clean_no_outliers.npy'))
    descriptors_shared = pd.read_csv(os.path.join(PATH_FEAT, "shared_set_features_mordred_clean_no_outliers.npy"))
    loc = list(descriptors_shared['Unnamed: 0'])
    features_shared = features_shared[loc, :]
    descriptors_shared = descriptors_shared.drop(['Unnamed: 0'], axis=1)

    if not os.path.exists(os.path.join(PATH_FEAT, "only2c9_features_mordred_clean_no_outliers.npy")):
        descriptors_only2c9 = drop_outliers(descriptors_only2c9, threshold=threshold)
        descriptors_only2c9.to_csv(os.path.join(PATH_FEAT, 'only2c9_features_mordred_clean_no_outliers.npy'))
    descriptors_only2c9 = pd.read_csv(os.path.join(PATH_FEAT, "only2c9_features_mordred_clean_no_outliers.npy"))
    loc = list(descriptors_only2c9['Unnamed: 0'])
    features_only_2c9 = features_only_2c9[loc, :]
    descriptors_only2c9 = descriptors_only2c9.drop(['Unnamed: 0'], axis=1)

assert descriptors_shared.shape[1] == descriptors_only2c9.shape[1]

if remove_outliers and not descriptors_only2c9.shape[1] == descriptors_shared.shape[1]:
    lst_shared_clean = list(descriptors_shared.columns.values)
    lst_only2c9_clean = list(descriptors_only2c9.columns.values)
    common_elements = list(set(lst_shared_clean) & set(lst_only2c9_clean))

    lst_shared_clean = list(set(lst_shared_clean) - set(common_elements))
    lst_only2c9_clean = list(set(lst_only2c9_clean) - set(common_elements))

    descriptors_shared = descriptors_shared.drop(lst_shared_clean, axis=1)
    descriptors_only2c9 = descriptors_only2c9.drop(lst_only2c9_clean, axis=1)

    descriptors_shared.to_csv(os.path.join(PATH_FEAT, 'shared_set_features_mordred_clean_no_outliers.npy'))
    descriptors_only2c9.to_csv(os.path.join(PATH_FEAT, 'only2c9_features_mordred_clean_no_outliers.npy'))

    labels_2c9 = np.array(descriptors_shared['p450-cyp2c9 Activity Outcome'])
    labels_testing_2c9 = np.array(descriptors_only2c9['p450-cyp2c9 Activity Outcome'])

labels_2c9 = np.array(descriptors_shared['p450-cyp2c9 Activity Outcome'])
labels_testing_2c9 = np.array(descriptors_only2c9['p450-cyp2c9 Activity Outcome'])

assert descriptors_shared.shape[0] == labels_2c9.shape[0]
assert descriptors_only2c9.shape[0] == labels_testing_2c9.shape[0]
assert descriptors_shared.shape[1] == descriptors_only2c9.shape[1]

norm_descriptors_shared = pd.DataFrame(normalize(descriptors_shared, norm='max', axis=0))
norm_descriptors_only2c9 = pd.DataFrame(normalize(descriptors_only2c9, norm='max', axis=0))

# Drop the first column which contain the chemical activity against 2c9
norm_descriptors_shared = norm_descriptors_shared.drop([0], axis=1)
norm_descriptors_only2c9 = norm_descriptors_only2c9.drop([0], axis=1)

assert features_shared.shape[0] == norm_descriptors_shared.shape[0]
assert features_only_2c9.shape[0] == norm_descriptors_only2c9.shape[0]

norm_descriptors_shared = np.asarray(norm_descriptors_shared).astype(
    np.float32)  # to avoid problems with the KFoldCrossValidation
norm_descriptors_only2c9 = np.asarray(norm_descriptors_only2c9).astype(
    np.float32)  # to avoid problems with the KFoldCrossValidation

layers_dim = layers_dimensions.copy()

if do_combination:
    if use_fingerprints and use_descriptors:
        percentil_com= product(percetile_fingerprint_ls, percetile_descriptors_ls)
    elif use_fingerprints and not use_descriptors:
        percentil_com= product(percetile_fingerprint_ls, [0])
    elif use_descriptors and not use_fingerprints:
        percentil_com= product([0], percetile_descriptors_ls)
else:
    print("Using specified percetile combinations.")


df=pd.DataFrame(index=['val_MCC', 'val_MCC_std', 'val_acc' , 'val_acc_std' , 'val_recall', 'val_recall_std', 'val_precision', 'val_precision_std' , 'val_F1', 'val_F1_std' , 'val_bal_acc' , 'val_bal_acc_std',
                      'test_MCC', 'test_MCC_std', 'test_acc' , 'test_acc_std' , 'test_recall', 'test_recall_std', 'test_precision', 'test_precision_std' , 'test_F1', 'test_F1_std' , 'test_bal_acc' , 'test_bal_acc_std'])

# TODO we could loop over a list of models (ie. list of dict)
for percetile_fingerprint, percetile_descriptors in percentil_com:
    MCCs_train_split, MCCs_val_split, MCCs_test_split = [], [], []
    accs_train_split, accs_val_split, accs_test_split = [], [], []
    recall_train_split, recall_val_split, recall_test_split = [], [], []
    precision_train_split, precision_val_split, precision_test_split = [], [], []
    F1_train_split, F1_val_split, F1_test_split = [], [], []
    balanced_acc_train_split, balanced_acc_val_split, balanced_acc_test_split = [], [], []

    for split, seed in enumerate(rand_num):
        rus = RandomUnderSampler(random_state=seed)  # NEW! To do undersampling -> Balance the dataset.
        print(f"--------------> STARTING SPLIT {split + 1} out of {num_train_val_splits} <--------------")
        if use_fingerprints:
            train_val_data_fp, _, train_val_labels_fp, _ = train_test_split(features_shared, labels_2c9,
                                                                            train_size=dataset_size, test_size=2,
                                                                            stratify=labels_2c9, random_state=seed)
            if balance_dataset:
                train_val_data_fp, train_val_labels_fp = rus.fit_resample(train_val_data_fp, train_val_labels_fp)
        if use_descriptors:
            train_val_data_des, _, train_val_labels_des, _ = train_test_split(norm_descriptors_shared, labels_2c9,
                                                                              train_size=dataset_size, test_size=2,
                                                                              stratify=labels_2c9, random_state=seed)
            if balance_dataset:
                train_val_data_des, train_val_labels_des = rus.fit_resample(train_val_data_des, train_val_labels_des)
        if use_descriptors and use_fingerprints:
            npt.assert_array_equal(train_val_labels_fp, train_val_labels_des,
                                   err_msg='Train labels do not coincide between descriptors and fingerprints.')
            assert train_val_data_fp.shape[0] == train_val_data_des.shape[0]

        best_features_split_fp, best_features_split_des = [], []
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        MCCs_train, MCCs_val = [], []
        accs_train, accs_val = [], []
        recall_train, recall_val = [], []
        precision_train, precision_val = [], []
        F1_train, F1_val = [], []
        balanced_acc_train, balanced_acc_val = [], []

        for i, (train_index, val_index) in enumerate(skf.split(train_val_data_fp, train_val_labels_fp)):
            if use_fingerprints:
                train_data_fp, val_data_fp = train_val_data_fp[train_index], train_val_data_fp[val_index]
                train_labels_fp, val_labels_fp = train_val_labels_fp[train_index], train_val_labels_fp[val_index]
                data_fs_fp, best_fold_fp = get_best_features(train_data_fp, train_labels_fp, val_data_fp,
                                                             percetile_fingerprint)
                best_features_split_fp.extend(list(best_fold_fp))

                assert train_data_fp.shape[1] == val_data_fp.shape[1]
                assert train_data_fp.shape[0] == train_labels_fp.shape[0]
                assert val_data_fp.shape[0] == val_labels_fp.shape[0]
                assert best_fold_fp.shape[0] == data_fs_fp['train_data_fs'].shape[1]

                if not use_descriptors:  # to avoid conditional statements
                    train_data, val_data = data_fs_fp['train_data_fs'], data_fs_fp['val_data_fs']
                    train_labels, val_labels = train_labels_fp, val_labels_fp

            if use_descriptors:
                train_data_des, val_data_des = train_val_data_des[train_index], train_val_data_des[val_index]
                train_labels_des, val_labels_des = train_val_labels_des[train_index], train_val_labels_des[val_index]
                data_fs_des, best_fold_des = get_best_features(train_data_des, train_labels_des, val_data_des,
                                                               percetile_descriptors)
                best_features_split_des.extend(list(best_fold_des))

                assert train_data_des.shape[1] == val_data_des.shape[1]
                assert train_data_des.shape[0] == train_labels_des.shape[0]
                assert val_data_des.shape[0] == val_labels_des.shape[0]
                assert best_fold_des.shape[0] == data_fs_des['train_data_fs'].shape[1]

                if not use_fingerprints:
                    train_data, val_data = data_fs_des['train_data_fs'], data_fs_des['val_data_fs']
                    train_labels, val_labels = train_labels_des, val_labels_des

            if use_descriptors and use_fingerprints:
                npt.assert_array_equal(train_labels_fp, train_labels_des,
                                       err_msg='Train labels do not coincide between descriptors and fingerprints.')
                npt.assert_array_equal(val_labels_fp, val_labels_des,
                                       err_msg='Validation labels do not coincide between descriptors and fingerprints.')
                assert data_fs_fp['train_data_fs'].shape[0] == data_fs_des['train_data_fs'].shape[0]
                assert data_fs_fp['val_data_fs'].shape[0] == data_fs_des['val_data_fs'].shape[0]

                train_labels, val_labels = train_labels_des, val_labels_des

                train_data = np.concatenate([data_fs_fp['train_data_fs'], data_fs_des['train_data_fs']], axis=1)
                val_data = np.concatenate([data_fs_fp['val_data_fs'], data_fs_des['val_data_fs']], axis=1)

            print(f"-----------> Calculating split {split + 1}/{num_train_val_splits} fold {i + 1}/{folds} <----------")

            if not layers_dim[0] == train_data.shape[1]:
                layers_dim.insert(0, train_data.shape[1])
                print(type(layers_dim))

            class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
            class_weight = {0: class_weights[0], 1: class_weights[1]}

            model = generate_model(layers_dim, lr, dropout, optimizer, L2)
            history = model.fit(train_data, train_labels, epochs=10, verbose=2,
                                class_weight=class_weight)  # , validation_data = (val_data, val_labels))

            pred_train = model.predict(train_data)
            dict_train = print_metrics(pred_train, train_labels)
            train_acc, train_mcc, train_recall, train_precision, train_f1, train_balanced_acc = dict_train['accuracy'], \
                                                                                                dict_train['mcc'], \
                                                                                                dict_train['recall'], \
                                                                                                dict_train['precision'], \
                                                                                                dict_train['f1'], \
                                                                                                dict_train[
                                                                                                    'balanced_accuracy']

            pred_val = model.predict(val_data)
            print(f"---> Validation set fold {i + 1}")
            dict_val = print_metrics(pred_val, val_labels)
            val_acc, val_mcc, val_recall, val_precision, val_f1, val_balanced_acc = dict_val['accuracy'], dict_val['mcc'], \
                                                                                    dict_val['recall'], dict_val[
                                                                                        'precision'], dict_val['f1'], \
                                                                                    dict_val['balanced_accuracy']

            MCCs_train.append(train_mcc), MCCs_val.append(val_mcc)
            accs_train.append(train_acc), accs_val.append(val_acc)
            recall_train.append(train_recall), recall_val.append(val_recall)
            precision_train.append(train_precision), precision_val.append(val_precision)
            F1_train.append(train_f1), F1_val.append(val_f1)
            balanced_acc_train.append(train_balanced_acc), balanced_acc_val.append(val_balanced_acc)

        if use_fingerprints:
            best_feat_CV_fp = find_best_features(best_features_split_fp, data_fs_fp['train_data_fs'].shape[1])
            train_val_data_fp = np.concatenate([train_data_fp, val_data_fp], axis=0)

            assert train_val_data_fp.shape[0] == train_data_fp.shape[0] + val_data_fp.shape[0]
            assert train_val_data_fp.shape[1] == train_data_fp.shape[1]

            train_val_data_fs_fp = train_val_data_fp[:, best_feat_CV_fp]
            test_data_fp = features_only_2c9[:, best_feat_CV_fp]

            if not use_descriptors:
                labels_split = np.concatenate([train_labels_fp, val_labels_fp], axis=0)
                train_val_data_fs = train_val_data_fs_fp
                test_data = test_data_fp

        if use_descriptors:
            best_feat_CV_des = find_best_features(best_features_split_des, data_fs_des['train_data_fs'].shape[1])
            train_val_data_des = np.concatenate([train_data_des, val_data_des], axis=0)

            assert train_val_data_des.shape[0] == train_data_des.shape[0] + val_data_des.shape[0]
            assert train_val_data_des.shape[1] == train_data_des.shape[1]

            train_val_data_fs_des = train_val_data_des[:, best_feat_CV_des]
            test_data_des = norm_descriptors_only2c9[:, best_feat_CV_des]

            if not use_fingerprints:
                labels_split = np.concatenate([train_labels_des, val_labels_des], axis=0)
                train_val_data_fs = train_val_data_fs_des
                test_data = test_data_des

        if use_fingerprints and use_descriptors:
            labels_split = np.concatenate([train_labels_fp, val_labels_fp], axis=0)
            train_val_data_fs = np.concatenate([train_val_data_fs_fp, train_val_data_fs_des], axis=1)
            test_data = np.concatenate([test_data_fp, test_data_des], axis=1)

        class_weights = compute_class_weight('balanced', np.unique(labels_split), labels_split)
        class_weight = {0: class_weights[0], 1: class_weights[1]}

        model = generate_model(layers_dim, lr, dropout, optimizer, L2)
        history = model.fit(train_val_data_fs, labels_split, epochs=10, verbose=2, class_weight=class_weight)

        pred_test = model.predict(test_data)
        dict_test = print_metrics(pred_test, labels_testing_2c9)
        test_acc, test_mcc, test_recall, test_precision, test_f1, test_balanced_acc = dict_test['accuracy'], dict_test[
            'mcc'], dict_test['recall'], dict_test['precision'], dict_test['f1'], dict_test['balanced_accuracy']

        print(f"----> Test set split {split + 1}")

        print(f"Computing test for split {split} out of {len(rand_num)}")
        plot_results_CV(MCCs_train, MCCs_val, accs_train, accs_val, recall_train, recall_val, precision_train,
                        precision_val, F1_train, F1_val, balanced_acc_train, balanced_acc_val, test_acc, test_mcc,
                        test_recall, test_precision, test_f1, test_balanced_acc,
                        filename=f'{PATH_CV_results}{percetile_fingerprint}-{percetile_descriptors}_fp-des_split{split + 1}')

        MCCs_train_split.extend(MCCs_train), MCCs_val_split.extend(MCCs_val), MCCs_test_split.append(test_mcc)
        accs_train_split.extend(accs_train), accs_val_split.extend(accs_val), accs_test_split.append(test_acc)
        recall_train_split.extend(recall_train), recall_val_split.extend(recall_val), recall_test_split.append(test_recall)
        precision_train_split.extend(precision_train), precision_val_split.extend(
            precision_val), precision_test_split.append(test_precision)
        F1_train_split.extend(F1_train), F1_val_split.extend(F1_val), F1_test_split.append(test_f1)
        balanced_acc_train_split.extend(balanced_acc_train), balanced_acc_val_split.extend(
            balanced_acc_val), balanced_acc_test_split.append(test_balanced_acc)

    plot_results_split(MCCs_train_split, MCCs_val_split, accs_train_split, accs_val_split, recall_train_split,
                       recall_val_split, precision_train_split, precision_val_split, F1_train_split, F1_val_split,
                       balanced_acc_train_split, balanced_acc_val_split, MCCs_test_split, accs_test_split,
                       recall_test_split, precision_test_split, F1_test_split, balanced_acc_test_split,
                      filename=f'{PATH_CV_results}Average_{percetile_fingerprint}-{percetile_descriptors}_fp-des.png')

    plot_results_split(MCCs_train_split, MCCs_val_split, accs_train_split, accs_val_split, recall_train_split, recall_val_split, precision_train_split, precision_val_split, F1_train_split, F1_val_split, balanced_acc_train_split, balanced_acc_val_split, MCCs_test_split, accs_test_split, recall_test_split, precision_test_split, F1_test_split, balanced_acc_test_split)
    df[f'{percetile_fingerprint}:{percetile_descriptors}'] = [np.nanmean(MCCs_val_split), np.nanstd(MCCs_val_split), np.nanmean(accs_val_split) , np.nanstd(accs_val_split) , np.nanmean(recall_val_split), np.nanstd(recall_val_split), np.nanmean(precision_val_split), np.nanstd(precision_val_split) , np.nanmean(F1_val_split), np.nanstd(F1_val_split) , np.nanmean(balanced_acc_val_split) , np.nanstd(balanced_acc_val_split),
                                                              np.nanmean(MCCs_test_split), np.nanstd(MCCs_test_split), np.nanmean(accs_test_split) , np.nanstd(accs_test_split) , np.nanmean(recall_test_split), np.nanstd(recall_test_split), np.nanmean(precision_test_split), np.nanstd(precision_test_split) , np.nanmean(F1_test_split), np.nanstd(F1_test_split) , np.nanmean(balanced_acc_test_split) , np.nanstd(balanced_acc_test_split)]
    df.to_csv(f'{PATH_SAVE}metrics_average_diff_percentile.csv', index=True) #To read this file do -> df = pd.read_csv('path/metrics_average_diff_percentile.csv', index_col=0)
