import os
import sys # To load yaml file using terminal
import numpy as np
import numpy.testing as npt
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import small_datasets_utils_coupled as utils
from mordred import Calculator, descriptors
from imblearn.under_sampling import RandomUnderSampler


###########  INPUTS ##############

try:
    YAML_FILE = str(sys.argv[1]) # yaml file can be introduced as argunment using terminal
except IndexError:
    YAML_FILE = 'three_SVM_models_SARS2.yaml'


PATH = '../'  # "/gpfs/scratch/bsc72/bsc72665/" # For Power9
PATH_DATA = "../datasets/SARS2/"  # f"{PATH}}datasets/CYP/"
PATH_FEAT = 'features/'  # f"{PATH}}2D_smaller_dataset/features"

balance_dataset = False   # if true -> it randomly remove molecules from the biggest class with RandomUnderSampler()

dataset_size = 500
seed = 1
num_train_val_splits = 10 # Number of splits of the whole dataset
folds = 10 # Number of folds in cross-validation

load_clean_data = True  # Otherwise it will load data with NaN
remove_outliers = True
normalize_descriptors = True

# Option used when coupled_NN = True
data_selection = 'all' # 'only_coincidences' to only use the coincident predictions or 'all' if you want to use all
selection_criteria = 'majority' # 'unanimous' if both predictions must be true or "majority"
prob_threshold = 0.5
##################################

models = utils.read_yaml(f'{YAML_FILE}')

# Check if each model has the same length for the lists: percentile_des_ls and percentile_fp_ls
for name, model in models.items():
    assert len(model['percentile_des_ls']) == len(model['percentile_fp_ls']), f"The length of both percentile_list must be equal in model {model}"

# Check if all the models have the same length for the lists: percentile_des_ls and percentile_fp_ls
model1 = models[list(models.keys())[0]]
for name, model in models.items():
    assert len(model1['percentile_des_ls']) == len(model['percentile_des_ls']), f"The length of the percentile_list must be equal for both models {list(models.keys())[0]} and {model}"

del model1

fp_used = []
for name in models.keys():
    if models[name]['fp'] is not None:
        fp_used.append(models[name]['fp'])

fp_used = list(set(fp_used))

np.random.seed(seed)  # set a numpy seed
rand_num = np.random.randint(10000, size=num_train_val_splits)
print(f'Random seeds: {rand_num}')


if balance_dataset:
    PATH_SAVE = f"{PATH}2D_smaller_dataset/SARS/multycoupling/balanced_dataset/{YAML_FILE[:-5]}/{dataset_size}molec/"
elif not balance_dataset:
    PATH_SAVE = f"{PATH}2D_smaller_dataset/SARS/multycoupling/unbalanced_dataset/{YAML_FILE[:-5]}/{dataset_size}molec/"

PATH_CV_results = f"{PATH_SAVE}CV_results/"
PATH_confusion = f"{PATH_SAVE}confusion/"

if not os.path.exists(PATH_CV_results):
    os.makedirs(PATH_CV_results)

if not os.path.exists(PATH_confusion):
    os.makedirs(PATH_confusion)

print(f'Data is going to be saved at: {PATH_SAVE}')


threshold_activity = 20

features = {}
for fp in fp_used:
    features[fp] = np.load(os.path.join("..", "2D", "features", "SARS1_SARS2", f"{fp.lower()}.npy"))
#del fp_used

data = pd.read_csv(os.path.join(PATH_DATA, "dataset_cleaned_SARS1_SARS2_common.csv"))
#features = np.load(os.path.join("..", "2D", "features", "SARS1_SARS2", "morgan.npy"))
active = (data["activity_merged"] < threshold_activity).values.astype(int)


if os.path.exists(os.path.join(PATH_FEAT, "SARS1_SARS2_mordred.csv")):
    df_descriptors=pd.read_csv(os.path.join(PATH_FEAT, "SARS1_SARS2_mordred.csv"))
else:
    calc = Calculator(descriptors, ignore_3D=True)
    molecules_file = os.path.join(PATH_DATA, "molecules_cleaned_SARS1_SARS2_common.sdf")
    mols = [mol for mol in (Chem.SDMolSupplier(molecules_file))]
    df_descriptors = calc.pandas(mols)
    df_descriptors = df_descriptors.apply(pd.to_numeric, errors='coerce')
    df_descriptors = df_descriptors.dropna(axis=1)
    df_descriptors.to_csv(os.path.join(PATH_FEAT, "SARS1_SARS2_mordred.csv"))

if os.path.exists(os.path.join(PATH_FEAT, 'SARS1_SARS2_mordred_clean_no_outliers.csv')):
    descriptors_shared = pd.read_csv(os.path.join(PATH_FEAT, 'SARS1_SARS2_mordred_clean_no_outliers.csv'))
    loc = list(descriptors_shared['Molecule_index'])
    for fp in features.keys():
        features[fp] = features[fp][loc, :]
    active = active[loc]
    if 'Unnamed: 0' in descriptors_shared.columns:
        descriptors_shared = descriptors_shared.drop(['Unnamed: 0'], axis=1)
else:
    if remove_outliers:
        threshold = 3
        df_descriptors["Molecule_index"] = range(df_descriptors.shape[0])
        descriptors_shared = utils.drop_outliers(df_descriptors, threshold=threshold)
        if 'Unnamed: 0' in descriptors_shared.columns:
            descriptors_shared = descriptors_shared.drop(['Unnamed: 0'], axis=1)
        descriptors_shared.to_csv(os.path.join(PATH_FEAT, 'SARS1_SARS2_mordred_clean_no_outliers.csv'))
        loc = list(descriptors_shared['Molecule_index'])
        for fp in features.keys():
            features[fp] = features[fp][loc, :]
        active = active[loc]

norm_descriptors_shared = pd.DataFrame(normalize(descriptors_shared, norm='max', axis=0))
norm_descriptors_shared = np.asarray(norm_descriptors_shared).astype(
    np.float32)  # to avoid problems with the KFoldCrossValidation

dataset_size = min(dataset_size, descriptors_shared.shape[0])  # training + validation + test sets
train_size = int(0.75 * dataset_size) # training + validation sets
test_size = dataset_size-train_size

for fp in features.keys():
    assert features[fp].shape[0] == norm_descriptors_shared.shape[0]

models_names = list(models.keys())
metrics_tup = ('MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc')
df = pd.DataFrame(
    index=['val_MCC', 'val_MCC_std', 'val_acc', 'val_acc_std', 'val_recall', 'val_recall_std', 'val_precision',
           'val_precision_std', 'val_F1', 'val_F1_std', 'val_bal_acc', 'val_bal_acc_std',
           'test_MCC', 'test_MCC_std', 'test_acc', 'test_acc_std', 'test_recall', 'test_recall_std', 'test_precision',
           'test_precision_std', 'test_F1', 'test_F1_std', 'test_bal_acc', 'test_bal_acc_std',
           'val_agreement_percentage', 'val_agreement_percentage_std', 'test_agreement_percentage',
           'test_agreement_percentage_std'])

for p in range(len(models[models_names[0]]['percentile_des_ls'])):
    # The following dict, will contain not relevant info, so it is not filled with data
    metrics_split_dummy = {'MCC_train': [], 'MCC_val': [], 'MCC_test': [], 'acc_train': [], 'acc_val': [],
                           'acc_test': [], 'recall_train': [], 'recall_val': [], 'recall_test': [],
                           'precision_train': [], 'precision_val': [], 'precision_test': [], 'F1_train': [],
                           'F1_val': [], 'F1_test': [], 'balanced_acc_train': [],
                           'balanced_acc_val': [], 'balanced_acc_test': []}
    # The following dict, will contain the metrics for the coupled NN
    metrics_split = {'MCC_train': [], 'MCC_val': [], 'MCC_test': [], 'acc_train': [], 'acc_val': [], 'acc_test': [],
                     'recall_train': [], 'recall_val': [], 'recall_test': [],
                     'precision_train': [], 'precision_val': [], 'precision_test': [], 'F1_train': [], 'F1_val': [],
                     'F1_test': [], 'balanced_acc_train': [],
                     'balanced_acc_val': [], 'balanced_acc_test': [], 'agreement_percentage_train': [],
                     'agreement_percentage_val': [], 'agreement_percentage_test': []}

    """pred_train_dict, pred_val_dict, pred_test_dict = {}, {}, {}
    for i in range(folds):
        pred_train_dict.update({f'CV_{i}':{}})
        pred_val_dict.update({f'CV_{i}':{}})
    for key in models_names:
        pred_test_dict.update({key:{}})"""

    for split, seed in enumerate(rand_num):

        pred_train_dict, pred_val_dict, pred_test_dict = {}, {}, {}
        for i in range(folds):
            pred_train_dict.update({f'CV_{i}': {}})
            pred_val_dict.update({f'CV_{i}': {}})
        for key in models_names:
            pred_test_dict.update({key: {}})

        rus = RandomUnderSampler(random_state=seed)
        for name, model in models.items():
            print(
                f"--------------> STARTING SPLIT {split + 1} out of {num_train_val_splits}, MODEL {name} <--------------")

            if model['fp'] is not None:
                use_fingerprints = True
            else:
                use_fingerprints = False
            if model['des'] is not None:
                use_descriptors = True
            else:
                use_descriptors = False

            if use_fingerprints:
                train_val_data_fp, test_data_fp, train_val_labels_fp, test_labels_fp = train_test_split(
                    features[model['fp']], active,
                    train_size=train_size,
                    test_size=test_size,
                    stratify=active, random_state=seed)
                if balance_dataset:
                    train_val_data_fp, train_val_labels_fp = rus.fit_resample(train_val_data_fp, train_val_labels_fp)
            if use_descriptors:
                train_val_data_des, test_data_des, train_val_labels_des, test_labels_des = train_test_split(
                    norm_descriptors_shared, active,
                    train_size=train_size, test_size=test_size,
                    stratify=active, random_state=seed)
                if balance_dataset:
                    train_val_data_des, train_val_labels_des = rus.fit_resample(train_val_data_des,
                                                                                train_val_labels_des)
            if use_descriptors and use_fingerprints:
                npt.assert_array_equal(train_val_labels_fp, train_val_labels_des,
                                       err_msg='Train labels do not coincide between descriptors and fingerprints.')
                assert train_val_data_fp.shape[0] == train_val_data_des.shape[0]
                npt.assert_array_equal(test_labels_fp, test_labels_des,
                                       err_msg='Test labels do not coincide between descriptors and fingerprints.')
                assert test_data_fp.shape[0] == test_data_des.shape[0]

            best_features_split_fp, best_features_split_des = [], []
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

            # The following dict, will contain not relevant info, so it is removed after the CV loop
            metrics_fold_dummy = {'MCC_train': [], 'MCC_val': [], 'acc_train': [], 'acc_val': [], 'recall_train': [],
                                  'recall_val': [],
                                  'precision_train': [], 'precision_val': [], 'F1_train': [], 'F1_val': [],
                                  'balanced_acc_train': [],
                                  'balanced_acc_val': []}
            # The following dict, will contain the metrics for the coupled NN
            metrics_fold = {'MCC_train': [], 'MCC_val': [], 'acc_train': [], 'acc_val': [], 'recall_train': [],
                            'recall_val': [],
                            'precision_train': [], 'precision_val': [], 'F1_train': [], 'F1_val': [],
                            'balanced_acc_train': [],
                            'balanced_acc_val': [], 'agreement_percentage_train': [], 'agreement_percentage_val': []}

            if use_fingerprints and not use_descriptors:
                print('Only using fingerprints')
                for i, (train_index, val_index) in enumerate(skf.split(train_val_data_fp, train_val_labels_fp)):
                    train_data, train_labels, val_data, val_labels, data_fs_fp, _ = utils.get_train_val_split(
                        train_index, val_index, use_fingerprints,
                        use_descriptors, best_features_split_fp,
                        best_features_split_des, train_val_data_fp=train_val_data_fp,
                        train_val_labels_fp=train_val_labels_fp,
                        percentile_fp=model['percentile_fp_ls'][p],
                        train_val_data_des=None,
                        train_val_labels_des=None,
                        percentile_des=model['percentile_des_ls'][p])
                    _, pred_train, pred_val = utils.training_model_CV(model, train_data, train_labels, val_data,
                                                                      val_labels, metrics_fold, fold=i,
                                                                      metrics_ls=metrics_tup)
                    pred_train_dict[f'CV_{i}'].update({name: pred_train})
                    pred_val_dict[f'CV_{i}'].update({name: pred_val})
                    del pred_train, pred_val

            elif use_descriptors and not use_fingerprints:
                print('Only using descriptors')
                for i, (train_index, val_index) in enumerate(skf.split(train_val_data_des, train_val_labels_des)):
                    train_data, train_labels, val_data, val_labels, _, data_fs_des = utils.get_train_val_split(
                        train_index, val_index, use_fingerprints,
                        use_descriptors, best_features_split_fp,
                        best_features_split_des, train_val_data_fp=None,
                        train_val_labels_fp=None,
                        percentile_fp=model['percentile_fp_ls'][p],
                        train_val_data_des=train_val_data_des,
                        train_val_labels_des=train_val_labels_des,
                        percentile_des=model['percentile_des_ls'][p])
                    _, pred_train, pred_val = utils.training_model_CV(model, train_data, train_labels, val_data,
                                                                      val_labels, metrics_fold_dummy, fold=i,
                                                                      metrics_ls=metrics_tup)
                    pred_train_dict[f'CV_{i}'].update({name: pred_train})
                    pred_val_dict[f'CV_{i}'].update({name: pred_val})
                    del pred_train, pred_val

            elif use_descriptors and use_fingerprints:
                print('Using fingerprints and descriptors')
                for i, (train_index, val_index) in enumerate(skf.split(train_val_data_fp, train_val_labels_fp)):
                    train_data, train_labels, val_data, val_labels, data_fs_fp, data_fs_des = utils.get_train_val_split(
                        train_index, val_index, use_fingerprints,
                        use_descriptors, best_features_split_fp,
                        best_features_split_des, train_val_data_fp=train_val_data_fp,
                        train_val_labels_fp=train_val_labels_fp,
                        percentile_fp=model['percentile_fp_ls'][p],
                        train_val_data_des=train_val_data_des,
                        train_val_labels_des=train_val_labels_des,
                        percentile_des=model['percentile_des_ls'][p])
                    _, pred_train, pred_val = utils.training_model_CV(model, train_data, train_labels, val_data,
                                                                      val_labels, metrics_fold_dummy, fold=i,
                                                                      metrics_ls=metrics_tup)
                    pred_train_dict[f'CV_{i}'].update({name: pred_train})
                    pred_val_dict[f'CV_{i}'].update({name: pred_val})
                    del pred_train, pred_val

            if use_fingerprints:
                best_feat_CV_fp = utils.find_best_features(best_features_split_fp, data_fs_fp['train_data_fs'].shape[1])
                train_val_data_fp = np.concatenate([train_val_data_fp[train_index], train_val_data_fp[val_index]],
                                                   axis=0)

                assert train_val_data_fp.shape[0] == train_size  # train_size = train+val

                train_val_data_fs_fp = train_val_data_fp[:, best_feat_CV_fp]
                test_data_fp = test_data_fp[:, best_feat_CV_fp]

                if not use_descriptors:
                    labels_split = np.concatenate([train_labels, val_labels], axis=0)
                    train_val_data_fs = train_val_data_fs_fp
                    test_data = test_data_fp

            if use_descriptors:
                best_feat_CV_des = utils.find_best_features(best_features_split_des,
                                                            data_fs_des['train_data_fs'].shape[1])
                train_val_data_des = np.concatenate([train_val_data_des[train_index], train_val_data_des[val_index]],
                                                    axis=0)

                assert train_val_data_des.shape[0] == train_size  # train_size = train+val

                train_val_data_fs_des = train_val_data_des[:, best_feat_CV_des]
                test_data_des = test_data_des[:, best_feat_CV_des]

                if not use_fingerprints:
                    labels_split = np.concatenate([train_labels, val_labels], axis=0)
                    train_val_data_fs = train_val_data_fs_des
                    test_data = test_data_des

            if use_fingerprints and use_descriptors:
                labels_split = np.concatenate([train_labels, val_labels], axis=0)
                train_val_data_fs = np.concatenate([train_val_data_fs_fp, train_val_data_fs_des], axis=1)
                test_data = np.concatenate([test_data_fp, test_data_des], axis=1)
                test_labels = test_labels_fp

            if use_fingerprints:
                test_labels = test_labels_fp
                _, _, pred_test = utils.train_predict_test(model, train_val_data_fs, labels_split, test_data,
                                                           test_labels_fp, metrics_split_dummy, split=split)
            else:
                test_labels = test_labels_des
                _, _, pred_test = utils.train_predict_test(model, train_val_data_fs, labels_split, test_data,
                                                           test_labels_des, metrics_split_dummy, split=split)
            pred_test_dict[name] = pred_test

        print("====>> COMPUTING COUPLED PREDICTIONS <<====")
        pred_coupled_CV_train, coincidences_CV_train = utils.get_multicoupled_prediction(pred_train_dict,
                                                                                         selection=data_selection,
                                                                                         criteria=selection_criteria,
                                                                                         threshold=prob_threshold)
        pred_coupled_CV_val, coincidences_CV_val = utils.get_multicoupled_prediction(pred_val_dict,
                                                                                     selection=data_selection,
                                                                                     criteria=selection_criteria,
                                                                                     threshold=prob_threshold)
        pred_coupled_test, coincidences_test = utils.get_multicoupled_prediction(pred_test_dict,
                                                                                 selection=data_selection,
                                                                                 criteria=selection_criteria,
                                                                                 threshold=prob_threshold)
        del pred_train_dict, pred_val_dict, pred_test_dict

        for i in range(len(coincidences_CV_train)):
            agr_train = 100 * len(coincidences_CV_train[i]) / len(train_labels)
            agr_val = 100 * len(coincidences_CV_val[i]) / len(val_labels)

            if data_selection == 'all':
                print(f"-----> Train coupled fold {i} \n {agr_train}")
                dict_train = utils.print_metrics(pred_coupled_CV_train[i], train_labels, agr_percentage=agr_train)
                print(f"-----> Validaton coupled fold {i} \n {agr_val}")
                dict_val = utils.print_metrics(pred_coupled_CV_val[i], val_labels, agr_percentage=agr_val)
            elif data_selection == 'only_coincidences':
                print(f"-----> Train coupled fold {i} \n {agr_train}")
                dict_train = utils.print_metrics(pred_coupled_CV_train[i], train_labels[coincidences_CV_train[i]],
                                                 agr_percentage=agr_train)
                print(f"-----> Validaton coupled fold {i} \n {agr_val}")
                dict_val = utils.print_metrics(pred_coupled_CV_val[i], val_labels[coincidences_CV_val[i]],
                                               agr_percentage=agr_val)

            metrics_fold['agreement_percentage_train'].append(agr_train)
            metrics_fold['agreement_percentage_val'].append(agr_val)

            metrics_fold = utils.append_metrics_to_dict(metrics_fold, 'train', dict_train, 'val', dict_val,
                                                        metrics_ls=metrics_tup)

            del dict_train, dict_val, agr_train, agr_val

        agr_test = 100 * len(coincidences_test) / len(test_labels)
        dict_test = utils.print_metrics(pred_coupled_test, test_labels, agr_percentage=agr_test)

        # Adding test data to metrics_split
        metrics_split = utils.append_metrics_to_dict(metrics_split, 'test', dict_test, None, None,
                                                     metrics_ls=metrics_tup)
        # Adding train and val data to metrics_split
        metrics_split['agreement_percentage_test'].append(agr_test)

        metrics_split = utils.append_metrics_to_dict(metrics_split, 'train', metrics_fold, 'val', metrics_fold,
                                                     metrics_ls=metrics_tup + ('agreement_percentage',))

        utils.plot_results_CV(metrics_fold['MCC_train'], metrics_fold['MCC_val'], metrics_fold['acc_train'],
                              metrics_fold['acc_val'], metrics_fold['recall_train'], metrics_fold['recall_val'],
                              metrics_fold['precision_train'],
                              metrics_fold['precision_val'], metrics_fold['F1_train'], metrics_fold['F1_val'],
                              metrics_fold['balanced_acc_train'], metrics_fold['balanced_acc_val'], dict_test['acc'],
                              dict_test['MCC'],
                              dict_test['recall'], dict_test['precision'], dict_test['F1'], dict_test['balanced_acc'],
                              filename=f'{PATH_CV_results}percentile_position_ls_{p}_split{split}')

    utils.plot_results_split(metrics_split['MCC_train'], metrics_split['MCC_val'], metrics_split['acc_train'],
                             metrics_split['acc_val'], metrics_split['recall_train'], metrics_split['recall_val'],
                             metrics_split['precision_train'],
                             metrics_split['precision_val'], metrics_split['F1_train'], metrics_split['F1_val'],
                             metrics_split['balanced_acc_train'], metrics_split['balanced_acc_val'],
                             metrics_split['MCC_test'], metrics_split['acc_test'],
                             metrics_split['recall_test'], metrics_split['precision_test'], metrics_split['F1_test'],
                             metrics_split['balanced_acc_test'],
                             filename=f'{PATH_CV_results}Average_percentile_position_ls_{p}.png')

    df[f'percentile_position_ls_{p}'] = [np.nanmean(metrics_split['MCC_val']),
                                         np.nanstd(metrics_split['MCC_val']),
                                         np.nanmean(metrics_split['acc_val']),
                                         np.nanstd(metrics_split['acc_val']),
                                         np.nanmean(metrics_split['recall_val']),
                                         np.nanstd(metrics_split['recall_val']),
                                         np.nanmean(metrics_split['precision_val']),
                                         np.nanstd(metrics_split['precision_val']),
                                         np.nanmean(metrics_split['F1_val']),
                                         np.nanstd(metrics_split['F1_val']),
                                         np.nanmean(metrics_split['balanced_acc_val']),
                                         np.nanstd(metrics_split['balanced_acc_val']),
                                         np.nanmean(metrics_split['MCC_test']),
                                         np.nanstd(metrics_split['MCC_test']),
                                         np.nanmean(metrics_split['acc_test']),
                                         np.nanstd(metrics_split['acc_test']),
                                         np.nanmean(metrics_split['recall_test']),
                                         np.nanstd(metrics_split['recall_test']),
                                         np.nanmean(metrics_split['precision_test']),
                                         np.nanstd(metrics_split['precision_test']),
                                         np.nanmean(metrics_split['F1_test']),
                                         np.nanstd(metrics_split['F1_test']),
                                         np.nanmean(metrics_split['balanced_acc_test']),
                                         np.nanstd(metrics_split['balanced_acc_test']),
                                         np.nanmean(metrics_split['agreement_percentage_val']),
                                         np.nanstd(metrics_split['agreement_percentage_val']),
                                         np.nanmean(metrics_split['agreement_percentage_test']),
                                         np.nanstd(metrics_split['agreement_percentage_test'])]

    df.to_csv(f'{PATH_SAVE}metrics_average_multicoupling.csv',
              index=True)  # To read this file do -> df = pd.read_csv('path/metrics_average_diff_percentile.csv', index_col=0)
