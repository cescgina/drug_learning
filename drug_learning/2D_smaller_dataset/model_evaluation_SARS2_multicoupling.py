import os
import argparse
import numpy as np
import numpy.testing as npt
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import small_datasets_utils_coupled as utils
from mordred import Calculator, descriptors
from imblearn.under_sampling import RandomUnderSampler


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate perfomance of ensemble models using multiple splits and cross validation")
    parser.add_argument('controlFile', type=str, default='three_SVM_models_SARS2.yaml', help="File containing the model information")
    parser.add_argument('--nfolds', type=int, default=10, help="Number of folds in cross validation")
    parser.add_argument('--nsplits', type=int, default=10, help="Number of splits in the training data")
    parser.add_argument('--seed', default=1, help="Seed for the random number generator")
    parser.add_argument('--selection_criteria', type=str, default="unanimous", help="Criteria to classify according to ensemble voting, options are unanimous (all must predict positive to be positive) and majority (the majority must predict positive to be positive)")
    arg = parser.parse_args()
    return arg.controlFile, arg.seed, arg.nsplits, arg.nfolds


###########  INPUTS ##############
YAML_FILE, seed, num_train_val_splits, folds, selection_criteria = parse_arguments()
PATH = '../'  # "/gpfs/scratch/bsc72/bsc72665/" # For Power9
PATH_DATA = "../datasets/SARS2/"  # f"{PATH}}datasets/CYP/"
PATH_FEAT = 'features/'  # f"{PATH}}2D_smaller_dataset/features"
balance_dataset = False   # if true -> it randomly remove molecules from the biggest class with RandomUnderSampler()
dataset_size = 500
remove_outliers = True

# Option used when coupled_NN = True
data_selection = 'all' # 'only_coincidences' to only use the coincident predictions or 'all' if you want to use all
prob_threshold = 0.5
##################################

models = utils.read_yaml(f'{YAML_FILE}')

# Check if each model has the same length for the lists: percentile_des_ls and percentile_fp_ls
for name, model in models.items():
    if model['feat2'] is not None:
        assert len(model['percentile_feat1']) == len(model['percentile_feat2']), f"The length of both percentile_list must be equal in model {model}"

# Check if all the models have the same length for the lists: percentile_des_ls and percentile_fp_ls
model1 = models[list(models.keys())[0]]
for name, model in models.items():
    assert len(model1['percentile_feat1']) == len(model['percentile_feat1']), f"The length of the percentile_list must be equal for both models {model1} and {model}"

del model1

np.random.seed(seed)  # set a numpy seed
rand_num = np.random.randint(10000, size=num_train_val_splits)
print(f'Random seeds: {rand_num}')


threshold_activity = 20

data = pd.read_csv(os.path.join(PATH_DATA, "dataset_cleaned_SARS1_SARS2_common.csv"))
active = (data["activity_merged"] < threshold_activity).values.astype(int)

features = utils.load_SARS_features(models, active, PATH_DATA, PATH_FEAT, remove_outliers=True)

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

dataset_size = min(dataset_size, features['Mordred'].shape[0])  # training + validation + test sets
train_size = int(0.75 * dataset_size) # training + validation sets
test_size = dataset_size-train_size

for fp in features:
    assert features[fp].shape[0] == features['Mordred'].shape[0]

models_names = list(models.keys())
metrics_tup = ('MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc')
df = pd.DataFrame(
    index=['val_MCC', 'val_MCC_std', 'val_acc', 'val_acc_std', 'val_recall', 'val_recall_std', 'val_precision',
           'val_precision_std', 'val_F1', 'val_F1_std', 'val_bal_acc', 'val_bal_acc_std',
           'test_MCC', 'test_MCC_std', 'test_acc', 'test_acc_std', 'test_recall', 'test_recall_std', 'test_precision',
           'test_precision_std', 'test_F1', 'test_F1_std', 'test_bal_acc', 'test_bal_acc_std',
           'val_agreement_percentage', 'val_agreement_percentage_std', 'test_agreement_percentage',
           'test_agreement_percentage_std'])

for p in range(len(models[models_names[0]]['percentile_feat1'])):
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

            if model['feat2'] is not None:
                use_feat2 = True
                print('feat2 will be concatenated with feat1')
            else:
                use_feat2 = False
                print('Only using feat1')

            train_val_data_1, test_data_1, train_val_labels_1, test_labels_1 = train_test_split(
                features[model['feat1']], active,
                train_size=train_size,
                test_size=test_size,
                stratify=active, random_state=seed)

            percentile_feat1 = model['percentile_feat1'][p]
            if balance_dataset:
                train_val_data_1, train_val_labels_1 = rus.fit_resample(train_val_data_1, train_val_labels_1)

            if not use_feat2:  # To avoid conditional statement using utils.get_train_val_split()
                best_features_split_1, best_features_split_2 = [], None
                train_val_data_2, train_val_labels_2 = None, None
                percentile_feat2 = None

            else:
                train_val_data_2, test_data_2, train_val_labels_2, test_labels_2 = train_test_split(
                    features[model['feat2']], active,
                    train_size=train_size, test_size=test_size,
                    stratify=active, random_state=seed)
                if balance_dataset:
                    train_val_data_2, train_val_labels_2 = rus.fit_resample(train_val_data_2, train_val_labels_2)

                npt.assert_array_equal(train_val_labels_1, train_val_labels_2,
                                       err_msg='Train labels do not coincide between feat1 and feat2.')
                assert train_val_data_1.shape[0] == train_val_data_2.shape[0]
                npt.assert_array_equal(test_labels_1, test_labels_2,
                                       err_msg='Test labels do not coincide between feat1 and feat2.')
                assert test_data_1.shape[0] == test_data_2.shape[0]

                best_features_split_1, best_features_split_2 = [], []
                percentile_feat2 = model['percentile_feat2'][p]

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

            for i, (train_index, val_index) in enumerate(skf.split(train_val_data_1, train_val_labels_1)):
                train_data, train_labels, val_data, val_labels, data_fs_1, data_fs_2 = utils.get_train_val_split(
                    train_index, val_index, best_features_split_1,
                    train_val_data_1=train_val_data_1,
                    train_val_labels_1=train_val_labels_1,
                    percentile_1=percentile_feat1,
                    best_features_split_2=best_features_split_2,
                    train_val_data_2=train_val_data_2,
                    train_val_labels_2=train_val_labels_2,
                    percentile_2=percentile_feat2, concatenate=use_feat2)
                print(f'train data: {train_data.shape} \n train lab {train_labels.shape} \n val data: {val_data.shape} \n val lab {val_labels.shape}')

                _, pred_train, pred_val = utils.training_model_CV(model, train_data, train_labels, val_data,
                                                                  val_labels, metrics_fold_dummy, fold=i,
                                                                  metrics_ls=metrics_tup)
                pred_train_dict[f'CV_{i}'].update({name: pred_train})
                pred_val_dict[f'CV_{i}'].update({name: pred_val})
                del pred_train, pred_val

            best_feat_CV_1 = utils.find_best_features(best_features_split_1, data_fs_1['train_data_fs'].shape[1])
            train_val_data_1 = np.concatenate([train_val_data_1[train_index], train_val_data_1[val_index]], axis=0)

            assert train_val_data_1.shape[0] == train_size  # train_size = train+val

            train_val_data_fs_1 = train_val_data_1[:, best_feat_CV_1]
            test_data_1 = test_data_1[:, best_feat_CV_1]

            if not use_feat2:
                labels_split = np.concatenate([train_labels, val_labels], axis=0)
                train_val_data_fs = train_val_data_fs_1
                test_data = test_data_1
                test_labels = test_labels_1

            else:
                best_feat_CV_2 = utils.find_best_features(best_features_split_2, data_fs_2['train_data_fs'].shape[1])
                train_val_data_2 = np.concatenate([train_val_data_2[train_index], train_val_data_2[val_index]], axis=0)

                assert train_val_data_2.shape[0] == train_size  # train_size = train+val

                train_val_data_fs_2 = train_val_data_2[:, best_feat_CV_2]
                test_data_2 = test_data_2[:, best_feat_CV_2]

                labels_split = np.concatenate([train_labels, val_labels], axis=0)
                train_val_data_fs = np.concatenate([train_val_data_fs_1, train_val_data_fs_2], axis=1)
                test_data = np.concatenate([test_data_1, test_data_2], axis=1)
                test_labels = test_labels_1

            _, _, pred_test = utils.train_predict_test(model, train_val_data_fs, labels_split, test_data,
                                                       test_labels, metrics_split_dummy, split=split)

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

    df[f'percentile_position_ls_{p}'] = [np.nanmean(metrics_split['MCC_val']), np.nanstd(metrics_split['MCC_val']),
                                         np.nanmean(metrics_split['acc_val']), np.nanstd(metrics_split['acc_val']),
                                         np.nanmean(metrics_split['recall_val']), np.nanstd(metrics_split['recall_val']),
                                         np.nanmean(metrics_split['precision_val']), np.nanstd(metrics_split['precision_val']),
                                         np.nanmean(metrics_split['F1_val']), np.nanstd(metrics_split['F1_val']),
                                         np.nanmean(metrics_split['balanced_acc_val']), np.nanstd(metrics_split['balanced_acc_val']),
                                         np.nanmean(metrics_split['MCC_test']), np.nanstd(metrics_split['MCC_test']),
                                         np.nanmean(metrics_split['acc_test']), np.nanstd(metrics_split['acc_test']),
                                         np.nanmean(metrics_split['recall_test']), np.nanstd(metrics_split['recall_test']),
                                         np.nanmean(metrics_split['precision_test']), np.nanstd(metrics_split['precision_test']),
                                         np.nanmean(metrics_split['F1_test']), np.nanstd(metrics_split['F1_test']),
                                         np.nanmean(metrics_split['balanced_acc_test']), np.nanstd(metrics_split['balanced_acc_test']),
                                         np.nanmean(metrics_split['agreement_percentage_val']), np.nanstd(metrics_split['agreement_percentage_val']),
                                         np.nanmean(metrics_split['agreement_percentage_test']), np.nanstd(metrics_split['agreement_percentage_test'])]
    
    # To read this file do -> df = pd.read_csv('path/metrics_average_diff_percentile.csv', index_col=0)
    df.to_csv(f'{PATH_SAVE}metrics_average_multicoupling.csv', index=True)

