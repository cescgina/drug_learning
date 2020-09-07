import os
import random
import numpy as np
import numpy.testing as npt
import pandas as pd
import collections
import tensorflow as tf
from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize #, minmax_scale it could also be tried.
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem, DataStructs
#from mordred import Calculator, descriptors # To avoid problems with Power9


plt.style.use("ggplot")
matplotlib.rcParams.update({'font.size': 24})


def col_to_array(df, col_name='p450-cyp2c9 Activity Outcome'):
    col = df[col_name]
    arr = col.to_numpy() # class wants to make reference to active or inactive
    arr = np.reshape(arr, [arr.shape[0],1])
    return np.squeeze(arr)


def get_features(input_sdf, FINGERPRINT):
    structures_shared = Chem.SDMolSupplier(input_sdf)
    features = []
    if FINGERPRINT == 'RDKit':
        for mol in structures_shared:
            fp = Chem.RDKFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp,arr)
            features.append(arr)
        return np.array(features)
    elif FINGERPRINT == 'Morgan':
        for mol in structures_shared:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        return np.array(features)
    elif FINGERPRINT == 'MACCS':
        for mol in structures_shared:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        return np.array(features)

    else:
        print('Please select one of the three aviable fingerprints `Morgan`, `MACCS`, `RDKit`')

def get_descriptors(smi_arr, activity_labels, clean_dataset=True, save_to_npy=True, filename='shared_set_features_mordred'):
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smi_arr]
    df_descriptors = calc.pandas(mols)
    df_descriptors = df_descriptors.insert(0, "p450-cyp2c9 Activity Outcome", activity_labels, True)
    if clean_dataset:
        df_descriptors = descriptors_shared.apply(pd.to_numeric, errors='coerce')
        df_descriptors = df_descriptors.dropna(axis=1)
    if save_to_npy:
        df_descriptors.to_csv(os.path.join("features", filename + ".npy"))
    return df_descriptors

def compute_z_score(df_original):
    df=df_original.copy()
    headers = []
    for col in df.columns:
        df[f'{col}_zscore'] = (df[col] - df[col].mean())/df[col].std(ddof=0)
        headers.append(col)
    return df, headers


def outliers_detection(df, threshold=3):
    df_scored, headers=compute_z_score(df)
    zscore_col = list(set(df_scored.columns) - set(headers)) # to only evaluate zscore columns
    for col in zscore_col:
        df_scored[f'{col}_outlier'] = (abs(df_scored[f'{col}'])> threshold).astype(int)
    return df_scored, zscore_col


def drop_outliers(df, threshold=3):
    df_outlier, zscore_col = outliers_detection(df, threshold=threshold)
    for col in zscore_col:
        index = df_outlier[ df_outlier[f'{col}_outlier'] == 1 ].index
        df_outlier.drop(index , inplace=True)
        df_outlier.drop(col , inplace=True,axis = 1)
        df_outlier.drop(f'{col}_outlier' , inplace=True, axis = 1)
    return df_outlier


def split_features(features, labels, train_size=450, val_size=50, seed=1, plot_distribution=False, filename='labels_distribution'):
    train_data, val_data, train_labels, val_labels = train_test_split(features, labels, train_size=train_size,
                                                                      test_size=val_size, stratify=labels,
                                                                      random_state=seed)

    if plot_distribution:
        fig, ax = plt.subplots(1, 2, figsize=(14, 10))
        ax[0].hist(train_labels)
        ax[0].set_xlabel("Training set 2c9")
        ax[1].hist(val_labels)
        ax[1].set_xlabel("Validation set 2c9")
        plt.subplots_adjust(wspace=0.5)
        plt.savefig(filename)
        plt.show()

    return {'train_data': train_data, 'val_data': val_data, 'train_labels': train_labels, 'val_labels': val_labels}


def select_features(X_train, Y_train, X_test, score_func=chi2, k_best=None, percentile=None):
    """score_func=chi2 (default), mutual_info_classif"""
    if not k_best == None:
        fs = SelectKBest(score_func=score_func, k=k_best)
    elif not percentile == None:
        fs = SelectPercentile(score_func=score_func, percentile=percentile)
    else:
        print("Introduce the number of best features to be kept (`k_best`) or the percentile.")
        return
    fs.fit(X_train, Y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs


def plot_score(fs, print_scores=False, filename='score_features'):
    """plot the score for all the features"""
    if print_scores:
        for i in range(len(fs.scores_)):
            print('Feature %d: %f' % (i, fs.scores_[i]))
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.savefig(filename)
    plt.show()


def get_best_features_index(fs):
    """Returns a numpy array with the indexs of the best features."""
    mask = fs.get_support()
    best_features_tup = np.where(mask == True)

    return best_features_tup[0]


def find_best_features(best_feat_split, num_feat):
    """Given a list containing the index of `best_feat_split`, we will count the most repeated `num_feat`"""
    x = collections.Counter(best_feat_split)
    return [feat for feat, count in x.most_common(num_feat)]


def generate_model(layers_dim, lr, dropout, optimizer, L2):
    """layers_dim -- [n_input, n_hid_1, ..., n_output=1]"""
    hidden_layers = []
    for i in range(1, len(layers_dim) - 1): hidden_layers.extend([tf.keras.layers.Dropout(dropout)] + [
        tf.keras.layers.Dense(layers_dim[i], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2))])
    model = tf.keras.models.Sequential([
                                           tf.keras.layers.Dense(layers_dim[0], activation='relu',
                                                                 input_shape=(layers_dim[0],))] +
                                       hidden_layers +
                                       [tf.keras.layers.Dense(layers_dim[-1], activation="sigmoid")])
    loss_function = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


def plot_confusion(predicted_values, target_values, filename='confusion_matrix'):
    try:
        cm = confusion_matrix(target_values, predicted_values >= 0.5)
    except TypeError:
        cm = confusion_matrix(target_values, predicted_values)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt="g", cmap="Greens")
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['Inactive', 'Active']);
    ax.yaxis.set_ticklabels(['Inactive', 'Active'])
    plt.savefig(filename)
    plt.show()


def print_metrics(predicted_values, target_values, verbose=True, agr_percentage=1):
    if agr_percentage == 0:
        print('WARN: Agreement percentage was 0%, 0 were added to the metrics in dict')
        return {'acc': 0, 'precision': 0, 'recall': 0, 'specificity': 0, 'MCC': 0, 'ner': 0,
                'F1': 0, 'balanced_acc': 0}
    try:
        tn, fp, fn, tp = confusion_matrix(target_values, predicted_values >= 0.5).ravel()
        f1 = f1_score(target_values, predicted_values >= 0.5, average='binary')
        balanced_accuracy = balanced_accuracy_score(target_values, predicted_values >= 0.5, sample_weight=None, adjusted=True)
    except TypeError:
        tn, fp, fn, tp = confusion_matrix(target_values, predicted_values).ravel()
        f1 = f1_score(target_values, predicted_values, average='binary')
        balanced_accuracy = balanced_accuracy_score(target_values, predicted_values, sample_weight=None, adjusted=True)
    Sn = tp / (tp + fn)
    Sp = tn / (tn + fp)
    precision = tp / (tp + fp)
    ner = (Sn + Sp) / 2
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    #f1 = 2 * precision * recall / (precision + recall)
    #balanced_accuracy = Sp * recall / 2
    
    if verbose:
        print(
            f"NER: {ner:.3f}, Sensitivity or recall: {Sn:.3f}, Specificity: {Sp:.3f}, Precision: {precision:.3f}, Correctly classified: {accuracy:.3f}, MCC: {mcc:.3f}, F1 score: {f1:.3f}, Balanced accuracy: {balanced_accuracy:.3f}")
    return {'acc': accuracy, 'precision': precision, 'recall': Sn, 'specificity': Sp, 'MCC': mcc, 'ner': ner,
            'F1': f1, 'balanced_acc': balanced_accuracy}


def plot_results_CV(MCCs_train, MCCs_val, accs_train, accs_val, recall_train, recall_val, precision_train,
                    precision_val, F1_train, F1_val, balanced_acc_train, balanced_acc_val, test_acc, test_mcc,
                    test_recall, test_precision, test_f1, test_balanced_acc, filename='CV_results'):
    fig, ax = plt.subplots(2, 3, figsize=(16, 16))
    y_min = -0.1 + np.nanmin(
        [np.nanmin(balanced_acc_train), np.nanmin(balanced_acc_val), np.nanmin(test_balanced_acc), np.nanmin(F1_train),
         np.nanmin(F1_val), np.nanmin(test_f1), np.nanmin(accs_train), np.nanmin(accs_val), np.nanmin(test_acc),
         np.nanmin(MCCs_train), np.nanmin(MCCs_val), np.nanmin(test_mcc), np.nanmin(recall_train),
         np.nanmin(recall_val), np.nanmin(test_recall), np.nanmin(precision_train), np.nanmin(precision_val),
         np.nanmin(test_precision)])
    y_max = 0.1 + np.nanmax(
        [np.nanmax(balanced_acc_train), np.nanmax(balanced_acc_val), np.nanmax(test_balanced_acc), np.nanmax(F1_train),
         np.nanmax(F1_val), np.nanmax(test_f1), np.nanmax(accs_train), np.nanmax(accs_val), np.nanmax(test_acc),
         np.nanmax(MCCs_train), np.nanmax(MCCs_val), np.nanmax(test_mcc), np.nanmax(recall_train),
         np.nanmax(recall_val), np.nanmax(test_recall), np.nanmax(precision_train), np.nanmax(precision_val),
         np.nanmax(test_precision)])
    ax[0, 0].boxplot([np.array(accs_train)[~np.isnan(accs_train)], np.array(accs_val)[~np.isnan(accs_val)]],
                     labels=["Train", "Val"])
    ax[0, 1].boxplot([np.array(recall_train)[~np.isnan(recall_train)], np.array(recall_val)[~np.isnan(recall_val)]],
                     labels=["Train", "Val"])
    ax[0, 2].boxplot(
        [np.array(precision_train)[~np.isnan(precision_train)], np.array(precision_val)[~np.isnan(precision_val)]],
        labels=["Train", "Val"])
    ax[1, 0].boxplot([np.array(balanced_acc_train)[~np.isnan(balanced_acc_train)],
                      np.array(balanced_acc_val)[~np.isnan(balanced_acc_val)]], labels=["Train", "Val"])
    ax[1, 1].boxplot([np.array(F1_train)[~np.isnan(F1_train)], np.array(F1_val)[~np.isnan(F1_val)]],
                     labels=["Train", "Val"])
    ax[1, 2].boxplot([np.array(MCCs_train)[~np.isnan(MCCs_train)], np.array(MCCs_val)[~np.isnan(MCCs_val)]],
                     labels=["Train", "Val"])

    ax[0, 0].hlines(y=test_acc, linewidth=2, xmin=0, xmax=6, color='r')
    ax[0, 1].hlines(y=test_recall, linewidth=2, xmin=0, xmax=6, color='r')
    ax[0, 2].hlines(y=test_precision, linewidth=2, xmin=0, xmax=6, color='r')
    ax[1, 0].hlines(y=test_balanced_acc, linewidth=2, xmin=0, xmax=6, color='r')
    ax[1, 1].hlines(y=test_f1, linewidth=2, xmin=0, xmax=6, color='r')
    ax[1, 2].hlines(y=test_mcc, linewidth=2, xmin=0, xmax=6, color='r')

    ax[0, 0].set_ylim(top=y_max, bottom=y_min)
    ax[0, 1].set_ylim(top=y_max, bottom=y_min)
    ax[0, 2].set_ylim(top=y_max, bottom=y_min)
    ax[1, 0].set_ylim(top=y_max, bottom=y_min)
    ax[1, 1].set_ylim(top=y_max, bottom=y_min)
    ax[1, 2].set_ylim(top=y_max, bottom=y_min)
    ax[0, 0].set_xlim(left=0.5, right=2.5)
    ax[0, 1].set_xlim(left=0.5, right=2.5)
    ax[0, 2].set_xlim(left=0.5, right=2.5)
    ax[1, 0].set_xlim(left=0.5, right=2.5)
    ax[1, 1].set_xlim(left=0.5, right=2.5)
    ax[1, 2].set_xlim(left=0.5, right=2.5)
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 1].set_ylabel("Recall")
    ax[0, 2].set_ylabel("Precision")
    ax[1, 0].set_ylabel("Balanced Accuracy")
    ax[1, 1].set_ylabel("F1")
    ax[1, 2].set_ylabel("MCC")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_results_split(MCCs_train, MCCs_val, accs_train, accs_val, recall_train, recall_val, precision_train,
                       precision_val, F1_train, F1_val, balanced_acc_train, balanced_acc_val, test_mcc, test_acc,
                       test_recall, test_precision, test_f1, test_balanced_acc, filename='CV_results_1split'):
    fig, ax = plt.subplots(2, 3, figsize=(16, 16))
    y_min = -0.1 + np.nanmin(
        [np.nanmin(balanced_acc_train), np.nanmin(balanced_acc_val), np.nanmin(test_balanced_acc), np.nanmin(F1_train),
         np.nanmin(F1_val), np.nanmin(test_f1), np.nanmin(accs_train), np.nanmin(accs_val), np.nanmin(test_acc),
         np.nanmin(MCCs_train), np.nanmin(MCCs_val), np.nanmin(test_mcc), np.nanmin(recall_train),
         np.nanmin(recall_val), np.nanmin(test_recall), np.nanmin(precision_train), np.nanmin(precision_val),
         np.nanmin(test_precision)])
    y_max = 0.1 + np.nanmax(
        [np.nanmax(balanced_acc_train), np.nanmax(balanced_acc_val), np.nanmax(test_balanced_acc), np.nanmax(F1_train),
         np.nanmax(F1_val), np.nanmax(test_f1), np.nanmax(accs_train), np.nanmax(accs_val), np.nanmax(test_acc),
         np.nanmax(MCCs_train), np.nanmax(MCCs_val), np.nanmax(test_mcc), np.nanmax(recall_train),
         np.nanmax(recall_val), np.nanmax(test_recall), np.nanmax(precision_train), np.nanmax(precision_val),
         np.nanmax(test_precision)])
    ax[0, 0].boxplot([np.array(accs_train)[~np.isnan(accs_train)], np.array(accs_val)[~np.isnan(accs_val)],
                      np.array(test_acc)[~np.isnan(test_acc)]], labels=["Train", "Val", "Test"])
    ax[0, 1].boxplot([np.array(recall_train)[~np.isnan(recall_train)], np.array(recall_val)[~np.isnan(recall_val)],
                      np.array(test_recall)[~np.isnan(test_recall)]], labels=["Train", "Val", "Test"])
    ax[0, 2].boxplot(
        [np.array(precision_train)[~np.isnan(precision_train)], np.array(precision_val)[~np.isnan(precision_val)],
         np.array(test_precision)[~np.isnan(test_precision)]], labels=["Train", "Val", "Test"])
    ax[1, 0].boxplot([np.array(balanced_acc_train)[~np.isnan(balanced_acc_train)],
                      np.array(balanced_acc_val)[~np.isnan(balanced_acc_val)],
                      np.array(test_balanced_acc)[~np.isnan(test_balanced_acc)]], labels=["Train", "Val", "Test"])
    ax[1, 1].boxplot([np.array(F1_train)[~np.isnan(F1_train)], np.array(F1_val)[~np.isnan(F1_val)],
                      np.array(test_f1)[~np.isnan(test_f1)]], labels=["Train", "Val", "Test"])
    ax[1, 2].boxplot([np.array(MCCs_train)[~np.isnan(MCCs_train)], np.array(MCCs_val)[~np.isnan(MCCs_val)],
                      np.array(test_mcc)[~np.isnan(test_mcc)]], labels=["Train", "Val", "Test"])

    ax[0, 0].set_ylim(top=y_max, bottom=y_min)
    ax[0, 1].set_ylim(top=y_max, bottom=y_min)
    ax[0, 2].set_ylim(top=y_max, bottom=y_min)
    ax[1, 0].set_ylim(top=y_max, bottom=y_min)
    ax[1, 1].set_ylim(top=y_max, bottom=y_min)
    ax[1, 2].set_ylim(top=y_max, bottom=y_min)
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 1].set_ylabel("Recall")
    ax[0, 2].set_ylabel("Precision")
    ax[1, 0].set_ylabel("Balanced Accuracy")
    ax[1, 1].set_ylabel("F1")
    ax[1, 2].set_ylabel("MCC")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def select_features(X_train, Y_train, X_test, score_func=chi2, k_best=None, percentile=None):
    """score_func=chi2 (default), mutual_info_classif"""
    if not k_best == None:
        fs = SelectKBest(score_func=score_func, k=k_best)
    elif not percentile == None:
        fs = SelectPercentile(score_func=score_func, percentile=percentile)
    else:
        print("Introduce the number of best features to be kept (`k_best`) or the percentile.")
        return
    fs.fit(X_train, Y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs


def plot_score(fs, print_scores=False, filename='plt_score'):
    """plot the score for all the features"""
    if print_scores:
        for i in range(len(fs.scores_)):
            print('Feature %d: %f' % (i, fs.scores_[i]))
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.figsave(filename)
    plt.show()


def get_best_features_index(fs):
    """Returns a numpy array with the indexs of the best features."""
    mask = fs.get_support()
    best_features_tup = np.where(mask == True)

    return best_features_tup[0]


def get_best_features(train_data, train_labels, val_data, percentile):
    """calculates score fuction of the features and returns the best features"""
    train_data, val_data, fs = select_features(train_data, train_labels, val_data, score_func=mutual_info_classif,
                                               percentile=percentile)
    best_features = get_best_features_index(fs)

    assert train_data.shape[1] == val_data.shape[1]
    assert train_data.shape[1] == best_features.shape[0]

    return {'train_data_fs': train_data, 'val_data_fs': val_data}, best_features


def get_best_features_per_split(features, labels, percetile_fingerprint=60, percetile_descriptors=60, train_size=450,
                                val_size=50, seed=1, plot_distribution=False):
    """Splits the data and find the best features"""
    data = split_features(features, labels, train_size=train_size, val_size=val_size, seed=seed,
                          plot_distribution=plot_distribution)
    train_data, val_data, fs = select_features(data['train_data'], data['train_labels'], data['val_data'],
                                               score_func=mutual_info_classif, percentile=percetile_fingerprint)
    best_features = get_best_features_index(fs)

    assert train_data.shape[1] == val_data.shape[1]
    assert train_data.shape[1] == best_features.shape[0]

    return {'train_data_fs': train_data, 'val_data_fs': val_data, 'train_labels': data['train_labels'],
            'val_labels': data['val_labels']}, best_features


def select_feat_and_get_best_feat_per_split(use_fingerprints, use_descriptors, fp_data=None,
                                            descriptors_data=None, labels=None, percetile_fingerprint=60,
                                            percetile_descriptors=60, train_size=450, val_size=50, seed=1,
                                            plot_distribution=False):
    if use_fingerprints:
        data_fs_fp, best_feat_index_fp = get_best_features_per_split(fp_data, labels,
                                                                     percetile_fingerprint=percetile_fingerprint,
                                                                     percetile_descriptors=percetile_descriptors,
                                                                     train_size=train_size, val_size=val_size,
                                                                     seed=seed, plot_distribution=plot_distribution)
        if not use_descriptors:
            return data_fs_fp, None, best_feat_index_fp, None
    if use_descriptors:
        data_fs_des, best_feat_index_des = get_best_features_per_split(descriptors_data, labels,
                                                                       percetile_fingerprint=percetile_fingerprint,
                                                                       percetile_descriptors=percetile_descriptors,
                                                                       train_size=train_size, val_size=val_size,
                                                                       seed=seed, plot_distribution=plot_distribution)
        if not use_fingerprints:
            return None, data_fs_des, None, best_feat_index_des
    if use_descriptors and use_fingerprints:
        npt.assert_array_equal(data_fs_fp['train_labels'], data_fs_des['train_labels'],
                               err_msg='Train labels do not coincide between descriptors and fingerprints.')
        npt.assert_array_equal(data_fs_fp['val_labels'], data_fs_des['val_labels'],
                               err_msg='Validation labels do not coincide between descriptors and fingerprints.')
        assert data_fs_fp['train_data_fs'].shape[0] == data_fs_des['train_data_fs'].shape[0]
        assert data_fs_fp['val_data_fs'].shape[0] == data_fs_des['val_data_fs'].shape[0]

        return data_fs_fp, data_fs_des, best_feat_index_fp, best_feat_index_des

    
def append_metrics_to_dict(metrics_dict, name_set_1, dict_1, name_set_2=None, dict_2=None,  metrics_ls = ['MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc']):
    """
    All the metrics in `dict_1` and/or `dict_2` are saved into `metrics_dict` with key = metric_name_set1, ie. MCC_train.
    If dict_1=dict_2, it's used another nomenclature to avoid an error when copying the data from fold dict to split dict
    """
    if dict_2 == None:
        for metric in metrics_ls:
            metrics_dict[f'{metric}_{name_set_1}'].append(dict_1[f'{metric}'])
    elif dict_1 == dict_2:
        for metric in metrics_ls:
            metrics_dict[f'{metric}_{name_set_1}'].append(dict_1[f'{metric}_{name_set_1}'])
            metrics_dict[f'{metric}_{name_set_2}'].append(dict_2[f'{metric}_{name_set_2}'])
    else:
        for metric in metrics_ls:
            metrics_dict[f'{metric}_{name_set_1}'].append(dict_1[f'{metric}'])
            metrics_dict[f'{metric}_{name_set_2}'].append(dict_2[f'{metric}'])
            
    return metrics_dict

def training_model_CV(model_dict, train_data, train_labels, val_data, val_labels, metrics_dict, metrics_ls = ['MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc'], fold=0):
    layers_dim = model_dict['layers_dimensions'].copy()
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight = {0: class_weights[0], 1: class_weights[1]}
    
    if not layers_dim[0] == train_data.shape[1]:
        layers_dim.insert(0, train_data.shape[1])
    model = generate_model(layers_dim, model_dict['lr'], model_dict['dropout'], model_dict['optimizer'], model_dict['L2'])
    history = model.fit(train_data, train_labels, epochs=10, verbose=0, class_weight=class_weight)  # , validation_data = (val_data, val_labels))

    print(f"---> Trainig set fold {fold + 1}")
    pred_train = model.predict(train_data)
    dict_train = print_metrics(pred_train, train_labels)
    print(f"---> Validation set fold {fold + 1}")
    pred_val = model.predict(val_data)
    dict_val = print_metrics(pred_val, val_labels)

    metrics_dict = append_metrics_to_dict(metrics_dict, 'train', dict_train, 'val', dict_val, metrics_ls=metrics_ls)
        
    return metrics_dict, pred_train, pred_val

def train_predict_test(model_dict, train_data, train_labels, test_data, test_labels, metrics_dict, metrics_ls = ['MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc'], split=0):
    layers_dim = model_dict['layers_dimensions'].copy()
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight = {0: class_weights[0], 1: class_weights[1]}
    
    if not layers_dim[0] == train_data.shape[1]:
        layers_dim.insert(0, train_data.shape[1])
    model = generate_model(layers_dim, model_dict['lr'], model_dict['dropout'], model_dict['optimizer'], model_dict['L2'])
    history = model.fit(train_data, train_labels, epochs=10, verbose=0, class_weight=class_weight)  # , validation_data = (val_data, val_labels))

    print(f"---> Test set split {split + 1}")
    pred_test = model.predict(test_data)
    dict_test = print_metrics(pred_test, test_labels)

    metrics_dict = append_metrics_to_dict(metrics_dict, 'test', dict_test, None, None, metrics_ls=metrics_ls)
        
    return metrics_dict, dict_test, pred_test

def get_coupled_prediction(pred_fp, pred_des, labels=None):
    assert pred_fp.shape[0]==pred_des.shape[0], "The arrays do not have the same number of predicctions."
    i=0
    coincidences = []
    pred_coupled = []
    for fp, des in zip(pred_fp>=0.5, pred_des>=0.5):
        if fp == des:
            pred_coupled.append(fp[0])
            coincidences.append(i)
        i+=1
    
    assert len(coincidences) == len(pred_coupled), "Something went wrong. `coincidences` and `pred_coupled` should have the same length. "
    
    if not labels == None:
        coincident_labels = labels[coincidences]
        return pred_coupled, coincident_labels
    
    return pred_coupled, coincidences

