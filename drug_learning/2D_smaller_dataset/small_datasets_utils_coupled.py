import os
import io
import collections
import yaml
import numpy as np
import numpy.testing as npt
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
try:
    from mordred import Calculator, descriptors # To avoid problems with Power9
    MORDRED_INSTALLED = True
except ImportError:
    MORDRED_INSTALLED = False


plt.style.use("ggplot")
matplotlib.rcParams.update({'font.size': 24})


def write_yaml(data, filename):
    with io.open(f'{filename}.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

def read_yaml(filename):
    """filename must contain name + file extension, ie. .yaml or .yml"""
    with open(f'{filename}', 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


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
    if FINGERPRINT == 'Morgan':
        for mol in structures_shared:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        return np.array(features)
    if FINGERPRINT == 'MACCS':
        for mol in structures_shared:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        return np.array(features)
    print('Please select one of the three aviable fingerprints `Morgan`, `MACCS`, `RDKit`')
    return None

def get_descriptors(smi_arr, activity_labels, clean_dataset=True, save_to_npy=True, filename='shared_set_features_mordred'):
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smi_arr]
    df_descriptors = calc.pandas(mols)
    df_descriptors = df_descriptors.insert(0, "p450-cyp2c9 Activity Outcome", activity_labels, True)
    if clean_dataset:
        df_descriptors = df_descriptors.apply(pd.to_numeric, errors='coerce')
        df_descriptors = df_descriptors.dropna(axis=1)
    if save_to_npy:
        df_descriptors.to_csv(os.path.join("features", filename + ".npy"))
    return df_descriptors

def compute_z_score(df_original):
    df=df_original.copy()
    headers = []
    for col in df.columns:
        headers.append(col)
        if col == "Molecule_index":
            continue
        df[f'{col}_zscore'] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df, headers


def outliers_detection(df, threshold=3):
    df_scored, headers=compute_z_score(df)
    zscore_col = list(set(df_scored.columns) - set(headers)) # to only evaluate zscore columns
    for col in zscore_col:
        df_scored[f'{col}_outlier'] = (abs(df_scored[f'{col}'])> threshold).astype(int)
    return df_scored, zscore_col


def drop_outliers(df, threshold=3):
    df_outlier, zscore_col = outliers_detection(df, threshold=threshold)
    for col in df.columns:
        if col == "Molecule_index":
            continue
        if df_outlier[f'{col}_zscore_outlier'].sum() > 0:
            df_outlier.drop(col , inplace=True,axis = 1)
        df_outlier.drop(f'{col}_zscore_outlier', inplace=True, axis=1)
        df_outlier.drop(f'{col}_zscore', inplace=True, axis = 1)
    return df_outlier

def drop_outliers_samples(df, threshold=3):
    df_outlier, zscore_col = outliers_detection(df, threshold=threshold)
    for col in zscore_col:
        index = df_outlier[df_outlier[f'{col}_outlier'] == 1].index
        df_outlier.drop(index, inplace=True)
        df_outlier.drop(col, inplace=True, axis=1)
        df_outlier.drop(f'{col}_outlier', inplace=True, axis=1)
    return df_outlier

def split_features(features, labels, train_size=450, val_size=50, seed=1, plot_distribution=False, filename='labels_distribution.png', show_plots=False):
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
        fig.savefig(filename)
        if show_plots:
            plt.show()

    return {'train_data': train_data, 'val_data': val_data, 'train_labels': train_labels, 'val_labels': val_labels}


def select_features(X_train, Y_train, X_test, score_func=chi2, k_best=None, percentile=None):
    """score_func=chi2 (default), mutual_info_classif"""
    if k_best is not None:
        fs = SelectKBest(score_func=score_func, k=k_best)
    elif percentile is not None:
        fs = SelectPercentile(score_func=score_func, percentile=percentile)
    else:
        print("Introduce the number of best features to be kept (`k_best`) or the percentile.")
        return
    fs.fit(X_train, Y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs


def plot_score(fs, print_scores=False, filename='score_features.png', show_plots=False):
    """plot the score for all the features"""
    if print_scores:
        for i in range(len(fs.scores_)):
            print('Feature %d: %f' % (i, fs.scores_[i]))
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.savefig(filename)
    if show_plots:
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


def generate_model(model_dict, train_data):
    """layers_dim -- [n_input, n_hid_1, ..., n_output=1]"""
    if model_dict['type'] == "NN":
        dropout = model_dict['dropout']
        lr = model_dict['lr']
        #optimizer = model_dict['optimizer']
        if model_dict['optimizer'].lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif model_dict['optimizer'].lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif model_dict['optimizer'].lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        L2 = model_dict['L2']
        layers_dim = model_dict['layers_dimensions'].copy()
        if not layers_dim[0] == train_data.shape[1]:
            layers_dim.insert(0, train_data.shape[1])
        hidden_layers = []
        for i in range(1, len(layers_dim) - 1):
            hidden_layers.extend([tf.keras.layers.Dropout(dropout)] + [tf.keras.layers.Dense(layers_dim[i], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2))])
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(layers_dim[0], activation='relu', input_shape=(layers_dim[0],))] + hidden_layers +
                                           [tf.keras.layers.Dense(layers_dim[-1], activation="sigmoid")])
        loss_function = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_function,
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    elif model_dict['type'] == 'svm':
        C = model_dict['C']
        kernel_name = model_dict.get('kernel', 'rbf')
        gamma = model_dict.get('gamma', 'scale')
        class_weight = model_dict.get('class_weight')
        model = SVC(C=C, kernel=kernel_name, gamma=gamma, class_weight=class_weight)
    elif model_dict['type'] == 'random_forest':
        n_estimators = model_dict.get('n_estimators', 100)
        seed = model_dict.get('seed', 0)
        class_weight = model_dict.get('class_weight')
        model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=seed)
    elif model_dict['type'] == 'knn':
        num_neigh = model_dict.get('neighbors', 5)
        model = KNeighborsClassifier(n_neighbors=num_neigh)
    elif model_dict['type'] == 'voting':
        estimators = [(id_model, generate_model(submodel, train_data)) for id_model, submodel in model_dict['models'].items()]
        voting_criteria = model_dict.get('criteria', 'hard')
        weights = model_dict.get('weights')
        model = VotingClassifier(estimators, voting=voting_criteria, weights=weights)
    return model


def plot_confusion(predicted_values, target_values, filename='confusion_matrix.png', show_plots=False):
    try:
        cm = confusion_matrix(target_values, predicted_values >= 0.5)
    except TypeError:
        cm = confusion_matrix(target_values, predicted_values)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt="g", cmap="Greens")
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Inactive', 'Active'])
    ax.yaxis.set_ticklabels(['Inactive', 'Active'])
    plt.savefig(filename)
    if show_plots:
        plt.show()


def print_metrics(predicted_values, target_values, verbose=True, agr_percentage=1):
    if agr_percentage == 0:
        print('WARN: Agreement percentage was 0%, 0 were added to the metrics in dict')
        return {'acc': 0, 'precision': 0, 'recall': 0, 'specificity': 0, 'MCC': 0, 'ner': 0,
                'F1': 0, 'balanced_acc': 0}
    try:
        tn, fp, fn, tp = confusion_matrix(target_values, predicted_values >= 0.5, labels=[0,1]).ravel()
        f1 = f1_score(target_values, predicted_values >= 0.5, average='binary')
        balanced_accuracy = balanced_accuracy_score(target_values, predicted_values >= 0.5, sample_weight=None, adjusted=True)
    except TypeError:
        tn, fp, fn, tp = confusion_matrix(target_values, predicted_values, labels=[0,1]).ravel()
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
                    test_recall, test_precision, test_f1, test_balanced_acc,
                    filename='CV_results.png', show_plots=False):
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

    try:
        ax[0, 0].set_ylim(top=y_max, bottom=y_min)
        ax[0, 1].set_ylim(top=y_max, bottom=y_min)
        ax[0, 2].set_ylim(top=y_max, bottom=y_min)
        ax[1, 0].set_ylim(top=y_max, bottom=y_min)
        ax[1, 1].set_ylim(top=y_max, bottom=y_min)
        ax[1, 2].set_ylim(top=y_max, bottom=y_min)
    except ValueError:
        y_max, y_min = 1.1, -0.1
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
    fig.savefig(filename)
    if show_plots:
        plt.show()


def plot_results_split(MCCs_train, MCCs_val, accs_train, accs_val, recall_train, recall_val, precision_train,
                       precision_val, F1_train, F1_val, balanced_acc_train, balanced_acc_val, test_mcc, test_acc,
                       test_recall, test_precision, test_f1, test_balanced_acc,
                       filename='CV_results_1split.png', show_plots=False):
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

    try:
        ax[0, 0].set_ylim(top=y_max, bottom=y_min)
        ax[0, 1].set_ylim(top=y_max, bottom=y_min)
        ax[0, 2].set_ylim(top=y_max, bottom=y_min)
        ax[1, 0].set_ylim(top=y_max, bottom=y_min)
        ax[1, 1].set_ylim(top=y_max, bottom=y_min)
        ax[1, 2].set_ylim(top=y_max, bottom=y_min)
    except ValueError:
        y_max, y_min = 1.1, -0.1
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
    fig.savefig(filename)
    if show_plots:
        plt.show()


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


def append_metrics_to_dict(metrics_dict, name_set_1, dict_1, name_set_2=None, dict_2=None, metrics_ls=None):
    """
    All the metrics in `dict_1` and/or `dict_2` are saved into `metrics_dict` with key = metricName_set1, ie. MCC_train.
    If dict_1=dict_2, it's used another nomenclature to avoid an error when copying the data from fold dict to split dict
    """
    if metrics_ls is None:
        metrics_ls = ['MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc']
    if dict_2 is None:
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

def training_model_CV(model_dict, train_data, train_labels, val_data, val_labels, metrics_dict, metrics_ls=None, fold=0):
    if metrics_ls is None:
        metrics_ls = ['MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc']
    model = generate_model(model_dict, train_data)
    if model_dict['type'] == 'NN':
        model.summary()
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weight = {0: class_weights[0], 1: class_weights[1]}
        model.fit(train_data, train_labels, epochs=10, verbose=0, class_weight=class_weight)
    else:
        model.fit(train_data, train_labels)

    print(f"---> Trainig set fold {fold + 1}")
    pred_train = model.predict(train_data)
    dict_train = print_metrics(pred_train, train_labels)
    print(f"---> Validation set fold {fold + 1}")
    pred_val = model.predict(val_data)
    dict_val = print_metrics(pred_val, val_labels)

    metrics_dict = append_metrics_to_dict(metrics_dict, 'train', dict_train, 'val', dict_val, metrics_ls=metrics_ls)
    return metrics_dict, pred_train, pred_val


def train_predict_test(model_dict, train_data, train_labels, test_data, test_labels, metrics_dict, metrics_ls=None, split=0):
    if metrics_ls is None:
        metrics_ls = ['MCC', 'acc', 'recall', 'precision', 'F1', 'balanced_acc']
    model = generate_model(model_dict, train_data)
    if model_dict['type'] == 'NN':
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weight = {0: class_weights[0], 1: class_weights[1]}
        model.fit(train_data, train_labels, epochs=10, verbose=0, class_weight=class_weight)
    else:
        model.fit(train_data, train_labels)

    print(f"---> Test set split {split + 1}")
    pred_test = model.predict(test_data)
    dict_test = print_metrics(pred_test, test_labels)

    metrics_dict = append_metrics_to_dict(metrics_dict, 'test', dict_test, None, None, metrics_ls=metrics_ls)
    return metrics_dict, dict_test, pred_test

def get_coupled_prediction(pred_fp, pred_des, selection='all',criteria="unanimous", threshold = 0.5):
    assert pred_fp.shape[0]==pred_des.shape[0], "The arrays do not have the same number of predicctions."
    pred_fp_pos = pred_fp >= threshold
    pred_des_pos = pred_des >= threshold
    if selection == 'all':
        coincidences = np.where(pred_fp_pos == pred_des_pos)[0]
        if criteria == "unanimous":
            pred_coupled = pred_fp_pos & pred_des_pos
        elif criteria == "any_positive":
            pred_coupled = pred_fp_pos | pred_des_pos

    elif selection == 'only_coincidences':
        if criteria == "unanimous":  # Only selects True = True and False = False predictions
            coincidences = np.where(pred_fp_pos == pred_des_pos)[0]
            pred_coupled = pred_fp_pos[coincidences]
        elif criteria == "any_positive": # Selects all False = False and all the predictions with at least one pred True
            coincidences = np.where(np.logical_not(np.logical_xor(pred_fp_pos, pred_des_pos)) | np.logical_or(pred_fp_pos, pred_des_pos))[0]
            pred_coupled = pred_fp_pos[coincidences]
            pred_coupled_des = pred_des_pos[coincidences]
            any_pos = np.where(np.logical_xor(pred_coupled, pred_coupled_des))[0]
            pred_coupled[any_pos] = True

        assert coincidences.shape[0] == pred_coupled.shape[0], "Something went wrong. `coincidences` and `pred_coupled` should have the same length."

    return pred_coupled, coincidences

def selection_criteria(addition, num_models, selection='all', criteria='majority'):
    if criteria == 'majority':
        threshold = round(num_models/2)
        coincidences = np.where(addition>=threshold)[0]
        if selection == 'all':
            pred_coupled = np.where(addition>=threshold, True, False) # returns array with True when the majority is True, and false when the majority is False
        elif selection == 'only_coincidences':
            pred_coupled = addition[coincidences]/num_models # returns coupled probabilities normalised to 1.
    elif criteria == 'unanimous':
        coincidences = np.where(addition==num_models)[0]
        if selection == 'all':
            pred_coupled = np.where(addition==num_models, True, False) # returns array with True all the predictions are True, and false when there is at least one False
        elif selection == 'only_coincidences':
            pred_coupled = addition[coincidences]/num_models # returns coupled probabilities normalised to 1.
    else:
        raise Exception(f"Your criteria: {criteria}, has not been implemented. Try with: 'majority', 'unanimous' or 'average'.")
    return pred_coupled, coincidences


def get_multicoupled_prediction(pred_dict, selection='all', criteria='majority', threshold=0.5):
    keys = list(pred_dict.keys())
    if type(pred_dict[keys[0]]) is np.ndarray: # Case for pred_test_dict
        addition = np.zeros(pred_dict[keys[0]].shape[0])
        for name, pred in pred_dict.items():
            addition = np.add(addition, pred>=threshold)
        pred_coupled, coincidences = selection_criteria(addition, len(keys), selection=selection, criteria=criteria)

        return pred_coupled, coincidences

    elif type(pred_dict[keys[0]]) is dict: # Case for pred_train_dict and pred_val_dict
        addition_CV = []
        for cv, item in pred_dict.items():
            name1=list(item.keys())[0]
            addition=np.zeros(item[name1].shape[0])
            for name, pred in item.items():
                addition = np.add(addition, pred>=threshold)
            addition_CV.append(addition)
        pred_coupled_CV, coincidences_CV = [], []
        for add in addition_CV:
            pred_coupled, coincidences = selection_criteria(addition, len(list(pred_dict[keys[0]].keys())), selection=selection, criteria=criteria)
            pred_coupled_CV.append(pred_coupled)
            coincidences_CV.append(coincidences)
 
        assert len(pred_coupled_CV) == len(coincidences_CV)
        assert len(pred_coupled_CV) == len(list(pred_dict.keys()))
        return pred_coupled_CV, coincidences_CV
    else:
        raise Exception('Unknown pred_dict structure')
        
        
def get_SARS_descriptors(PATH_DATA, PATH_FEAT):
    calc = Calculator(descriptors, ignore_3D=True)
    molecules_file = os.path.join(PATH_DATA, "molecules_cleaned_SARS1_SARS2_common.sdf")
    mols = [mol for mol in (Chem.SDMolSupplier(molecules_file))]
    df_descriptors = calc.pandas(mols)
    df_descriptors = df_descriptors.apply(pd.to_numeric, errors='coerce')
    df_descriptors = df_descriptors.dropna(axis=1)
    df_descriptors.to_csv(os.path.join(PATH_FEAT, "SARS1_SARS2_mordred.csv"))
    return df_descriptors


def load_SARS_features(models, active, PATH_DATA, PATH_FEAT, remove_outliers=True):
    feat_used = []
    for name, model in models.items():
        feat_used.append(models[name]['feat1'])
        if model['feat2'] is not None:
            feat_used.append(models[name]['feat2'])
    feat_used = set(feat_used)
    feat_used.discard('Mordred')
    fp_used = list(feat_used)

    features = {}
    for fp in fp_used:
        if not fp.lower() == 'rdkit':  # rdkit.npy does not exist in this folder. Its name is RDKIT.npy
            features[fp] = np.load(os.path.join("..", "2D", "features", "SARS1_SARS2", f"{fp.lower()}.npy"))
        else:  # TODO Change RDKIT.npy filename to rdkit.npy to avoid if statement. (WARN, it may affect to Joan's code...)
            features[fp] = np.load(os.path.join("..", "2D", "features", "SARS1_SARS2", f"{fp.upper()}.npy"))

    if os.path.exists(os.path.join(PATH_FEAT, "SARS1_SARS2_mordred.csv")):
        df_descriptors = pd.read_csv(os.path.join(PATH_FEAT, "SARS1_SARS2_mordred.csv"))
    else:
        df_descriptors = get_SARS_descriptors(PATH_DATA, PATH_FEAT)

    if os.path.exists(os.path.join(PATH_FEAT, 'SARS1_SARS2_mordred_clean_no_outliers.csv')):
        descriptors_shared = pd.read_csv(os.path.join(PATH_FEAT, 'SARS1_SARS2_mordred_clean_no_outliers.csv'))
        if 'Unnamed: 0' in descriptors_shared.columns:
            descriptors_shared = descriptors_shared.drop(['Unnamed: 0'], axis=1)

    else:
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

    # if fp_used: # We always need Mordred, so it has no sense to have a if statement
    norm_descriptors_shared = pd.DataFrame(normalize(descriptors_shared, norm='max', axis=0))
    norm_descriptors_shared = np.asarray(norm_descriptors_shared).astype(
        np.float32)  # to avoid problems with the KFoldCrossValidation
    features.update({'Mordred': norm_descriptors_shared})

    return features


def get_train_val_split(train_index, val_index, best_features_split_1, train_val_data_1, train_val_labels_1,
                        percentile_1, best_features_split_2=None, train_val_data_2=None, train_val_labels_2=None,
                        percentile_2=None, concatenate=False):

    train_data_1, val_data_fp2 = train_val_data_1[train_index], train_val_data_1[val_index]
    train_labels_1, val_labels_1 = train_val_labels_1[train_index], train_val_labels_1[val_index]
    data_fs_1, best_fold_1 = get_best_features(train_data_1, train_labels_1, val_data_fp2, percentile_1)
    best_features_split_1.extend(list(best_fold_1))

    assert train_data_1.shape[1] == val_data_fp2.shape[1]
    assert train_data_1.shape[0] == train_labels_1.shape[0]
    assert val_data_fp2.shape[0] == val_labels_1.shape[0]
    assert best_fold_1.shape[0] == data_fs_1['train_data_fs'].shape[1]

    if concatenate is False:
        train_data, val_data = data_fs_1['train_data_fs'], data_fs_1['val_data_fs']
        train_labels, val_labels = train_labels_1, val_labels_1

        return train_data, train_labels, val_data, val_labels, data_fs_1, None

    elif concatenate is True:
        train_data_2, val_data_2 = train_val_data_2[train_index], train_val_data_2[val_index]
        train_labels_2, val_labels_2 = train_val_labels_2[train_index], train_val_labels_2[val_index]
        data_fs_2, best_fold_2 = get_best_features(train_data_2, train_labels_2, val_data_2, percentile_2)
        best_features_split_2.extend(list(best_fold_2))

        assert train_data_2.shape[1] == val_data_2.shape[1]
        assert train_data_2.shape[0] == train_labels_2.shape[0]
        assert val_data_2.shape[0] == val_labels_2.shape[0]
        assert best_fold_2.shape[0] == data_fs_2['train_data_fs'].shape[1]

        npt.assert_array_equal(train_labels_1, train_labels_2,
                               err_msg='Train labels do not coincide between feat1 and feat2.')
        npt.assert_array_equal(val_labels_1, val_labels_2,
                               err_msg='Validation labels do not coincide between feat1 and feat2.')
        assert data_fs_1['train_data_fs'].shape[0] == data_fs_2['train_data_fs'].shape[0]
        assert data_fs_1['val_data_fs'].shape[0] == data_fs_2['val_data_fs'].shape[0]

        train_labels, val_labels = train_labels_2, val_labels_2

        train_data = np.concatenate([data_fs_1['train_data_fs'], data_fs_2['train_data_fs']], axis=1)
        val_data = np.concatenate([data_fs_1['val_data_fs'], data_fs_2['val_data_fs']], axis=1)

        return train_data, train_labels, val_data, val_labels, data_fs_1, data_fs_2
