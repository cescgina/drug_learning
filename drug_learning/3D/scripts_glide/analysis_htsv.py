import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
plt.switch_backend("pdf")


def parse_args():
    """
        Parse command line arguments
        :returns: object -- Object containing command line options
    """
    parser = argparse.ArgumentParser(description="Select common structures from a series of docking jobs ")
    parser.add_argument("input_files", nargs='*', type=str, help="Files to process")
    parser.add_argument("-o", "--output_path", type=str, default="", help="Path where to write the structures")
    args = parser.parse_args()
    return args.input_files, args.output_path


def main(reports, output):
    id_field = "Title"
    data_frames = []
    ligands = []
    all_common = None
    if output and not os.path.exists(output):
        os.makedirs(output)
    csv_folder = os.path.join(output, "processed_csv")
    plots_folder = os.path.join(output, "plots")
    os.makedirs(csv_folder)
    os.makedirs(plots_folder)
    for report in reports:
        # avoid first 11 lines of useless header
        data = pd.read_csv(report, sep=" ", skiprows=11, usecols=["Rank", "Title", "Lig#", "Score", "Intern"], skipinitialspace=True, skipfooter=34, comment="=", engine="python")
        # data["From file"] = report
        # keep only the first appearance of any ligands (best ranked)
        data = data.drop_duplicates("Title", keep="first")
        data_frames.append(data)
        lig = set(data[id_field])
        ligands.append(lig)
        if all_common is None:
            all_common = lig
        else:
            all_common &= lig

    names = ["" for _ in reports]
    for i, j in itertools.combinations(range(len(reports)), 2):
        if i == j:
            continue
        common_ligands = ligands[i] & ligands[j]
        name_i, _ = os.path.splitext(reports[i])
        if not names[i]:
            names[i] = name_i
        name_j, _ = os.path.splitext(reports[j])
        if not names[j]:
            names[j] = name_j
        mask_i = data_frames[i][id_field].map(lambda x: x in common_ligands)
        mask_j = data_frames[j][id_field].map(lambda x: x in common_ligands)
        data = pd.merge(data_frames[i][mask_i], data_frames[j][mask_j], on=id_field, suffixes=("_%s" % name_i, "_%s" % name_j))
        score_cols = [col for col in data.columns if col.startswith("Score")]
        data["Total scores"] = data[score_cols].sum(axis=1)
        data = data.sort_values(by="Total scores")
        min_scores = data[score_cols].min(axis="columns")
        min_receptor = data[score_cols].idxmin(axis="columns")
        data["Best score"] = min_scores
        data["Best receptor"] = ["_".join(l.split("_")[1:-1]) for l in min_receptor]
        data.to_csv(os.path.join(csv_folder, "merged_%s_%s.csv" % (name_i, name_j)), index=False)
        score_i = "Score_%s" % name_i
        score_j = "Score_%s" % name_j
        trend = np.polyfit(data[score_i], data[score_j], 1)
        y = np.poly1d(trend)(data[score_i])
        ax = data.plot.scatter(x=score_i, y=score_j)
        ax.plot(data[score_i], y)
        ax.text(-9.5, 4, "$R^2 = %.3f$" % r2_score(y, data[score_i]))
        ax.get_figure().savefig(os.path.join(plots_folder, "scatter_%s_%s.png" % (name_i, name_j)), bbox_inches="tight", dpi=300)
        # data = pd.concat([data_frames[i][mask_i]], data_frames[j][mask_j]], ignore_index=True)

    common_data = None
    for i, df in enumerate(data_frames):
        mask = df[id_field].map(lambda x: x in common_ligands)
        rename_columns = {"Rank": "Rank_%s" % names[i], "Score": "Score_%s" % names[i], "Lig#": "Lig#_%s" % names[i], "Intern": "Intern_%s" % names[i]}
        if common_data is None:
            common_data = df[mask]
            common_data = common_data.rename(columns=rename_columns)
        else:
            df_new = df.rename(columns=rename_columns)
            common_data = pd.merge(common_data, df_new[mask], on=id_field, suffixes=("", ""))
    score_cols = [col for col in common_data.columns if col.startswith("Score")]
    common_data["Total scores"] = common_data[score_cols].sum(axis=1)
    common_data = common_data.sort_values(by="Total scores")
    min_scores = common_data[score_cols].min(axis="columns")
    min_receptor = common_data[score_cols].idxmin(axis="columns")
    common_data["Best score"] = min_scores
    common_data["Best receptor"] = ["_".join(l.split("_")[1:-1]) for l in min_receptor]
    common_data.to_csv(os.path.join(csv_folder, "merged_all.csv"), index=False)

if __name__ == "__main__":
    in_files, out_path = parse_args()
    main(in_files, out_path)
