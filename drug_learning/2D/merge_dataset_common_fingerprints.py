import pandas as pd
from rdkit import Chem

data_morgan = pd.read_csv("../datasets/SARS2/dataset_cleaned_SARS1_SARS2_morgan.csv")
data_maccs = pd.read_csv("../datasets/SARS2/dataset_cleaned_SARS1_SARS2_maccs.csv")
data_rdkit = pd.read_csv("../datasets/SARS2/dataset_cleaned_SARS1_SARS2_rdkit.csv")

common = pd.merge(data_morgan, data_rdkit, how='inner')
common = pd.merge(common, data_maccs, how='inner')
common.to_csv("../datasets/SARS2/dataset_cleaned_SARS1_SARS2_common.csv", index=False)

mols_morgan = Chem.SDMolSupplier("../datasets/SARS2/molecules_cleaned_SARS1_SARS2_morgan.sdf")
mols_maccs = Chem.SDMolSupplier("../datasets/SARS2/molecules_cleaned_SARS1_SARS2_maccs.sdf")
mols_rdkit = Chem.SDMolSupplier("../datasets/SARS2/molecules_cleaned_SARS1_SARS2_rdkit.sdf")

mols = {}
for mol in mols_morgan:
    name = mol.GetProp("_Name")
    if not name in mols:
        mols[name] = mol

for mol in mols_rdkit:
    name = mol.GetProp("_Name")
    if not name in mols:
        mols[name] = mol

for mol in mols_maccs:
    name = mol.GetProp("_Name")
    if not name in mols:
        mols[name] = mol

# common molecules from data frames
fw = Chem.SDWriter("../datasets/SARS2/molecules_cleaned_SARS1_SARS2_common.sdf")
for mol_name in common["Molecule_ID"]:
    fw.write(mols[str(mol_name)])
