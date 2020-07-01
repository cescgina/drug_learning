import os
import argparse
import pandas as pd
from schrodinger import structure


def parse_args():
    """
        Parse command line arguments
        :returns: object -- Object containing command line options
    """
    parser = argparse.ArgumentParser(description="Extract selected structures from a docking maegz file")
    parser.add_argument("csv_common_molecules", type=str, help="Csv file with the molecules to select")
    parser.add_argument("structures_file", nargs='*', type=str, help="File with the strcutures to select")
    parser.add_argument("-o", "--output_path", type=str, default="", help="Path where to write the structures")
    parser.add_argument("-of", "--output_file", type=str, default="extracted_structures.sdf", help="File to write the structures, the extension will determine the format")
    parser.add_argument("-n", type=int, default=-1, help="Number of molecules to extract")
    args = parser.parse_args()
    return args.csv_common_molecules, args.structures_file, args.output_path, args.output_file, args.n


def main(csv_input, structures_input, output_path, output_file, num_mols):
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = os.path.join(output_path, output_file)
    out_file_root, out_file_ext = os.path.splitext(output_file)
    data = pd.read_csv(csv_input)
    common_molecules = set(data["Title"])
    data = data.set_index(keys="Title")
    if num_mols == -1:
        num_mols = data.shape[0]
    for struct_file in structures_input:
        name, _ = os.path.splitext(os.path.split(struct_file)[1])
        output_file_mod = "".join([out_file_root, "_", name, out_file_ext])
        with structure.StructureWriter(output_file_mod) as writer:
            written = 0
            with structure.StructureReader(struct_file) as reader:
                for i, st in enumerate(reader):
                    if i == 0:
                        # first molecule should be the protein
                        writer.append(st)
                    elif st.title in common_molecules and st.property["i_i_glide_lignum"] == data.loc[st.title, "Lig#_%s_report" % name]:
                        writer.append(st)
                        written += 1
                    if written == n_mols:
                        break


if __name__ == "__main__":
    csv_input_file, structs_input, out_path, out_file, n_mols = parse_args()
    main(csv_input_file, structs_input, out_path, out_file, n_mols)
