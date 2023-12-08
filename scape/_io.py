import pandas as pd
import numpy as np
import os
import zipfile
import lzma
import tempfile


def compress(file_path, zip_file_path=None, delete=False):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if zip_file_path is None:
        zip_file_path = file_path + ".zip"

    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.compresslevel = 9
        zipf.write(file_path, os.path.basename(file_path))
    if delete:
        os.remove(file_path)
    return zip_file_path


def compress_folder(folder, file):
    with lzma.open(file, "wb") as f:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                # Write the file's relative path and data to the compressed file
                with open(file_path, "rb") as file_data:
                    f.write(file_data.read())


def load(file, import_method):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Decompress the .xz file into the temporary directory
        with lzma.open(file, "rb") as f:
            content = f.read()
            temp_file_path = os.path.join(tmpdir, "decompressed_content")
            with open(temp_file_path, "wb") as f_out:
                f_out.write(content)
        import_method(tmpdir)


def load_slogpvals(file):
    df = pd.read_parquet(file)
    df.drop(columns=["sm_lincs_id", "SMILES", "control"], inplace=True)
    df.set_index(["cell_type", "sm_name"], inplace=True)
    return df


def load_lfc(pseudo_counts_file, control="Dimethyl Sulfoxide", group=True):
    df_pseudo_counts = pd.read_parquet(pseudo_counts_file)
    idx_control = df_pseudo_counts.index.get_level_values("sm_name") == control
    df_pseudo_control = df_pseudo_counts.loc[idx_control].droplevel("sm_name")
    df_pseudo_treatment = df_pseudo_counts.loc[~idx_control]

    df_treat = pd.DataFrame(index=df_pseudo_treatment.index).join(
        df_pseudo_control.droplevel(["plate", "well"]),
        on=["cell_type", "donor", "library"],
        how="left",
    )
    df_pseudo_fold_change = -np.log2(df_treat + 1) + np.log2(df_pseudo_treatment + 1)
    if group:
        df_pseudo_fold_change = df_pseudo_fold_change.groupby(
            ["cell_type", "sm_name"]
        ).mean()
    return df_pseudo_fold_change
