import os
from glob import glob
import buteo as beo
import numpy as np
import pandas as pd
import tabulate

ARGMAX = False
MERGE = False
EXP1 = False

if MERGE:
    FOLDER = "./predictions/"
    FOLDER_OUT = "./visualisations/"

    for path in glob(os.path.join(FOLDER, "pred_EXP1-0*.tif")):
        name = os.path.splitext(os.path.basename(path))[0]
        name = "_".join(name.split("_")[:-1])
        outpath = os.path.join(FOLDER_OUT, f"{name}.tif")

        image_paths = glob(path.replace("EXP1-0", "EXP1-*"))

        pred = None
        for idx, image_path in enumerate(image_paths):
            if idx == 0:
                pred = beo.raster_to_array(image_path, filled=True, fill_value=0.0, cast=np.float32)
            else:
                pred += beo.raster_to_array(image_path, filled=True, fill_value=0.0, cast=np.float32)

        pred = pred / len(image_paths)

        beo.array_to_raster(
            pred,
            reference=path,
            out_path=outpath,
        )

if ARGMAX:
    FOLDER = "./visualisations/"
    FOLDER_OUT = "./visualisations/"

    for path in glob(os.path.join(FOLDER, "pred_EXP1-0_SOFT*.tif")):
        name = os.path.splitext(os.path.basename(path))[0]
        outpath = os.path.join(FOLDER_OUT, f"{name}_argmax.tif")

        if os.path.exists(outpath):
            continue

        arr = beo.raster_to_array(path, filled=True, fill_value=0.0, cast=np.float32)

        beo.array_to_raster(
            np.argmax(arr, axis=2, keepdims=True).astype(np.uint8),
            reference=path,
            out_path=outpath,
        )

if EXP1:
    FOLDER = "./logs/"

    dst = pd.DataFrame(columns=["soft", "var", "smo", "method", "loss", "jac", "f1", "prec", "rec", "std", "std_jac", "std_f1", "std_prec", "std_rec"])
    for path in glob(os.path.join(FOLDER, "EXP1-0*")):
        tmp = pd.DataFrame(columns=["iter", "soft", "var", "smo", "method", "loss", "jac", "f1", "prec", "rec", "std", "std_jac", "std_f1", "std_prec", "std_rec"])

        for idx, path2 in enumerate(glob(path.replace("EXP1-0", "EXP1-*"))):
            name = path2.replace("kernel_half", "kernelhalf")
            name = name.split("_")
            name[-1] = name[-1].replace(".csv", "")
            name[0] = os.path.basename(name[0])

            iteration = name[0][-1]
            soft = name[1].replace("SOFT-", "")

            smo = "-"
            for n in name:
                if n in ["SMOOTH-0.0", "SMOOTH-0.1", "SMOOTH-0.2"]:
                    smo = n.replace("SMOOTH-", "")
                    break
            
            method = "-"
            for n in name:
                if n in ["PROT-half", "PROT-kernelhalf", "PROT-max", "PROT-None"]:
                    method = n.replace("PROT-", "")
                    if method == "None":
                        method = "No Protection"
                    elif method == "max":
                        method = "Max"
                    elif method == "kernelhalf":
                        method = "Kernel Half"
                    elif method == "half":
                        method = "Half"
                    break

            var = "-"
            for n in name:
                if n in ["VAR-True", "VAR-False"]:
                    var = n.replace("VAR-", "")
                    break

            try:
                csv = pd.read_csv(path2)
            except pd.errors.EmptyDataError:
                continue

            row = csv.iloc[-1].dropna()
            row["iter"] = iteration
            row["soft"] = soft
            row["method"] = method
            row["var"] = var
            row["smo"] = smo

            # Add row to dst without using dst.append
            tmp.loc[idx] = row

        # Cast to float
        values = ["loss", "jac", "f1", "prec", "rec"]
        tmp[values] = tmp[values].astype(float)

        # Compute mean and std along the columns dimensions
        mean = tmp[values].mean(axis=0)
        std = tmp[values].std(axis=0)

        try:
            # Add mean and std to dst
            new_row = [
                soft,
                var,
                smo,
                method,
                mean["loss"],
                mean["jac"],
                mean["f1"],
                mean["prec"],
                mean["rec"],
                std["loss"],
                std["jac"],
                std["f1"],
                std["prec"],
                std["rec"],
            ]
        
            dst.loc[len(dst)] = new_row
        except:
            import pdb; pdb.set_trace()

    dst.to_csv("./logs/experiment1.csv", index=False)

    table = pd.read_csv("./logs/experiment1.csv")
    table = table.sort_values(by=["soft", "var", "smo", "method"])
    
    # Drop the loss
    table = table.drop(columns=["loss"])
 
    # Cast to string and ensure four decimals after the comma
    table["jac"] = table["jac"].round(4).astype(str).str.pad(6, side="right", fillchar="0")
    table["f1"] = table["f1"].round(4).astype(str).str.pad(6, side="right", fillchar="0")
    table["prec"] = table["prec"].round(4).astype(str).str.pad(6, side="right", fillchar="0")
    table["rec"] = table["rec"].round(4).astype(str).str.pad(6, side="right", fillchar="0")

    table["std_jac"] = table["std_jac"].round(3).astype(str).str.pad(5, side="right", fillchar="0")
    table["std_f1"] = table["std_f1"].round(3).astype(str).str.pad(5, side="right", fillchar="0")
    table["std_prec"] = table["std_prec"].round(3).astype(str).str.pad(5, side="right", fillchar="0")
    table["std_rec"] = table["std_rec"].round(3).astype(str).str.pad(5, side="right", fillchar="0")

    table["jac"] = table["jac"] + " ± " + table["std_jac"]
    table["f1"] = table["f1"] + " ± " + table["std_f1"]
    table["prec"] = table["prec"] + " ± " + table["std_prec"]
    table["rec"] = table["rec"] + " ± " + table["std_rec"]

    # Drop the std
    table = table.drop(columns=["std", "std_jac", "std_f1", "std_prec", "std_rec"])

    # Rename columns
    table = table.rename(columns={
        "soft": "SoftLabels",
        "var": "VarianceScaling",
        "smo": "GlobalSmoothing",
        "method": "ProtectionMethod",
        "jac": "Jaccard",
        "f1": "F1",
        "prec": "Precision",
        "rec": "Recall",
    })

    # Reorder columns
    table = table[["SoftLabels", "GlobalSmoothing", "ProtectionMethod", "VarianceScaling", "Jaccard", "F1", "Precision", "Recall"]]

    # Order by F1
    table = table.sort_values(by=["SoftLabels", "GlobalSmoothing", "ProtectionMethod", "VarianceScaling"], ascending=False)

    # Save table
    table.to_csv("./logs/experiment1_table.csv", index=False, encoding="latin1")

    print(tabulate.tabulate(table, tablefmt = 'psql', showindex=False, headers="keys"))