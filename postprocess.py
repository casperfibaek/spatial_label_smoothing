import os
from glob import glob
import buteo as beo
import numpy as np
import pandas as pd

ARGMAX = False
EXP1 = True
EXP2 = False


if ARGMAX:
    FOLDER = "./predictions/"

    for path in glob(os.path.join(FOLDER, "*.tif")):
        outpath = path.replace(".tif", "_argmax.tif")
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

    dst = pd.DataFrame(columns=["loss_func", "soft", "prot", "kr", "kc", "ks", "var", "smo", "loss", "jac", "f1", "prec", "rec", "std", "std_jac", "std_f1", "std_prec", "std_rec"])
    for path in glob(os.path.join(FOLDER, "*0_LOSS-cross_entropy_SOFT-True_PROT-*")):
        tmp = pd.DataFrame(columns=["iter", "loss_func", "soft", "prot", "kr", "kc", "ks", "var", "smo", "loss", "jac", "f1", "prec", "rec"])

        for idx, path2 in enumerate(glob(path.replace("0_LOSS", "*_LOSS"))):
            name = path2.split("_")
            iteration = name[0][-1]
            loss_func = (name[1] + name[2]).replace("LOSS-", "")
            soft = name[3].replace("SOFT-", "")
            prot = name[4].replace("PROT-", "")
            kr = name[5].replace("KR-", "")
            kc = name[6].replace("KC-", "")
            ks = name[7].replace("KS-", "")
            var = name[8].replace("VAR-", "").replace(".csv", "")

            try:
                csv = pd.read_csv(path2)
            except pd.errors.EmptyDataError:
                continue

            row = csv.iloc[-1].dropna()
            row["iter"] = iteration
            row["loss_func"] = loss_func
            row["soft"] = soft
            row["prot"] = prot
            row["kr"] = kr
            row["kc"] = kc
            row["ks"] = ks
            row["var"] = var
            row["smo"] = "NA"

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
                loss_func,
                soft,
                prot,
                kr,
                kc,
                ks,
                var,
                "NA",
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


    for path in glob(os.path.join(FOLDER, "0_LOSS-cross_entropy_SOFT-False_SMOOTH-*")):
        tmp = pd.DataFrame(columns=["iter", "loss_func", "soft", "prot", "kr", "kc", "ks", "var", "smo", "loss", "jac", "f1", "prec", "rec"])

        for idx, path2 in enumerate(glob(path.replace("0_LOSS", "*_LOSS"))):
            name = path2.split("_")
            iteration = name[0][-1]
            loss_func = (name[1] + name[2]).replace("LOSS-", "")
            soft = False
            prot = "NA"
            kr = "NA"
            kc = "NA"
            ks = "NA"
            var = "NA"
            smo = name[4].replace("SMOOTH-", "").replace(".csv", "")

            try:
                csv = pd.read_csv(path2)
            except pd.errors.EmptyDataError:
                continue

            row = csv.iloc[-1].dropna()
            row["iter"] = iteration
            row["loss_func"] = loss_func
            row["soft"] = soft
            row["prot"] = prot
            row["kr"] = kr
            row["kc"] = kc
            row["ks"] = ks
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
                loss_func,
                soft,
                prot,
                kr,
                kc,
                ks,
                var,
                smo,
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


if EXP2:

 FOLDER = "./logs/"

dst = pd.DataFrame(columns=["loss_func", "soft", "var", "smo", "loss", "jac", "f1", "prec", "rec", "std", "std_jac", "std_f1", "std_prec", "std_rec"])
for path in glob(os.path.join(FOLDER, "EXP2-0*")):
    tmp = pd.DataFrame(columns=["iter", "loss_func", "soft", "var", "smo", "loss", "jac", "f1", "prec", "rec", "std", "std_jac", "std_f1", "std_prec", "std_rec"])

    for idx, path2 in enumerate(glob(path.replace("EXP2-0", "EXP2-*"))):

        name = path2.split("_")
        iteration = name[0][-1]
        if "cross_entropy" in name:
            loss_func = (name[1] + name[2]).replace("LOSS-", "")
            soft = name[3].replace("SOFT-", "")
            if soft == "True":
                var = "NA"
                smo = name[4].replace("SMO-", "").replace(".csv", "")
            else:
                var = name[4].replace("VAR-", "").replace(".csv", "")
                smo = "NA"
        else:
            loss_func = name[1].replace("LOSS-", "")
            soft = name[2].replace("SOFT-", "")
            if soft == "True":
                var = "NA"
                smo = name[4].replace("SMO-", "").replace(".csv", "")
            else:
                var = name[4].replace("VAR-", "").replace(".csv", "")
                smo = "NA"

        try:
            csv = pd.read_csv(path2)
        except pd.errors.EmptyDataError:
            continue

        row = csv.iloc[-1].dropna()
        row["iter"] = iteration
        row["loss_func"] = loss_func
        row["soft"] = soft
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

    # Add mean and std to dst
    new_row = [
        loss_func,
        soft,
        var,
        smo,
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

    dst.to_csv("./logs/experiment2.csv", index=False)