import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import json

def create_splits(csv_path="dataset_manifest.csv"):
    df = pd.read_csv(csv_path)
    df = df.pivot(index=["cert_id", "grade"], columns="side", values="path").reset_index()
    df["grade"] = df["grade"].astype(int)

    skf = StratifiedGroupKFold(n_splits=10)
    splits = next(skf.split(df, df["grade"], df["cert_id"]))
    train_idx, temp_idx = splits
    val_size = int(0.15 * len(temp_idx))
    val_idx, test_idx = temp_idx[:val_size], temp_idx[val_size:]

    splits_dict = {
        "train": df.iloc[train_idx].to_dict(orient="records"),
        "val":   df.iloc[val_idx].to_dict(orient="records"),
        "test":  df.iloc[test_idx].to_dict(orient="records"),
    }
    json.dump(splits_dict, open("splits.json","w"), indent=2)

if __name__ == "__main__":
    create_splits()