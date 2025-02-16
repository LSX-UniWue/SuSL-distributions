from functools import reduce
from pathlib import PurePath
from typing import Iterable, Tuple

from numpy import where, unique, ndarray, array
from numpy.random import default_rng
from scanpy import read_h5ad


def train_val_test_split(
    classes: ndarray, indices: ndarray, test_size: float = 0.1, with_val: bool = True
) -> Tuple[ndarray, ndarray, ndarray]:
    unique_classes, class_counts = unique(classes[indices], return_counts=True)
    train_idx, val_idx, test_idx = [], [], []
    rng = default_rng(42)
    for c, cc in zip(unique_classes, class_counts):
        if cc < 3:
            print(f"Class {c} has less than 3 samples.")
            continue
        idx, *_ = where(classes == c)
        rng.shuffle(idx)
        test_count = max(1, int(test_size * cc))
        val_count = test_count if with_val else 0
        train_count = cc - test_count - val_count
        train_idx.extend(idx[:train_count])
        val_idx.extend(idx[train_count : train_count + val_count])
        test_idx.extend(idx[train_count + val_count :])
    return array(train_idx), array(val_idx), array(test_idx)


def create_splits(file_path: str, obs_column: str, target_file_path: str, test_size: float = 0.1) -> None:
    # Read file
    data = read_h5ad(file_path)
    # Create split indices
    # idx, *_ = where(data.obs[obs_column].astype(str) != 'nan')
    idx, *_ = where(data.obs[obs_column] != "")
    idx_train, idx_val, idx_test = train_val_test_split(
        classes=data.obs[obs_column].to_numpy(), indices=idx, test_size=test_size
    )
    # Print stats
    print("train:", unique(data.obs[obs_column].iloc[idx_train], return_counts=True))
    print("val:", unique(data.obs[obs_column].iloc[idx_val], return_counts=True))
    print("test:", unique(data.obs[obs_column].iloc[idx_test], return_counts=True))
    # Set new columns
    # 0 = train_u, 1 = train, 2 = val, 3 = test
    data.obs["train_mode"] = 0
    data.obs.loc[data.obs.index[idx_train], "train_mode"] = 1
    data.obs.loc[data.obs.index[idx_val], "train_mode"] = 2
    data.obs.loc[data.obs.index[idx_test], "train_mode"] = 3
    # Write file
    data.write_h5ad(compression="lzf", filename=PurePath(target_file_path))


def create_splits_mixed(
    file_paths: Iterable[str],
    file_paths_test: Iterable[str],
    obs_column: str,
    target_file_path: str,
    test_size: float = 0.1,
) -> None:
    # Read files
    # Read train files
    data = [read_h5ad(path) for path in file_paths]
    data = reduce(lambda left, right: left.concatenate(right, join="outer"), data)
    # Create split indices
    idx, *_ = where(data.obs[obs_column] != "")
    idx_train, _, idx_val = train_val_test_split(
        classes=data.obs[obs_column].to_numpy(), indices=idx, test_size=test_size, with_val=False
    )
    # Print stats
    print("train:", unique(data.obs[obs_column].iloc[idx_train], return_counts=True))
    print("val:", unique(data.obs[obs_column].iloc[idx_val], return_counts=True))
    # Read test files
    data_test = [read_h5ad(path) for path in file_paths_test]
    data_test = reduce(lambda left, right: left.concatenate(right, join="outer"), data_test)
    # Filter unlabeled data for test
    data_test = data_test[data_test.obs[obs_column] != ""]
    print("test:", unique(data_test.obs[obs_column], return_counts=True))
    # Set new columns
    # 0 = train_u, 1 = train, 2 = val, 3 = test
    # Use test as unlabelled dataset, too, i.e., append it 2 times
    data = data.concatenate(data_test, join="outer")
    size = data.n_obs
    data = data.concatenate(data_test, join="outer")
    data.obs["train_mode"] = 0
    data.obs.loc[data.obs.index[idx_train], "train_mode"] = 1
    data.obs.loc[data.obs.index[idx_val], "train_mode"] = 2
    data.obs.loc[size:, "train_mode"] = 3
    # Write file
    data.write_h5ad(compression="lzf", filename=PurePath(target_file_path))


if __name__ == "__main__":
    # Tabular Muris
    create_splits(
        file_path="../datasets_raw/tabula-muris-droplet.h5ad",
        obs_column="cell_ontology_class",
        target_file_path="../datasets_processed/tabula-muris-droplet.h5ad",
    )
    # Tabular Muris Senis
    create_splits(
        file_path="../datasets/RAW/Gene_Mouse/tabula-muris-senis-droplet-official-raw-obj.h5ad",
        obs_column="cell_ontology_class",
        target_file_path="../datasets_processed/tabula-muris-senis.h5ad",
    )
    # Mixture
    create_splits_mixed(
        file_paths=("../datasets/RAW/Gene_Mouse/tabula-muris-senis-droplet-official-raw-obj.h5ad",),
        file_paths_test=("../datasets_raw/tabula-muris-droplet.h5ad",),
        obs_column="cell_ontology_class",
        target_file_path="../datasets_processed/tabula-muris-mixture.h5ad",
    )
