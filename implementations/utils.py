from collections import defaultdict
from typing import Dict, Sequence

from torch import Tensor, LongTensor, tensor
from torch.utils.data import Dataset, Subset


class DatasetFacade(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset=dataset, indices=indices)

    def __getitem__(self, index) -> Dict[str, Tensor]:
        sample, _ = super().__getitem__(index)
        target = sample.detach().clone()
        return {"x_u": sample / sample.sum(), "x_u_target": target}


class LabeledDatasetFacade(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], class_mapper: LongTensor | int) -> None:
        super().__init__(dataset=dataset, indices=indices)
        if isinstance(class_mapper, int):
            from torch import arange

            self.__class_mapper = arange(class_mapper).long()
        else:
            self.__class_mapper = class_mapper

    def __getitem__(self, index) -> Dict[str, Tensor]:
        sample, label = super().__getitem__(index)
        target = sample.detach().clone()
        return {"x_l": sample / sample.sum(), "x_l_target": target, "y_l": self.__class_mapper[label]}


def create_susl_dataset(
    dataset: Dataset, num_labels: float = 0.2, classes_to_hide: Sequence[int] = None
) -> (Dataset, Dataset):
    # Find ids for each class
    ids = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        ids[label].append(i)
    ids_labeled, ids_unlabeled = [], []
    class_mapper = tensor(len(ids) * [0]).long()
    # Hide classes
    if classes_to_hide is not None:
        for class_to_hide in classes_to_hide:
            ids_unlabeled.extend(ids.pop(class_to_hide))
            class_mapper[class_to_hide] = -1
    # Create ssl
    for v in ids.values():
        size = max(1, int(len(v) * num_labels))
        ids_labeled.extend(v[:size])
        ids_unlabeled.extend(v[size:])
    # Update class mappings
    for i, k in enumerate(sorted(ids.keys())):
        class_mapper[k] = i
    # Return facades
    return LabeledDatasetFacade(dataset, ids_labeled, class_mapper=class_mapper), DatasetFacade(dataset, ids_unlabeled)
