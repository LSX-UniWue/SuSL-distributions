from collections import defaultdict
from typing import Sequence, Callable, Tuple

from torch import Tensor, int64, zeros
from torch.utils.data import Dataset

from .susl_dataset import TransformableDatasetFacade, TransformableLabeledDatasetFacade


def create_susl_dataset(
    dataset: Dataset,
    num_labels: float = 0.2,
    classes_to_hide: Sequence[int] = None,
    input_transform: Callable[[Tensor], Tensor] = lambda x: x / x.sum(),
    target_transform: Callable[[Tensor], Tensor] = lambda x: x,
) -> Tuple[Dataset, Dataset, Tensor]:
    # Find ids for each class
    ids = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        ids[label].append(i)
    ids_labeled, ids_unlabeled = [], []
    class_mapper = zeros(len(ids), dtype=int64)
    # Hide classes
    if classes_to_hide is not None:
        for class_to_hide in classes_to_hide:
            ids_unlabeled.extend(ids.pop(class_to_hide))
    # Create ssl
    for v in ids.values():
        size = max(1, int(len(v) * num_labels))
        ids_labeled.extend(v[:size])
        ids_unlabeled.extend(v[size:])
    # Update class mappings
    for i, k in enumerate(sorted(ids.keys())):
        class_mapper[k] = i
    for i, k in enumerate(sorted(classes_to_hide), start=len(ids)):
        class_mapper[k] = i
    # Return facades
    return (
        TransformableLabeledDatasetFacade(
            dataset,
            ids_labeled,
            class_mapper=class_mapper,
            input_transform=input_transform,
            target_transform=target_transform,
        ),
        TransformableDatasetFacade(
            dataset, ids_unlabeled, input_transform=input_transform, target_transform=target_transform
        ),
        class_mapper,
    )
