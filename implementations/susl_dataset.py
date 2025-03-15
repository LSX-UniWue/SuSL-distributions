from typing import Sequence, Callable, Dict

from torch import Tensor, LongTensor
from torch.utils.data import Dataset

from susl_base.data.susl_dataset import DatasetFacade, LabeledDatasetFacade


class TransformableDatasetFacade(DatasetFacade):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        input_transform: Callable[[Tensor], Tensor] = lambda x: x / x.sum(),
        target_transform: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> None:
        super().__init__(dataset=dataset, indices=indices)
        self.__input_transform = input_transform
        self.__target_transform = target_transform

    def __getitem__(self, index) -> Dict[str, Tensor]:
        sample = super().__getitem__(index)
        return {
            "x_u": self.__input_transform(sample["x_u"]),
            "x_u_target": self.__target_transform(sample["x_u_target"]),
        }


class TransformableLabeledDatasetFacade(LabeledDatasetFacade):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        class_mapper: LongTensor | int,
        input_transform: Callable[[Tensor], Tensor] = lambda x: x / x.sum(),
        target_transform: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> None:
        super().__init__(dataset=dataset, indices=indices, class_mapper=class_mapper)
        self.__input_transform = input_transform
        self.__target_transform = target_transform

    def __getitem__(self, index) -> Dict[str, Tensor]:
        sample = super().__getitem__(index)
        return {
            "x_l": self.__input_transform(sample["x_l"]),
            "x_l_target": self.__target_transform(sample["x_l_target"]),
            "y_l": sample["y_l"],
        }
