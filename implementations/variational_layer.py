from torch import Tensor
from torch.distributions import NegativeBinomial
from torch.nn import Module, Linear

from susl_base.networks.variational_layer import VariationalLayer


class NegativeBinomialVariationalLayer(VariationalLayer, NegativeBinomial):
    def __init__(self, feature_extractor: Module, module_init=Linear, **kwargs) -> None:
        VariationalLayer.__init__(self, feature_extractor)
        NegativeBinomial.__init__(self, logits=0, total_count=0)
        self.__log_counts = module_init(**kwargs)
        self.__logits = module_init(**kwargs)

    def forward(self, x: Tensor, y: Tensor = None) -> None:
        latent = VariationalLayer.forward(self, x, y)
        # TODO: clamp counts???
        NegativeBinomial.__init__(self, total_count=self.__log_counts(latent).exp(), logits=self.__logits(latent))
