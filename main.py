from lightning import Trainer
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, Identity
from torchmetrics import MetricCollection

from datasets.tabula_muris import TabulaMurisDataset
from implementations.lightning import LightningGMMModelWeightDecay
from implementations.utils import create_susl_dataset
from implementations.variational_layer import NegativeBinomialVariationalLayer
from susl_base.data.data_module import SemiUnsupervisedDataModule
from susl_base.metrics.cluster_and_label import ClusterAccuracy
from susl_base.networks.gmm_dgm import EntropyRegularizedGaussianMixtureDeepGenerativeModel
from susl_base.networks.latent_layer import LatentLayer
from susl_base.networks.losses import GaussianMixtureDeepGenerativeLoss
from susl_base.networks.variational_layer import GaussianVariationalLayer


# Copied from susl_base.main
def get_prior(n_l: int, n_aug: int) -> Tensor:
    from torch import tensor

    # Unsupervised
    if n_l <= 0:
        return tensor(n_aug * [1 / n_aug])
    # (Semi-)Supervised
    if n_aug <= 0:
        return tensor(n_l * [1 / n_l])
    # SuSL
    return 0.5 * tensor(n_l * [1 / n_l] + n_aug * [1 / n_aug])


def run() -> None:
    min_counts, min_genes = 1000, 500
    train_dataset = TabulaMurisDataset(stage="train", min_counts=min_counts, min_genes=min_genes)
    classes_to_hide = [
        train_dataset.classes.index(c) for c in ["B cell", "hepatocyte", "keratinocyte", "mesenchymal cell"]
    ]
    train_dataset_labeled, train_dataset_unlabeled = create_susl_dataset(
        dataset=train_dataset,
        num_labels=0.2,
        classes_to_hide=classes_to_hide,
    )
    validation_dataset = TabulaMurisDataset(stage="val", min_counts=min_counts, min_genes=min_genes)
    test_dataset = TabulaMurisDataset(stage="test", min_counts=min_counts, min_genes=min_genes)
    datamodule = SemiUnsupervisedDataModule(
        train_dataset_labeled=train_dataset_labeled,
        train_dataset_unlabeled=train_dataset_unlabeled,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=128,
    )
    # Create model
    n_l, n_aug, n_classes = len(test_dataset.classes) - len(classes_to_hide), 40, len(test_dataset.classes)
    n_x, n_y, n_z = 2537, n_l + n_aug, 50
    hidden_dim = 100
    q_y_x_module = Sequential(
        Linear(in_features=n_x, out_features=hidden_dim),
        ReLU(),
        Linear(in_features=hidden_dim, out_features=hidden_dim),
        ReLU(),
        Linear(in_features=hidden_dim, out_features=n_y),
    )
    # Change for Gaussian or Bernoulli, depending on experiment for decoder
    p_x_z_module = NegativeBinomialVariationalLayer(
        feature_extractor=Sequential(
            Linear(in_features=n_z, out_features=hidden_dim),
            ReLU(),
            Linear(in_features=hidden_dim, out_features=hidden_dim),
            ReLU(),
        ),
        out_features=n_x,
        in_features=hidden_dim,
    )
    p_z_y_module = GaussianVariationalLayer(feature_extractor=Identity(), out_features=n_z, in_features=n_y)
    q_z_xy_module = GaussianVariationalLayer(
        feature_extractor=LatentLayer(
            pre_module=Sequential(
                Linear(in_features=n_x, out_features=hidden_dim),
                ReLU(),
                Linear(in_features=hidden_dim, out_features=hidden_dim),
                ReLU(),
            )
        ),
        out_features=n_z,
        in_features=hidden_dim + n_y,
    )
    model = EntropyRegularizedGaussianMixtureDeepGenerativeModel(
        n_y=n_y,
        n_z=n_z,
        n_x=n_x,
        q_y_x_module=q_y_x_module,
        p_x_z_module=p_x_z_module,
        p_z_y_module=p_z_y_module,
        q_z_xy_module=q_z_xy_module,
        log_priors=get_prior(n_l=n_l, n_aug=n_aug).log(),
    )
    print(model)
    # Create trainer and run
    lt_model = LightningGMMModelWeightDecay(
        model=model,
        loss_fn=GaussianMixtureDeepGenerativeLoss(gamma=5e-5),
        val_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_classes, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_classes, average="macro"),
            },
            prefix="val_",
        ),
        test_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_classes, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_classes, average="macro"),
            },
            prefix="test_",
        ),
        weight_decay=1e-6,
    )
    trainer = Trainer(max_epochs=150, check_val_every_n_epoch=2)
    trainer.fit(model=lt_model, datamodule=datamodule)
    trainer.test(model=lt_model, datamodule=datamodule)


if __name__ == "__main__":
    run()
