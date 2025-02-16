from lightning import Trainer
from torch.nn import Sequential, Linear, ReLU, Identity
from torchmetrics import MetricCollection

from datasets.tabula_muris import TabulaMurisDataset
from implementations.utils import create_susl_dataset
from implementations.variational_layer import NegativeBinomialVariationalLayer
from susl_base.data.data_module import SemiUnsupervisedDataModule
from susl_base.metrics.cluster_and_label import ClusterAccuracy
from susl_base.networks.gmm_dgm import EntropyRegularizedGaussianMixtureDeepGenerativeModel
from susl_base.networks.latent_layer import LatentLayer
from susl_base.networks.lightning import LightningGMMModel
from susl_base.networks.losses import GaussianMixtureDeepGenerativeLoss
from susl_base.networks.variational_layer import GaussianVariationalLayer


def run() -> None:
    train_dataset = TabulaMurisDataset(stage="train")
    classes_to_hide = [
        train_dataset.classes.index(c) for c in ["B cell", "hepatocyte", "keratinocyte", "mesenchymal cell"]
    ]
    train_dataset_labeled, train_dataset_unlabeled = create_susl_dataset(
        dataset=train_dataset,
        num_labels=0.2,
        classes_to_hide=classes_to_hide,
    )
    validation_dataset = TabulaMurisDataset(stage="val")
    test_dataset = TabulaMurisDataset(stage="test")
    datamodule = SemiUnsupervisedDataModule(
        train_dataset_labeled=train_dataset_labeled,
        train_dataset_unlabeled=train_dataset_unlabeled,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=128,
    )
    # Create model
    n_x, n_y, n_z = 2537, len(test_dataset.classes) + 20, 50
    hidden_dim = 500
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
    )
    print(model)
    # Create trainer and run
    lt_model = LightningGMMModel(
        model=model,
        loss_fn=GaussianMixtureDeepGenerativeLoss(gamma=5e-5),
        val_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_y, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_y, average="macro"),
            },
            prefix="val_",
        ),
        test_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_y, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_y, average="macro"),
            },
            prefix="test_",
        ),
    )
    trainer = Trainer(max_epochs=150, check_val_every_n_epoch=1)
    trainer.fit(model=lt_model, datamodule=datamodule)
    trainer.test(model=lt_model, datamodule=datamodule)


if __name__ == "__main__":
    run()
