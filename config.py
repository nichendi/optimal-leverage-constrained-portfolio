from dataclasses import dataclass
import torch
import models


@dataclass
class DataConfig:
    dataname: str


@dataclass
class ModelConfig:
    name: str
    params: dict

    def get_model(self, device: torch.device):
        return getattr(models, self.name)(**self.params).to(device)


@dataclass
class OptimizerConfig:
    optimizer: torch.optim.Optimizer
    learning_rate: float
    batch_size: int
    decay_every: int
    decay_rate: float
    num_epoch: int
    info_every: int
    big_loss: float = 1e10


@dataclass
class ExperimentConfig:
    cash_injection: float
    rebalance_frequency: int
    wealth_init: float
    weight_benchmark: list[float]
    beta: float


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    experiment: ExperimentConfig


config = Config(
    data=DataConfig(dataname="random_data.pt"),
    model=ModelConfig(
        name="RCNN",
        params={"num_assets": 4, "num_hidden_nodes": 10, "p_max": 1.3},
    ),
    optimizer=OptimizerConfig(
        optimizer=torch.optim.Adam,
        learning_rate=0.02,
        batch_size=1000,
        decay_every=200,
        decay_rate=0.95,
        num_epoch=10000,
        info_every=100,
    ),
    experiment=ExperimentConfig(
        cash_injection=0,
        rebalance_frequency=12,
        wealth_init=100,
        weight_benchmark=[
            0.7,
            0.3,
            0,
            0,
        ],  # size and value need to match the number and order of assets
        beta=0.03,
    ),
)
