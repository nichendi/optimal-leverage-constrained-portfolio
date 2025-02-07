import copy
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config, DataConfig, config
from wealth import get_wealth_trajectories
from objective import cumulative_quadratic_shortfall


class MarketHistoryDataset(Dataset):
    """
    Since the data is a high-dimensional tensor, we implement a custom dataset to allow indexing with the rebalancing step
    """

    def __init__(self, data, device: torch.device):
        self.data = data.to(device)

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(data_config: DataConfig, device: torch.device):
    data_dir = Path.cwd() / "data"
    data = torch.load(data_dir / data_config.dataname)
    dataset = MarketHistoryDataset(data, device=device)
    return dataset


def get_compute_device() -> torch.device:
    # get compute device, use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # otherwise, use mps if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def train(
    config: Config = config,
) -> torch.nn.Module:
    loss_min = config.optimizer.big_loss
    device = get_compute_device()
    print(f"Computing device is: {device}")
    # load data
    dataset = load_data(config.data, device=device)
    # initialize model
    model = config.model.get_model(device=device)
    # set up optimizer
    optimizer = config.optimizer.optimizer(
        model.parameters(), lr=config.optimizer.learning_rate
    )
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config.optimizer.decay_every,
        gamma=config.optimizer.decay_rate,
    )
    batchloader = DataLoader(
        dataset=dataset, batch_size=config.optimizer.batch_size, shuffle=False
    )
    start_time = time.time()
    # mini-batch gradient descent: update gradient after each mini-batch
    for t in range(config.optimizer.num_epoch):
        # Forward pass: compute predicted y by passing x to the model.
        model.train()
        for batch_data in batchloader:
            outputt = get_wealth_trajectories(
                batch_data,
                model,
                wealth_init=config.experiment.wealth_init,
                cash_injection=config.experiment.cash_injection,
                rebalance_frequency=config.experiment.rebalance_frequency,
                weights_benchmark=config.experiment.weight_benchmark,
            )
            loss = cumulative_quadratic_shortfall(
                outputt.wealth_trajectories,
                outputt.wealth_trajectories_benchmark,
                beta=config.experiment.beta,
            )
            if loss < loss_min:
                model_min = copy.deepcopy(model)
                loss_min = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print info
        if t % config.optimizer.info_every == 0:
            model.eval()
            print(
                f"epoch: {t:03d} | loss: {loss.item():.2f} | elapsed time(s): {time.time()-start_time:.1f}"
            )
        my_lr_scheduler.step()
    return model_min


if __name__ == "__main__":
    # train the model
    model = train()
    # save model parameters
    torch.save(model.state_dict(), "model.pt")
