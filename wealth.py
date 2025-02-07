from dataclasses import dataclass
import torch


@dataclass
class Trajectories:
    wealth_trajectories: torch.Tensor
    wealth_trajectories_benchmark: torch.Tensor
    allocation_trajectories: torch.Tensor


def get_scaled_features(
    wealth: torch.Tensor,
    wealth_benchmark: torch.Tensor,
    tau: torch.Tensor,
    wealth_init: float,
):
    # a simple scaling of state variables to accelerate training
    return torch.cat([wealth / wealth_init, wealth_benchmark / wealth_init, tau], dim=1)


def get_wealth_trajectories(
    data: torch.Tensor,
    model: torch.nn.Module,
    wealth_init: float,
    cash_injection: float,
    rebalance_frequency: int,
    weights_benchmark: list[float],
) -> Trajectories:
    """
    Simple wealth evolution function that calculates the wealth trajectories given model and data trajectories.
    This function accomodates cash injections, and allow different rebalancing frequencies.
    Additional functionalities such as transaction cost and short selling cost can be added easily.

    Parameters:
        data: a large tensor of size (num_paths, num_secs, num_periods)
        model: a neural network that outputs the allocation weights
        wealth_init: initial wealth of the portfolio
        cash_injection: annual cash injection amount at rebalancing
        rebalance_frequency: how many rebalances in a year
        weights_benchmark: weight vector of the constant benchmark portfolio
    """
    num_paths, num_assets, num_periods = data.size()
    assert (
        len(weights_benchmark) == num_assets
    ), "Benchmark weights should have the same length as the number of assets!"

    # get compute device, i.e., cuda, mps, or cpu
    # all tensors involved need to be on the same device
    device = data.device
    # assuming monthly data
    # this can be easily modified if data frequency changes
    num_periods_yearly = 12

    # initialize wealth vectors for both portfolios at starting time
    wealth_vector = torch.ones(num_paths, 1).to(device) * (wealth_init)
    wealth_vector_bm = wealth_vector.clone()
    # initialize wealth trajectories, dimension of (num_paths x num_periods), for both portfolios
    wealth_traj = torch.zeros(
        num_paths, num_periods // num_periods_yearly * rebalance_frequency + 1
    ).to(device)
    wealth_traj_bm = torch.zeros(
        num_paths, num_periods // num_periods_yearly * rebalance_frequency + 1
    ).to(device)
    # initialize allocation trajectories, dimension of (num_paths x num_periods), for the active portfolio
    allocation_traj = torch.zeros(
        num_paths,
        num_assets,
        num_periods // num_periods_yearly * rebalance_frequency + 1,
    ).to(device)
    # expand dimension of benchmark allocation vector for computation
    allocation_bm = (torch.tensor(weights_benchmark) * torch.ones(num_paths, 1)).to(
        device
    )
    # index for tracking which rebalancing period we are in
    index = 0
    # main loop - calculate wealth evolution over rebalancing periods
    for i in range(0, num_periods):
        if i % (num_periods_yearly // rebalance_frequency) == 0:
            # calculate features at the beginning of each rebalancing period
            tau = torch.ones(num_paths, 1).to(device) * (num_periods - i) / num_periods
            features = get_scaled_features(
                wealth_vector, wealth_vector_bm, tau, wealth_init
            )
            # calculate allocation weights
            allocation = model(features)
            # calculate wealth fractions in each asset
            wealth_fractions = wealth_vector * allocation
            wealth_fractions_bm = wealth_vector_bm * allocation_bm
            # store the current allocation and wealth vectors into trajectories
            allocation_traj[:, :, index] = allocation
            wealth_traj[:, index : index + 1] = wealth_vector
            wealth_traj_bm[:, index : index + 1] = wealth_vector_bm
            index = index + 1
        # calculate wealth fraction growth over period
        wealth_fractions = wealth_fractions * (data[:, :, i] + 1)
        wealth_fractions_bm = wealth_fractions_bm * (data[:, :, i] + 1)
        # calculate portfolio wealth by combining fractions and adding cash injection
        wealth_vector = (
            torch.sum(wealth_fractions, dim=1, keepdim=True).clamp(min=0)
            + cash_injection / rebalance_frequency
        )
        wealth_vector_bm = (
            torch.sum(wealth_fractions_bm, dim=1, keepdim=True).clamp(min=0)
            + cash_injection / rebalance_frequency
        )
    # store final wealth and allocation vector to the trajectories
    wealth_traj[:, index : index + 1] = wealth_vector
    wealth_traj_bm[:, index : index + 1] = wealth_vector_bm
    allocation_traj[:, :, index] = allocation

    return Trajectories(
        wealth_trajectories=wealth_traj,
        wealth_trajectories_benchmark=wealth_traj_bm,
        allocation_trajectories=allocation_traj,
    )
