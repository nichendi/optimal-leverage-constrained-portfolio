import torch


def cumulative_quadratic_shortfall(
    wealth_traj: torch.Tensor,
    wealth_traj_bm: torch.Tensor,
    beta=0.05,
) -> torch.Tensor:
    """
    Cumulative quadratic shortfall (CS) objective function

    Parameters:
        wealth_traj: a tensor of (num_paths, num_periods)
        wealth_traj_bm: a tensor of (num_paths, num_periods)
        beta: annual outperformance target
    """
    # wealth_traj is a tensor with dimension of (num_paths, num_periods)
    device = wealth_traj.device
    num_paths, num_periods = wealth_traj.size()
    num_years = num_periods / 12
    assert (
        wealth_traj.size() == wealth_traj_bm.size()
    ), "The wealth trajectories should have the same size!"
    dt = num_years / num_periods
    factor = (
        torch.linspace(dt, num_years, steps=num_periods).repeat(num_paths, 1).to(device)
    )
    factor = torch.exp(beta * factor)
    elevated_target_traj = wealth_traj_bm * factor
    downside_squared = torch.square((wealth_traj - elevated_target_traj).clamp(max=0))
    sum = torch.sum(downside_squared * dt, dim=1, keepdim=True)
    return sum.mean()
