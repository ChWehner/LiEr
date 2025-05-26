import torch

from src.baseline import ReactiveBaseline
from src.utility import discount_cumsum

class Reward:
    """Base class for the reward."""

    def __init__(self, lambda_factor: float, gamma: float = 0.99):
        """Inits the reward.

        Args:
            lambda_factor (float): The sensitivity of the baseline to changes in the reward.
            gamma (float, optional): The discount factor. Defaults to 0.99.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.rewards = torch.empty(
            0, device=self.device
        )  # shape [batch_size, num_reward(=num_hops)]
        self.uncertainty = 0
        self.discounted_rewards = torch.empty(
            0, device=self.device
        )  # shape [batch_size, num_rewards(=num_hops)]
        self.baseline = ReactiveBaseline(lambda_factor=lambda_factor)
        self.training_flag = True

    def discount_rewards(self) -> None:
        """Discounts rewards. Sets self.discounted_rewards."""
        self.discounted_rewards = discount_cumsum(self.rewards, self.gamma, self.device).detach()
        return

    def update_baseline(self) -> None:
        """Triggers the baseline update."""
        self.baseline.update(
            target=torch.mean(self.discounted_rewards)
        )  # mean over flatten rewards
        return

    def reset(self, gamma=0.99) -> None:
        """resets RewardEstimator. The function is called at the beginning or end of each epoch.
        Also allows to alter the gamma value and the interactive parameter.

        Args:
            gamma (float, optional): Discount factor of the reward estimator. Defaults to 0.99.
        """
        self.rewards = torch.empty(
            0, device=self.device
        )  # shape = shape [batch_size x hops, reward(=1)]
        self.uncertainty = 0
        self.discounted_rewards = torch.empty(0, device=self.device)
        self.gamma = gamma
        return
    
    def get_hits(self, goals: torch.Tensor, current_states: torch.Tensor, shape: tuple[int]) -> int:
        """calculates the hits

        Args:
            goals (torch.Tensor): the goals of all instances of a batch.
            current_states (torch.Tensor): the current states of all instances of a batch.
            shape (tuple[int]): Gives us the number of rollouts per batch.

        Returns:
            int: the number of hits of query that once correctly answered by at least on rollout
        """
        hits = torch.eq(goals, current_states.unsqueeze(dim=1)).reshape(shape)
        hits = torch.any(hits, dim=1)
        return torch.sum(hits).item()


class BasicReward(Reward):
    """The basic and non-interative reward. Manly here to have a baseline for the implementation."""

    def __init__(self, lambda_factor:float, gamma:float=0.99):
        """its the basic reward.

        Args:
            lambda_factor (float): The sensitivity of the baseline to changes in the reward.
            gamma (float, optional): The discount factor. Defaults to 0.99.
        """
        super().__init__(lambda_factor, gamma)

    def __call__(
        self, goals: torch.Tensor, current_states: torch.Tensor, ultimate_hop: bool
    ):
        """Calculates and saves the reward on call of the instance of the basic reward estimator.

        Args:
            goals (torch.Tensor): the goals of all instances of a batch.
            current_states (torch.Tensor): the current states of all instances of a batch.
            ultimate_hop (bool): flag if this is the ultimate (i.e., last) hop (step) of the model.
        """

        if ultimate_hop:
            rewards = torch.eq(goals, current_states.unsqueeze(dim=1))

        else:
            rewards = torch.zeros(
                (goals.size(0), 1), device=self.device, requires_grad=False
            )

        self.rewards = torch.cat((self.rewards, rewards), axis=1)

        return
