import torch


class Baseline:
    """Baseline class - just there for the case that different baselines shall be implemented."""

    def get_baseline_value(self) -> None:
        pass

    def update(self) -> None:
        pass


class ReactiveBaseline(Baseline):
    """Reactive Baseline - baseline with rolling discounted reward."""
    def __init__(self, lambda_factor: float = 0.02):
        """inits baseline with rolling reward.

        Args:
            lambda_factor (float, optional): sensitiy of the baseline to changes in reward. Defaults to 0.02.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_factor = lambda_factor
        self.b = torch.tensor(0.0, requires_grad=False, device=self.device)

    def get_baseline_value(self) -> torch.Tensor:
        """Getter for baseline value

        Returns:
            torch.Tensor: scalar baseline value
        """
        return self.b

    def update(self, target: torch.Tensor) -> None:
        """Updates the baseline value b by the mean of the discounted reward of the current batch

        Args:
            target (torch.Tensor): mean of the discounted reward of a batch
        """
        self.b = torch.add(
            (1 - self.lambda_factor) * self.b, self.lambda_factor * target
        )
        return

    def reset(self, lambda_factor: float = None) -> None:
        """Resets the baseline value and the lambda factor.

        Args:
            lambda_factor (float, optional): sensitiy of the baseline to changes in reward. Defaults to None.
        """
        if lambda_factor != None:
            self.lambda_factor = lambda_factor
        self.b = torch.tensor(0.0, requires_grad=False, device=self.device)
        return
