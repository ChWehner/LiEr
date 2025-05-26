from typing import List

import torch
from collections import defaultdict
from sklearn.utils import gen_batches
from torch import Tensor

from src.utility import smart_sort, unique, MappingError
from src.policy_network import PolicyNetwork
from src.environment import Environment

from src.logger import Logger

logger = Logger()


class Evaluation:
    """A class to evaluate the model."""

    def __init__(self, num_hops: int, rollouts_test: int):
        """Inits the evaluation of the model.

        Args:
            num_hops (int): the number of hops (steps)
            rollouts_test (int): number of rollouts per test instance
            roi_dir (str): name of the directory containing the region of interest files
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = defaultdict(lambda: defaultdict(lambda: 0))

        self.num_hops = num_hops
        self.rollouts_test = rollouts_test

    def __call__(
        self,
        epoch: int,
        model: PolicyNetwork,
        environment: Environment,
        at_k: list[int] = [1, 3, 5, 10, 20],
        eval_batch_size: int = 8,
    ):
        """Calculates all metrices for the current epoch and returns them. The search strategy is beam search.
            Calls self.batch and self.calculate_metrices.

        Args:
            epoch (int): number of the current epoch.
            model (PolicyNetwork): the model to evaluate.
            environment (Environment) : the environment to evaluate the model in.
            at_k (list, int): allows to set which @k's to calculate. Defaults to [1,3,5,10,20].
            eval_batch_size (int): sets the size of a batch processed at once while evaluation. Defaults to 126.

        Returns:
            dict: All metrices calculated for the current epoch
        """

        batch_slices = gen_batches(
            n=environment.test_triples.size(0), batch_size=eval_batch_size
        )

        for batch_slice in batch_slices:
            logits_tracker = self.batch(
                environment=environment, model=model, batch_slice=batch_slice
            )
            self.calculate_metrices(
                environment, logits_tracker, epoch, batch_slice, at_k
            )

        self.metrics[f"{epoch}"] = dict(self.metrics[f"{epoch}"])
        logger().debug(f'Metrics epoch "{epoch}": {self.metrics[f"{epoch}"]}')

        return self.metrics[f"{epoch}"]

    def batch(self, environment: Environment, model: PolicyNetwork, batch_slice: slice):
        """Executes one runthrough of a batch on the test set. Calls self.hop.

        Args:
            environment (Environment) : the environment to evaluate the model in.
            model (PolicyNetwork): the model to evaluate.
            batch_slice (slice): The slice of the current batch.

        Returns:
            torch.Tensor: the logits for each instance of a batch over all hops.
        """

        # reset environment
        environment.reset(
            sources=["test"], num_rollouts=self.rollouts_test, batch_slice=batch_slice
        )
        # init hidden state
        hidden_state = None
        missing_relation_embeddings = None
        # init batch_size
        batch_size = batch_slice.stop - batch_slice.start
        # init logits
        logits_tracker = torch.empty(
            (batch_size, self.rollouts_test, 1), device=self.device, requires_grad=False
        )
        # run
        for hop in range(self.num_hops):
            logits_tracker, hidden_state, missing_relation_embeddings = self.hop(
                environment,
                model,
                logits_tracker,
                hidden_state,
                missing_relation_embeddings,
                hop,
                batch_size,
            )
        return logits_tracker

    def hop(
        self,
        environment: Environment,
        model: PolicyNetwork,
        logits_tracker: torch.Tensor,
        hidden_state: torch.Tensor,
        missing_relation_embeddings: torch.Tensor,
        hop: int,
        batch_size: int,
    ):
        """Executes one beam search hop.

        Args:
            environment (Environment) : the environment to evaluate the model in.
            model (PolicyNetwork): the model to evaluate.
            logits_tracker (torch.Tensor): the logits for each instance of a batch over all previous hops.
            hidden_state (torch.Tensor): the previous hidden states of the policy network.
            missing_relation_embeddings (torch.Tensor): the embeddings of the missing relations for each instance of the batch.
            hop (int): the current hop number.
            batch_size (int): the batch size.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor,): logits_tracker, hidden_state, missing_relation_embeddings
        """
        states = environment.states[-1]
        actions = environment.actions[-1]
        available_actions = environment.available_actions
        padding_mask = environment.padding_mask
        # let the agent make one step
        logits, _, _, hidden_state, missing_relation_embeddings = model(
            actions,
            states,
            available_actions,
            padding_mask,
            hidden_state,
            missing_relation_embeddings,
        )

        # get logits in rollout view and add previous logits to it, such that we have the total likelyhood of an rollout
        logits_rollout_view = logits.view(
            batch_size, self.rollouts_test, environment.max_degree
        )

        logits_accumulated = (logits_tracker + logits_rollout_view).reshape(
            batch_size, -1
        )
        top_k_idx = torch.topk(logits_accumulated, k=self.rollouts_test, dim=1).indices
        top_rollout_idx = top_k_idx.div(environment.max_degree, rounding_mode="floor")
        action_idx = top_k_idx.remainder(environment.max_degree).flatten()

        scaler = torch.arange(
            start=0, end=states.size(0), step=self.rollouts_test, device=self.device
        ).unsqueeze(dim=1)
        batch_idx = (top_rollout_idx + scaler).flatten()

        # update logits_tracker
        logits_tracker += logits[batch_idx, action_idx].reshape(
            batch_size, self.rollouts_test, 1
        )

        # gather the correct hidden_states
        hidden_state = [
            (h_s[0][0, batch_idx].unsqueeze(0), h_s[1][0, batch_idx].unsqueeze(0))
            for h_s in hidden_state
        ]

        # adapt environment
        environment.adapt_environment(batch_idx)
        # finally do the step
        ultimate_hop = (hop + 1) == self.num_hops
        penultimate_hop = (hop + 2) == self.num_hops
        environment.step(action_idx, penultimate_hop, ultimate_hop=ultimate_hop)

        return logits_tracker, hidden_state, missing_relation_embeddings

    def calculate_metrices(
        self,
        environment: Environment,
        logits_tracker: torch.Tensor,
        epoch: int,
        batch_slice: slice,
        at_k: list[int],
    ):
        """Calcualtes all metrices per batch by handling calling self.mrr and self.hit_at_k. 
        If localization is true, than also the roi@k and roi_mrr are calculated.
        Results are saved in self.metrices[epoch].
        Results in self.metrices[epoch] are only reliable after all batches are processed.

        Args:
            environment (Environment): the environment to evaluate the model in.
            logits_tracker (torch.Tensor): the logits for each instance of a batch over all previous hops.
            epoch (int): number of the current epoch.
            batch_slice (slice): The slice of the current batch.
            at_k (list, int): allows to set which @k's to calculate. Defaults to [1,3,5,10,20].
        """
        # reshape logits, final states and goals to rollout view
        batch_size = batch_slice.stop - batch_slice.start
        goals = torch.reshape(environment.goals, (batch_size, self.rollouts_test))
        final_states = torch.reshape(
            environment.states[-1, :, 0], (batch_size, self.rollouts_test)
        )
        logits = logits_tracker.squeeze(dim=2)

        # get ranking within rollouts via logits -> if low logit a rollout has a high rank
        sorted_logits, ranked_idx = torch.sort(
            logits, stable=True, descending=False, dim=1
        )
        # sort final states according to ranking
        ranked_final_states = smart_sort(final_states, ranked_idx)

        # get indicies of the first appearing final states
        mask = torch.zeros(logits.size(), dtype=torch.bool)
        _, _, _, unique_idx = unique(ranked_final_states, dim = 1)
        mask[:, unique_idx] = True
        logits = sorted_logits
        final_states = ranked_final_states

        # set the the logits of the final states that do not appear first to -99999
        logits_masked = torch.where(mask, logits, -99999.0)

        # make a mask that is true if goal was reached in a rollout, otherwise false
        success_mask = goals == final_states

        # make ranking -> if low logit a rollout has a high rank
        ranking = torch.argsort(logits_masked, descending=False, dim=1)

        # apply ranking
        ranked_success_mask = smart_sort(success_mask, ranking)
        # calculate Hit@k
        self.hit_at_k(environment, ranked_success_mask, epoch, at_k=at_k)
        # calculate MRR
        self.mrr(environment, ranked_success_mask, epoch)
        # calculate Localization score
        if environment.localization:
            try:
                paths = environment.collect_paths()
                delex_paths = environment.delexicalize_paths(paths, num_rollouts=self.rollouts_test)
                missing_relations = torch.reshape(
                    environment.states[0, :, 2], (-1, self.rollouts_test)
                )
                self.localization_score_at_k(
                    environment,
                    delexicalized_paths=delex_paths,
                    predictions=missing_relations,
                    epoch=epoch,
                    ranking=ranking,
                    at_k=at_k,
                )
                self.roi_mrr(
                    environment,
                    delexicalized_paths=delex_paths,
                    predictions=missing_relations,
                    epoch=epoch,
                    ranking=ranking,
                )
            except MappingError as e:
                print(f"Error: {e}")

        return

    def hit_at_k(
        self,
        environment: Environment,
        ranked_success_mask: torch.Tensor,
        epoch: int,
        at_k: list[int] = [1, 3, 5, 10, 20],
    ):
        """Calculates the hit@k metrices and safes them in self.metrices. Results in self.metrices[epoch] are only reliable after all batches are processed.
        For more information on hit@k please read https://pykeen.readthedocs.io/en/stable/api/pykeen.metrics.ranking.HitsAtK.html#pykeen.metrics.ranking.HitsAtK.

        Args:
            environment (Environment): the environment to evaluate the model in.
            ranked_success_mask (torch.Tensor): holds the ranking for each rollout of the batch.
                                                Is true if rollout hits the goal state otherwise false.
            epoch (int): number of epoch we are in.
            at_k (list, int): allows to set which @k's to calculate. Defaults to [1,3,5,10,20].
        """
        weight = environment.test_triples.size(0)
        for k in at_k:
            hit_at_k = torch.sum(torch.any(ranked_success_mask[:, -k:], dim=1))
            self.metrics[f"{epoch}"][f"hit@{k}"] += (hit_at_k / weight).item()

        return

    def mrr(self, environment, ranked_success_mask, epoch):
        """Calculates the MRR metric and safe it in self.metrices. Results in self.metrices[epoch] are only reliable after all batches are processed.
        For more information on MRR please read https://www.wikiwand.com/en/Mean_reciprocal_rank

        Args:
            environment (Environment): the environment to evaluate the model in.
            ranked_success_mask (torch.Tensor): holds the ranking for each rollout of the batch.
                                                Is true if rollout hits the goal state otherwise false.
            epoch (int): number of epoch we are in.
        """

        def inverse_rank(row):
            for rank, value in enumerate(torch.flip(row, dims=(0,))):
                if value.item():
                    return torch.tensor([1 / (rank + 1)])
            return torch.tensor([0])

        weight = environment.test_triples.size(0)
        inverse_ranks = torch.stack(
            [inverse_rank(row) for row in torch.unbind(ranked_success_mask, dim=0)],
            dim=0,
        )
        self.metrics[f"{epoch}"]["mrr"] += (torch.sum(inverse_ranks) / weight).item()
        return

    def roi_mrr(
            self,
            environment: Environment,
            delexicalized_paths: List,
            predictions: Tensor,
            ranking: Tensor,
            epoch: int
        ) -> None:
        """
        Calculates the mean reciprocal rank based on the ROI ground truth.

        Args:
            environment (environment): Current environment containing kg, mappings, etc.
            delexicalized_paths (list): List containing the delexicalized paths.
            predictions (torch.Tensor): List of predicted relation types.
            ranking (torch.Tensor): Tensor containing rankings for paths.
            epoch (int): The current epoch.
        """
        if not environment.id_to_class:
            raise MappingError(
                "The environment must have an id_to_class mapping to calculate the ROI MRR."
            )

        rollouts_reciprocal_rank = []
        for idx_rollouts in range(predictions.shape[0]):
            rankings = ranking[idx_rollouts]
            path_rollouts = delexicalized_paths[idx_rollouts]
            ranked_paths = [path_rollouts[rank] for rank in rankings]
            missing_link = predictions[idx_rollouts][0].item()
            missing_link_key = environment.id_to_relation[missing_link]

            if missing_link_key in environment.roi:
                nested_values = environment.roi[missing_link_key].values()
            else:
                break

            reciprocal_rank = 0
            for idx, path in enumerate(ranked_paths):
                if any(path == val for val in nested_values):
                    reciprocal_rank = 1 / (idx+1)
                    break

            rollouts_reciprocal_rank.append(reciprocal_rank)

        weight = environment.test_triples.size(0)
        for rollout_reciprocal_rank in rollouts_reciprocal_rank:
            self.metrics[f"{epoch}"]["roi_mrr"] += rollout_reciprocal_rank / weight

    def localization_score_at_k(
        self,
        environment: Environment,
        delexicalized_paths: List,
        predictions: Tensor,
        ranking: Tensor,
        epoch: int = 0,
        at_k: List[int] = [1, 3, 5, 10, 20],
    ) -> None:
        """
        Computes the localization score at k given a set of paths and predictions.

        Args:
            environment (environment): Current environment containing kg, mappings, etc.
            delexicalized_paths (list): List containing the delexicalized paths.
            predictions (torch.Tensor): Tensor of predicted relation types.
            ranking (torch.Tensor): Tensor containing rankings for paths.
            epoch (int): The current epoch.
            at_k (list[int]): Which @k's to calculate. Defaults to [1,3,5,10,20].

        """
        if not environment.id_to_class:
            raise MappingError(
                    "The environment must have an id_to_class mapping to calculate the localization score."
                )

        hits_at_k = {k: 0 for k in at_k}

        for idx_rollouts in range(predictions.shape[0]):
            rankings = ranking[idx_rollouts]
            path_rollouts = delexicalized_paths[idx_rollouts]
            ranked_paths = [path_rollouts[rank] for rank in rankings]
            missing_link = predictions[idx_rollouts][0].item()
            missing_link_key = environment.id_to_relation[missing_link]

            if missing_link_key in environment.roi:
                nested_values = environment.roi[missing_link_key].values()
            else:
                break

            for k in at_k:
                top_k_paths = ranked_paths[:k]
                found_hit = False
                for path in top_k_paths:
                    if any(path == val for val in nested_values):
                        found_hit = True
                        break

                if found_hit:
                    hits_at_k[k] += 1

        weight = environment.test_triples.size(0)

        for k in at_k:
            hit_ratio = hits_at_k[k] / weight
            self.metrics[f"{epoch}"][f"loc@{k}"] += hit_ratio

