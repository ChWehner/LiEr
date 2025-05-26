import torch
from math import floor

from src.utility import unique
from src.interactive_reward_modules.feedback import Feedback

import time

from src.logger import Logger

logger = Logger()


class PairsStorage:
    """A storage for the top n pairs."""

    def __init__(
        self,
        top_k: int = 20,
        use_entities: bool = True,
        feedback_engine: str = "automated",
        error_rate: float = -1,
        kwargs={},
    ):
        """inits the pairs storage

        Args:
            top_k (int, optional): the amount of pairs in the storage. Defaults to 20.
            use_entities (bool, optional): if entities shall be considered while calculating the reward. Defaults to True.
            feedback_engine (bool, optional): possible types of feedback engines are "automated", "shell", "mongodb".
            error_rate (float, optional): artificial error rate of labels (only used in combination with automated feedback)
        """
        self.top_k = top_k
        self.use_entities = use_entities
        self.error_rate = error_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feedback_engine = feedback_engine
        self.feedback = Feedback(feedback_engine, kwargs)
        self.training_flag = True
        self.iteration = 0

        self.buffer_pairs_states = torch.empty(0, device=self.device, dtype=torch.int32)
        self.buffer_pairs_actions = torch.empty(
            0, device=self.device, dtype=torch.int32
        )
        self.buffer_pairs_goals = torch.empty(0, device=self.device, dtype=torch.int32)
        self.buffer_pairs_uncertainty = torch.empty(
            0, device=self.device, dtype=torch.int32
        )

        self.pairs_states = torch.empty(0, device=self.device, dtype=torch.int32)
        self.pairs_actions = torch.empty(0, device=self.device, dtype=torch.int32)
        self.pairs_preferences = torch.empty(0, device=self.device, dtype=torch.float32)
        self.pairs_goals = torch.empty(0, device=self.device, dtype=torch.float32)

        self.train_split = torch.empty(0, device=self.device, dtype=torch.int32)
        self.train_split_temp = torch.empty(0, device=self.device, dtype=torch.int32)
        self.valid_split = torch.empty(0, device=self.device, dtype=torch.int32)
        self.valid_split_temp = torch.empty(0, device=self.device, dtype=torch.int32)

    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> None:
        """takes one batch, creates pairs, and inserts them into the paths storage.

        Args:
            states (torch.Tensor): The states of all instances of a batch.
            actions (torch.Tensor): The actions of all instances of a batch.
            goals (torch.Tensor): The goals of all instances of a batch - only used if feedback is automated.
            uncertainty (torch.Tensor): The uncertainty of the reward ensemble regarding all instances of a batch.
        """
        # make batch dim first and step dim second
        states = states.swapaxes(0, 1)
        actions = actions.swapaxes(0, 1)

        states_for_unique = states.detach().clone()
        actions_for_unique = actions.detach().clone()
        goals_for_unique = goals.unsqueeze(dim=1).repeat(1, actions.size(1), 1)
        if not self.use_entities:
            # occlude all entities
            states_for_unique[:, :, 0:2] = -1
            actions_for_unique[:, :, 1] = -1

        paths = torch.cat(
            (states_for_unique, actions_for_unique, goals_for_unique), dim=2
        )
        _, _, _, unique_idx = unique(paths, dim=0)

        states_unique = states[unique_idx]
        actions_unique = actions[unique_idx]
        goals_unique = goals[unique_idx]
        uncertainty_unique = uncertainty[unique_idx]

        # only make pairs with identical missing relation
        missing_relations = states_unique[:, 0, 2].unique()
        states_temp = []
        actions_temp = []
        goals_temp = []
        uncertainty_temp = []
        pairs_idx_temp = []
        baseline = 0
        # TODO: how to make this faster?

        for missing_relation in missing_relations:
            missing_relation_mask = states_unique[:, 0, 2] == missing_relation
            states_temp.append(states_unique[missing_relation_mask])
            actions_temp.append(actions_unique[missing_relation_mask])
            goals_temp.append(goals_unique[missing_relation_mask])
            uncertainty_temp.append(uncertainty_unique[missing_relation_mask])
            end = missing_relation_mask.sum() + baseline
            idx = torch.arange(baseline, end, device=self.device)
            baseline = end
            pairs_idx_temp.append(torch.combinations(idx, r=2))

        # apfel mit birnen
        """
        num_samples = min(states_unique.size(0), 10*self.top_k)
        
        samples_idx = torch.randperm(states_unique.size(0))[:num_samples]
        states_temp.append(states_unique[samples_idx])
        actions_temp.append(actions_unique[samples_idx])
        goals_temp.append(goals_unique[samples_idx])
        uncertainty_temp.append(uncertainty_unique[samples_idx])
        idx = torch.arange(end=num_samples, device=self.device)
        pairs_idx_temp.append(torch.combinations(idx, r=2))
        """
        # TODO: filte out all pairs that are already in the buffer or permanent storage
        # stack everything back together
        states = torch.cat(states_temp, dim=0)
        actions = torch.cat(actions_temp, dim=0)
        goals = torch.cat(goals_temp, dim=0)
        uncertainty = torch.cat(uncertainty_temp, dim=0)
        pairs_idx = torch.cat(pairs_idx_temp, dim=0)

        pairs_idx = self.filter_duplicate_pairs(actions, states, goals, pairs_idx)
        # identify top k uncertain pairs
        pairs_uncertainty = uncertainty[pairs_idx[:, 0]] + uncertainty[pairs_idx[:, 1]]
        top_k = min(self.top_k, pairs_uncertainty.size(0))
        _, top_k_idx = torch.topk(pairs_uncertainty, top_k)

        pairs_idx = pairs_idx[top_k_idx]
        left_idx = pairs_idx[:, 0]
        right_idx = pairs_idx[:, 1]

        # collect top k uncertain pairs of this batch

        pairs_states = torch.stack(
            (states[left_idx], states[right_idx]), dim=1
        )  ## new shape: [batch, pair, step, state]
        pairs_actions = torch.stack(
            (actions[left_idx], actions[right_idx]), dim=1
        )  # # new shape: [batch, pair, step, action]
        pairs_goals = torch.stack((goals[left_idx], goals[right_idx]), dim=1)
        pairs_uncertainty = pairs_uncertainty[top_k_idx]

        # combine them with all pairs
        pairs_states = torch.cat((self.buffer_pairs_states, pairs_states), dim=0)
        pairs_actions = torch.cat((self.buffer_pairs_actions, pairs_actions), dim=0)
        pairs_goals = torch.cat((self.buffer_pairs_goals, pairs_goals), dim=0)
        pairs_uncertainty = torch.cat(
            (self.buffer_pairs_uncertainty, pairs_uncertainty), dim=0
        )

        # figure out buffer top k pairs
        buffer_top_k = min(self.top_k, pairs_uncertainty.size(0))
        _, buffer_top_k_idx = torch.topk(pairs_uncertainty, buffer_top_k)

        # store buffer top k uncertain pairs
        self.buffer_pairs_states = pairs_states[buffer_top_k_idx]
        self.buffer_pairs_actions = pairs_actions[buffer_top_k_idx]
        self.buffer_pairs_goals = pairs_goals[buffer_top_k_idx]
        self.buffer_pairs_uncertainty = pairs_uncertainty[buffer_top_k_idx]
        return

    def filter_duplicate_pairs(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
        goals: torch.Tensor,
        pairs_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.buffer_pairs_goals.size(0) == 0 and self.pairs_goals.size(0) == 0:
            return pairs_idx

        states = states.detach().clone()
        actions = actions.detach().clone()
        goals_for_unique = goals.unsqueeze(dim=1).repeat(1, actions.size(1), 1)
        if not self.use_entities:
            # occlude all entities
            states[:, :, 0:2] = -1
            actions[:, :, 1] = -1
        paths = torch.cat((states, actions, goals_for_unique), dim=2)

        # get paths from buffer and storage
        if self.buffer_pairs_goals.size(0) > 0:
            buffer_pairs_states = self.buffer_pairs_states.detach().clone()
            buffer_pairs_actions = self.buffer_pairs_actions.detach().clone()
            buffer_pairs_goals = self.buffer_pairs_goals.unsqueeze(dim=2).repeat(
                1, 1, actions.size(1), 1
            )
            if not self.use_entities:
                # occlude all entities
                buffer_pairs_states[:, :, :, 0:2] = -1
                buffer_pairs_actions[:, :, :, 1] = -1
            buffer_pairs = torch.cat(
                (buffer_pairs_states, buffer_pairs_actions, buffer_pairs_goals), dim=3
            )

        if self.pairs_goals.size(0) > 0:
            storage_pairs_states = self.pairs_states.detach().clone()
            storage_pairs_actions = self.pairs_actions.detach().clone()
            storage_pairs_goals = self.pairs_goals.unsqueeze(dim=2).repeat(
                1, 1, actions.size(1), 1
            )
            if not self.use_entities:
                # occlude all entities
                storage_pairs_states[:, :, :, 0:2] = -1
                storage_pairs_actions[:, :, :, 1] = -1
            storage_pairs = torch.cat(
                (storage_pairs_states, storage_pairs_actions, storage_pairs_goals),
                dim=3,
            )

        # stack pairs
        pairs = torch.stack((paths[pairs_idx[:, 0]], paths[pairs_idx[:, 1]]), dim=1)

        if self.buffer_pairs_goals.size(0) > 0 and self.pairs_goals.size(0) > 0:
            pairs = torch.cat((pairs, buffer_pairs, storage_pairs), dim=0)
        elif self.buffer_pairs_goals.size(0) == 0 and self.pairs_goals.size(0) > 0:
            pairs = torch.cat((pairs, storage_pairs), dim=0)
        elif self.buffer_pairs_goals.size(0) > 0 and self.pairs_goals.size(0) == 0:
            pairs = torch.cat((pairs, buffer_pairs), dim=0)
        _, _, counts, unique_idx = unique(pairs, dim=0)

        left_intersecting = unique_idx[counts > 1]

        # get right intersecting idx
        pairs = torch.stack((paths[pairs_idx[:, 1]], paths[pairs_idx[:, 0]]), dim=1)

        if self.buffer_pairs_goals.size(0) > 0 and self.pairs_goals.size(0) > 0:
            pairs = torch.cat((pairs, buffer_pairs, storage_pairs), dim=0)
        elif self.buffer_pairs_goals.size(0) == 0 and self.pairs_goals.size(0) > 0:
            pairs = torch.cat((pairs, storage_pairs), dim=0)
        elif self.buffer_pairs_goals.size(0) > 0 and self.pairs_goals.size(0) == 0:
            pairs = torch.cat((pairs, buffer_pairs), dim=0)
        _, _, counts, unique_idx = unique(pairs, dim=0)

        right_intersecting = unique_idx[counts > 1]

        # concat intersecting idx
        intersecting = torch.cat((left_intersecting, right_intersecting), dim=0)
        # filter out intersecting idx >= pairs_idx.size(0)
        intersecting = intersecting[intersecting < pairs_idx.size(0)]
        # create mask with false if indersecting idx
        mask = torch.ones(pairs_idx.size(0), dtype=torch.bool, device=self.device)
        mask[intersecting] = False
        # apply mask to pairs idx
        pairs_idx = pairs_idx[mask]
        return pairs_idx

    def collect_and_store_feedback(self) -> None:
        """Collects feedback and stores buffered pairs in permanent storage. The test validation split is created while storing.
        Calls self.feedback.
        """
        # collect feedback
        if self.feedback_engine == "automated":
            kwargs = {
                "pairs_final_nodes": self.buffer_pairs_states[:, :, -1, 0].squeeze(),
                "pairs_goals": self.buffer_pairs_goals.squeeze(),
            }
            preferences = self.feedback(kwargs)
        elif self.feedback_engine == "roi":
            kwargs = {
                "pairs_goals": self.buffer_pairs_goals.squeeze(),
                "pairs_states": self.buffer_pairs_states,
                "pairs_actions": self.buffer_pairs_actions
            }
            preferences = self.feedback(kwargs)
        elif self.feedback_engine == "mongodb":
            kwargs = {
                "pairs_actions": self.buffer_pairs_actions,
                "pairs_states": self.buffer_pairs_states,
                "uncertainty": self.buffer_pairs_uncertainty,
                "goals": self.buffer_pairs_goals,
                "iteration": self.iteration,
            }
            pairs_actions, pairs_states, goals, preferences = self.feedback(kwargs)
            self.buffer_pairs_actions = pairs_actions
            self.buffer_pairs_states = pairs_states
            self.buffer_pairs_goals = goals
            self.iteration += 1

        # set training flag
        self.training_flag = self.feedback.feedback_engine.training_flag

        # apply artificial error
        if self.error_rate > 0:
            num_errors = int(preferences.size(0) * self.error_rate)
            errors_idx = torch.randperm(preferences.size(0))[:num_errors]
            errors = torch.rand(
                size=(num_errors,), device=self.device, requires_grad=False
            )
            preferences[errors_idx] = errors
        # num_preferences = preferences.size(0)
        # TODO: do that differently: store all BUT as indiffert/different than for training select all different pairs and 10% indifferent
        # split into
        # indifferent_mask = (preferences == 0.5)
        # different_mask = (preferences != 0.5)

        # num_indifferent = min(max(floor(0.1* torch.sum(different_mask)), 5), preferences.size(0))
        # indifferent_idx = torch.arange(num_preferences, device=self.device)[(preferences == 0.5)][:num_indifferent]
        # ratio = torch.sum(indifferent_mask)/num_preferences

        # collect
        #preferences =preferences[different_mask]
        #pairs_states = self.buffer_pairs_states[different_mask]
        #pairs_actions = self.buffer_pairs_actions[different_mask]
        #pairs_goals = self.buffer_pairs_goals[different_mask]

        # num_preferences = preferences.size(0)
        # indifferent_mask = (preferences == 0.5)

        # ratio = torch.sum(indifferent_mask)/num_preferences

        # figure out the global idx of the pairs
        end = self.pairs_states.size(0) + self.buffer_pairs_states.size(0)
        pairs_idx = torch.arange(self.pairs_states.size(0), end, device=self.device)

        # make test-validation split
        size_validation = floor(self.buffer_pairs_states.size(0) * 0.3)
        size_validation = max(size_validation, 2)
        randomized_pairs_buffer_idx = pairs_idx[
            torch.randperm(self.buffer_pairs_states.size(0))
        ]

        self.train_split = torch.cat(
            (self.train_split, randomized_pairs_buffer_idx[size_validation:]), dim=0
        )
        self.valid_split = torch.cat(
            (self.valid_split, randomized_pairs_buffer_idx[:size_validation]), dim=0
        )
        # move to permanent storage
        self.pairs_states = torch.cat(
            (self.pairs_states, self.buffer_pairs_states), dim=0
        )
        self.pairs_actions = torch.cat(
            (self.pairs_actions, self.buffer_pairs_actions), dim=0
        )
        self.pairs_preferences = torch.cat((self.pairs_preferences, preferences), dim=0)
        self.pairs_goals = torch.cat((self.pairs_goals, self.buffer_pairs_goals), dim=0)
        # reset buffer
        self.buffer_pairs_states = torch.empty(0, device=self.device, dtype=torch.int32)
        self.buffer_pairs_actions = torch.empty(
            0, device=self.device, dtype=torch.int32
        )
        self.buffer_pairs_goals = torch.empty(0, device=self.device, dtype=torch.int32)
        self.buffer_pairs_uncertainty = torch.empty(
            0, device=self.device, dtype=torch.int32
        )
        return

    def shuffel_training_instances(self) -> None:
        """shuffels the indicies of the training instances"""
        # get all indicies of all different and indifferent instances from train_split
        different_mask = self.pairs_preferences[self.train_split] != 0.5
        indifferent_mask = torch.logical_not(different_mask)
        # randomly select indices of |num_different| different instances
        num_different = torch.sum(different_mask)
        if num_different != 0:
            different_idx = self.train_split[different_mask][
                torch.randint(high=num_different, size=(num_different,))
            ]
        else:
            different_idx = torch.empty(0, device=self.device, dtype=torch.int32)
        # than select indices of |num_different|*0.3 indifferent instances
        num_init_indifferent = torch.sum(indifferent_mask)
        # num_indifferent = min(
        #    max(floor(num_different * 0.3), 5), num_init_indifferent
        # )
        num_indifferent = 0
        if num_indifferent != 0:
            indifferent_idx = self.train_split[indifferent_mask][
                torch.randint(high=num_init_indifferent, size=(num_indifferent,))
            ]
        else:
            indifferent_idx = torch.empty(0, device=self.device, dtype=torch.int32)
        # concat indicies
        #logger().debug(
        #    f"Train: differnt {different_idx.size(0)}, indifferent {indifferent_idx.size(0)}"
        #)
        idx = torch.cat((different_idx, indifferent_idx), dim=0)
        # shuffel indicies
        if idx.size(0) > 0:
            idx = idx[torch.randperm(idx.size(0))]
        # assign to train_split_temp
        self.train_split_temp = idx

    def balance_validation_instances(self) -> None:
        # get all indicies of all different and indifferent instances from valid_split
        different_mask = self.pairs_preferences[self.valid_split] != 0.5
        indifferent_mask = torch.logical_not(different_mask)
        # randomly select indices of |num_different| different instances
        num_different = torch.sum(different_mask)
        if num_different != 0:
            different_idx = self.valid_split[
                different_mask
            ]  # [torch.randint(low = 0, high=num_different, size=(num_different,))]
        else:
            different_idx = torch.empty(0, device=self.device, dtype=torch.int32)
        # than select indices of |num_different|*0.3 indifferent instances
        num_init_indifferent = torch.sum(indifferent_mask)
        num_indifferent = min(
            2, num_init_indifferent
        )
        num_indifferent = 0
        if num_different == 0:
            indifferent_idx = self.valid_split[indifferent_mask][
                torch.randperm(num_init_indifferent)[:num_indifferent]
            ]
        else:
            indifferent_idx = torch.empty(0, device=self.device, dtype=torch.int32)
        # concat indicies
        #logger().debug(
        #    f"Validation: differnt {different_idx.size(0)}, indifferent {indifferent_idx.size(0)}"
        #)
        self.valid_split_temp = torch.cat((different_idx, indifferent_idx), dim=0)

    def get_pairs_states(self, batch_slice: slice, source: str) -> torch.Tensor:
        """gets paris of states from the storage

        Args:
            batch_slice (slice): the slice of the batch
            source (str): "test" or "valid"

        Returns:
            torch.Tensor: a batch of states
        """
        if source == "train":
            idx = self.train_split_temp[batch_slice]
        elif source == "valid":
            idx = self.valid_split_temp[batch_slice]

        return self.pairs_states[idx]

    def get_pairs_actions(self, batch_slice: slice, source: str) -> torch.Tensor:
        """gets paris of actions from the storage

        Args:
            batch_slice (slice): the slice of the batch
            source (str): test/valid

        Returns:
            torch.Tensor: a batch of actions
        """

        if source == "train":
            idx = self.train_split_temp[batch_slice]
        elif source == "valid":
            idx = self.valid_split_temp[batch_slice]

        return self.pairs_actions[idx]

    def get_pairs_goals(self, batch_slice: slice, source: str) -> torch.Tensor:
        """gets paris of goals from the storage

        Args:
            batch_slice (slice): the slice of the batch
            source (str): test/valid

        Returns:
            torch.Tensor: a batch of goals
        """

        if source == "train":
            idx = self.train_split_temp[batch_slice]
        elif source == "valid":
            idx = self.valid_split_temp[batch_slice]

        return self.pairs_goals[idx]

    def get_preferences(self, batch_slice: slice, source: str) -> torch.Tensor:
        """gets preferences from the storage

        Args:
            batch_slice (slice): the slice of the batch
            source (str): test/valid

        Returns:
            torch.Tensor: a batch of preferences
        """
        if source == "train":
            idx = self.train_split_temp[batch_slice]
        elif source == "valid":
            idx = self.valid_split_temp[batch_slice]

        return self.pairs_preferences[idx]

    def reset_storage(self) -> None:
        """resets the storage"""
        self.pairs_states = torch.empty(0, device=self.device, dtype=torch.int32)
        self.pairs_actions = torch.empty(0, device=self.device, dtype=torch.int32)
        self.pairs_preferences = torch.empty(0, device=self.device, dtype=torch.int32)
        self.pairs_goals = torch.empty(0, device=self.device, dtype=torch.int32)

        self.train_split = torch.empty(0, device=self.device, dtype=torch.int32)
        self.valid_split = torch.empty(0, device=self.device, dtype=torch.int32)

        return


if __name__ == "__main__":
    pairs_storage = PairsStorage(
        top_k=5, use_entities=False, feedback_engine="automated", kwargs={}
    )

    states = torch.tensor(
        [
            [
                [76, 108, 18],
                [76, 108, 18],
                [58, 108, 18],
                [16, 16, 44],
                [66, 16, 44],
                [49, 74, 28],
                [49, 74, 28],
                [24, 74, 28],
            ],
            [
                [45, 108, 18],
                [45, 108, 18],
                [76, 108, 18],
                [37, 16, 44],
                [78, 16, 44],
                [26, 74, 28],
                [26, 74, 28],
                [116, 74, 28],
            ],
        ],
        device=pairs_storage.device,
        dtype=torch.int32,
    )

    actions = torch.tensor(
        [
            [
                [80, 76],
                [80, 76],
                [18, 58],
                [92, 16],
                [3, 66],
                [68, 49],
                [68, 49],
                [1, 24],
            ],
            [
                [34, 18],
                [34, 18],
                [45, 76],
                [92, 16],
                [74, 78],
                [85, 26],
                [85, 26],
                [85, 116],
            ],
        ],
        device=pairs_storage.device,
        dtype=torch.int32,
    )

    goals = torch.tensor(
        [[45], [45], [45], [37], [37], [107], [107], [107]],
        device=pairs_storage.device,
        dtype=torch.int32,
    )

    uncertainty = torch.tensor(
        [
            8.6555e-05,
            8.6555e-05,
            1.0375e-04,
            8.5766e-05,
            9.8021e-05,
            1.0182e-04,
            1.0182e-04,
            7.6865e-05,
        ],
        device=pairs_storage.device,
    )

    pairs_storage(states, actions, goals, uncertainty)
    #### SECOND ROUND
    states = torch.tensor(
        [
            [
                [46, 44, 22],
                [38, 44, 22],
                [61, 56, 11],
                [98, 56, 11],
                [130, 109, 28],
                [61, 109, 28],
                [76, 108, 18],
                [58, 108, 18],
            ],
            [
                [95, 44, 22],
                [68, 44, 22],
                [129, 56, 11],
                [97, 56, 11],
                [3, 109, 28],
                [56, 109, 28],
                [18, 108, 18],
                [76, 108, 18],
            ],
        ],
        device=pairs_storage.device,
        dtype=torch.int32,
    )

    actions = torch.tensor(
        [
            [
                [62, 46],
                [68, 38],
                [34, 61],
                [18, 98],
                [22, 130],
                [34, 61],
                [80, 76],
                [18, 58],
            ],
            [
                [17, 95],
                [85, 68],
                [62, 129],
                [26, 97],
                [68, 3],
                [3, 56],
                [34, 18],
                [45, 76],
            ],
        ],
        device=pairs_storage.device,
        dtype=torch.int32,
    )

    goals = torch.tensor(
        [
            [109],
            [109],
            [49],
            [49],
            [27],
            [27],
            [45],
            [45],
        ],
        device=pairs_storage.device,
        dtype=torch.int32,
    )

    uncertainty = torch.tensor(
        [
            1.4277e-06,
            1.7479e-06,
            3.1353e-06,
            5.5038e-06,
            1.4837e-06,
            1.0146e-06,
            8.6555e-05,
            1.0375e-04,
        ],
        device=pairs_storage.device,
    )

    pairs_storage(states, actions, goals, uncertainty)

    pairs_storage.collect_and_store_feedback()

    pairs_storage.shuffel_training_instances()
