from typing import Any, Dict
import torch
from src.utility import compare_paths

class FeedbackEngine:
    """Parent class for all feedback engines (automated, shell, mongodb)."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_flag = True

class AutomatedFeedback(FeedbackEngine):
    """Emulates human feedback."""
    def __init__(self,):
        """inits the automated feedback."""
        super().__init__()

    def __call__(self, pairs_final_nodes:torch.Tensor, pairs_goals:torch.Tensor) -> torch.Tensor:
        """ Basic (domain independent) automated feedback. Prefers paths by correctness.

        Args:
            final_nodes (torch.Tensor): the final node an agent is in.
            goals (torch.Tensor): the correct goal node of an query. 

        Returns:
            torch.Tensor: correctness based preferences
        """
   
        success = pairs_final_nodes == pairs_goals
      
        preferences = torch.empty((pairs_goals.size(0)), device=self.device, requires_grad=False, dtype=torch.float32).fill_(0.5)
        # if left true and right false -mu-> 1
        mu_one = (success[:, 0].to(torch.int8) + torch.logical_not(success[:, 1]).to(torch.int8)) == 2
        preferences[mu_one] = 1.
        # if left false and right true -mu-> 0
        mu_zero = (torch.logical_not(success[:, 0].to(torch.int8)) + success[:, 1].to(torch.int8)) == 2
        preferences[mu_zero] = 0.

        if torch.sum(mu_zero)+torch.sum(mu_one) < 10: #TODO: make this number to an config
            self.training_flag = False
        else:
            self.training_flag = True
        return preferences

class ROIFeedback(FeedbackEngine):
    """Feedback engine for localization score based feedback."""
    def __init__(
        self,
        id_to_class: Dict,
        id_to_relation: Dict,
        loop_relation_key: int,
        padding_relation_key: int,
        roi: Dict
    ):
        """
        Initialize the ROI-based feedback engine.

        Args:
            id_to_class (Dict[int, str]):
                Mapping from entity IDs to their class labels.
            id_to_relation (Dict[int, str]):
                Mapping from relation IDs to their names.
            loop_relation_key (int):
                Relation key used to represent self-loops in the graph.
            padding_relation_key (int):
                Relation key used for padding relation sequences.
            roi (Dict):
                Ground-truth mapping used to compute localization scores.  
                Contains the valid reasoning patterns.
        """
        super().__init__()
        self.id_to_class = id_to_class
        self.id_to_relation = id_to_relation
        self.loop_relation_key=loop_relation_key
        self.padding_relation_key=padding_relation_key
        self.roi=roi

    def __call__(
            self,
            pairs_actions: torch.Tensor,
            pairs_states: torch.Tensor,
            pairs_goals: torch.Tensor
    ) -> torch.Tensor:
        """
        Auguments localization-based feedback for candidate reasoning paths.

        Args:
            pairs_actions (torch.Tensor):
                Tensor containing the actions by pairs.
            pairs_states (torch.Tensor):
                Tensor containing the states by pairs. 
            pairs_goals (torch.Tensor):
                Tensor containing the goal by pairs.

        Returns:
            torch.Tensor:
                1-D Tensor with localization-based feedback for each pair.
        """

        # Init default preference to indifference
        preferences = torch.empty(
            (pairs_goals.size(0)), device=self.device, requires_grad=False, dtype=torch.float32
        ).fill_(0.5)

        success = pairs_states[:, :, -1, 0].squeeze() == pairs_goals

        # get roi
        pairs_path = torch.stack(
            tensors=(
                pairs_states[:, :, 0:1, 2],
                pairs_actions[:, :, 0:1, 0],
                pairs_states[:, :, 0:1, 0],
            ),
            dim=-1,
        ) # [batch_dim, pair_dim(2), step_dim, triple_dim] do this for the first step only, as we drop in pairs storage the initial state
        if pairs_states.size()[2] >= 2:
            further_steps = torch.stack(
                tensors=(
                    pairs_states[:, :, :-1, 0],
                    pairs_actions[:, :, 1:, 0],
                    pairs_states[:, :, 1:, 0],
                ),
                dim=-1,
            )  # [batch_dim, pair_dim(2), step_dim, triple_dim]

            pairs_path = torch.cat((pairs_path, further_steps), dim=2)

        pairs_path = torch.permute(
            pairs_path, (1, 2, 0, 3)
        )  # [pair_dim(2), step_dim, batch_dim, triple_dim]

        # next map entities to classes
        pairs_delexicalized_paths = []
        for pair in range(2):
            delexicalized_paths = []
            for batch_idx in range(pairs_path.shape[2]):

                path = pairs_path[pair, :, batch_idx, :]
                head_ids = path[:, 0].tolist()
                rel_ids = path[:, 1].tolist()
                tail_ids = path[:, 2].tolist()

                delex_path = [self.id_to_class[head_ids[0]]]
                for rel, tail in zip(rel_ids, tail_ids):
                    if rel != self.loop_relation_key and rel != 0:
                        delex_path.append(self.id_to_relation[rel])
                        delex_path.append(self.id_to_class[tail])

                delexicalized_paths.append(delex_path)

            pairs_delexicalized_paths.append(delexicalized_paths)

        # compute which of them overlaps with any region of interest.
        for batch_idx in range(pairs_path.shape[2]):
            left, right = (
                pairs_delexicalized_paths[0][batch_idx],
                pairs_delexicalized_paths[1][batch_idx],
            )
            predicted_rel = pairs_states[batch_idx, pair, 0, 2].item()

            relation_key = self.id_to_relation[predicted_rel]
            nested_values = self.roi.get(relation_key, dict())

            total_left_count = 0
            total_right_count = 0
            total_left_valid = False
            total_right_valid = False

            for ground_truth in nested_values.values():
                left_count, left_valid = compare_paths(ground_truth, left)
                right_count, right_valid = compare_paths(ground_truth, right)
                total_left_count += left_count
                total_right_count += right_count

                if left_valid:
                    total_left_valid = True
                if right_valid:
                    total_right_valid = True

            preference = 0.5
            if total_left_valid and not total_right_valid:
                preference = 0
            elif not total_left_valid and total_right_valid:
                preference = 1

            if preference == 0.5:
                if success[batch_idx,0].item() and not success[batch_idx,1].item(): 
                    preference = 0
                elif not success[batch_idx,0].item() and success[batch_idx,1].item():     
                    preference = 1

            if preference == 0.5:
                if total_left_count > total_right_count:
                    preference = 0
                elif total_right_count > total_left_count:
                    preference = 1
            preferences[batch_idx] = preference

        return preferences



class Feedback:
    """The feedback class"""
    def __init__(self, feedback_engine:str, kwargs:dict):
        self.feedback_engine = self.get_feedback_engine(feedback_engine, kwargs)

    def __call__(self, kwargs: Any) -> Any:
        return self.feedback_engine(**kwargs)

    def get_feedback_engine(self, feedback_engine:str, kwargs:dict) -> FeedbackEngine:
        if feedback_engine == "automated":
            return AutomatedFeedback()
        if feedback_engine == "roi":
            return ROIFeedback(
                id_to_class=kwargs["id_to_class"],
                id_to_relation=kwargs["id_to_relation"],
                loop_relation_key=kwargs["loop_relation_key"],
                padding_relation_key=kwargs["padding_relation_key"],
                roi=kwargs["roi"]
            )
        return
