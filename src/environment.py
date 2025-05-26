from typing import Dict, List

import torch
import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict
from sklearn.utils import gen_batches

from src.reward_estimator import BasicReward 
from src.interactive_reward_modules.preference_based_reward import PreferenceBasedReward
from src.logger import Logger

logger = Logger()

import random
import json

class Environment:
    """the environment of the agent. Holds states and actions."""

    def __init__(
        self,
        knowledge_graph_name: str,
        inverse_relations: bool = True,
        relation_types: list[str] = None,
        gamma: float = 0.99,
        lambda_factor: float = 0.1,
        interactive: bool = True,
        use_entities: bool = False,
        localization: bool = True,
        roi_dir: str = "localization",
        top_k: int = 20,
        feedback_engine: str = "automated",
        error_rate: float = -1.,
        prune: bool = True,
        max_degree: int = 200,
        embedding_size: int = 128,
        mlp_size: int = 256,
        dropout:float = 0.2,
        learning_rate:float = 0.001,
        goal_loss_impact:float = 0.1,
        batch_size:int = 64
    ):
        """creates the environment by loading the knowledge graph from the files.

        Args:
            knowledge_graph_name (str): file name of the knowledge graph.
            blind_spots (list[str], optional): relations that cannot be traversed by the agent. Defaults to None.
            inverse_relations(bool): Inserts inverse relations to the knowledge graph if true . Defaults to True.
            relation_types (list[str], optional): A list of strings with the names of the relation types include in the batch. Defaults to None.
            gamma (float, optional): The discount rate of the reward
            lambda_factor (float, optional): the sensitivity of the baseline to updates.
            interactive (bool, optional): Use interactive reward. Defaults to True.
            use_entities (bool, optional): if entities shall be used as inputs for the forward pass. Only relevant if interactive is True. Defaults to False.
            top_k (int, optional): The number of pairs shown to a user per traing of the reward estimator. Only relevant if interactive is True. Defaults to 20.
            automate_feedback (bool, optional): Automates the feedback process if set to true. Only relevant if interactive is True. Defaults to False.
            error_rate (float, optional): induces an random error into the human feedback. Only used for robustness experiments. Only relevant if interactive is True. Defaults to -1.
            max_degree (int, optional): Maximum actions per node. Defaults to 200.
            localization (bool, optional): Whether to calculate the localization score when evaluating.
            roi_dir (str, optional): The directory containing the ground truth for localization score calculation.
            prune (bool, optional): If set, prunes larger graphs to have a maximum degree of max_degree by randomly selecting actions per node. Defaults to True.
            max_degree (int, optional): Maximum number of outgoing edges (actions) per node after pruning. Defaults to 200.
            embedding_size (int, optional): Dimensionality of the learned node embeddings. Defaults to 128.
            mlp_size (int, optional): Number of hidden units in the policy/value MLP network. Defaults to 256.
            dropout (float, optional): Dropout rate applied within the MLP network. Defaults to 0.2.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            goal_loss_impact (float, optional): Weight of the goal-oriented loss component relative to the main loss. Defaults to 0.1.
            batch_size (int, optional): Number of samples per training batch. Defaults to 64.
        """

        self.relation_types = relation_types
        self.knowledge_graph_name = knowledge_graph_name
        self.path_to_knowledge_graph = (
            Path(__file__).parent / "knowledge_graphs" / f"{knowledge_graph_name}"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inverse_relations = inverse_relations

        self.id_to_relation = self.load_id_to_item("relations")
        self.id_to_entity = self.load_id_to_item("entities")
        self.localization = localization
        if self.localization:
            self.id_to_class = self.load_id_to_item("class")
            self.roi_dir = self.path_to_knowledge_graph/roi_dir
            self.roi = self.prepare_roi()

        self.relation_to_id = {
            v: k for k, v in self.id_to_relation.items()
        }  
        self.entity_to_id = {
            v: k for k, v in self.id_to_entity.items()
        }

        self.one_to_n_actions_dict = defaultdict(
            set
        )  # is populated in load_knowledge_graph and load_triples

        self.training_triples = self.load_triples("train")
        self.validation_triples = self.load_triples("valid")  # to identify overfitting
        self.test_triples = self.load_triples("test")  # to assess model performance

        self.loop_relation_key = self.set_loop_relation_key()
        self.padding_relation_key = self.set_padding_relation_key()
        self.num_relations = self.set_num_relations()
        self.padding_node_key = self.set_padding_node_key()
        self.num_nodes = self.set_num_nodes()
        self.max_degree = max_degree  # is set in load_knowledge_graph
        self.prune = prune # prunes kg to max_degree

        self.padding = None  # is set in load_knowledge_graph 
        self.knowledge_graph = self.load_knowledge_graph("train")
        self.one_to_n_actions = self.set_one_to_n_actions()

        self.states = torch.Tensor(
            []
        )  # shape (num_steps, batch_size x num_rollouts, |state|(=3)]
        self.actions = torch.Tensor(
            []
        )  # shape (num_steps, batch_size x num_rollouts, |actions|(=2))
        self.goals = torch.Tensor([])  # shape (batch_size x rollouts, node(=1))
        
        self.available_actions = torch.Tensor(
            []
        )  # shape (batch_size x rollouts, number_of_avalable_actions + max_node_key_fillup (= max_degree), |actions|(=2))
        self.padding_mask = torch.Tensor(
            []
        )  # shape (batch_size x rollouts, max degree ); true if relation exists; false if relation does not exist

        self.interactive = interactive

        self.reward_config = {
            "lambda_factor": lambda_factor,
            "embedding_size": embedding_size,
            "mlp_size": mlp_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "use_entities": use_entities,
            "num_nodes": self.num_nodes,
            "padding_node_key": self.padding_node_key,
            "num_relations": self.num_relations,
            "padding_relation_key": self.padding_relation_key,
            "gamma": gamma,
            "top_k": top_k,
            "feedback_engine": feedback_engine,
            "error_rate": error_rate,
            "goal_loss_impact": goal_loss_impact,
            "batch_size": batch_size,
            "kwargs": {
                "triples": self.training_triples,
                "id_to_relation": self.id_to_relation,
                "id_to_entity": self.id_to_entity,
                "id_to_class": self.id_to_class,
                "loop_relation_key": self.loop_relation_key,
                "padding_relation_key": self.padding_relation_key,
                "roi": self.roi if localization else None,
            },
        }
        if interactive:
            self.reward_estimator = PreferenceBasedReward(
                lambda_factor=lambda_factor,
                embedding_size=embedding_size,
                mlp_size=mlp_size,
                dropout=dropout,
                learning_rate=learning_rate,
                use_entities=use_entities,
                num_nodes=self.num_nodes,
                padding_node_key=self.padding_node_key,
                num_relations=self.num_relations,
                padding_relation_key=self.padding_relation_key,
                gamma=gamma,
                top_k=top_k,
                feedback_engine=feedback_engine,
                error_rate=error_rate,
                goal_loss_impact=goal_loss_impact,
                batch_size=batch_size,
                kwargs=self.reward_config['kwargs'],
            )
        else:
            self.reward_estimator = BasicReward(
                lambda_factor=self.reward_config["lambda_factor"],
                gamma=self.reward_config["gamma"]
            )

        return

    def load_id_to_item(self, item_type: str) -> dict:
        """loads a mapping from ids (nodes or relations) to their items (nodes or relations).

        Args:
            item_type (str): either "node" or "relation"

        Returns:
            items_to_id (dict): Keys: item name; Value: id

        """

        assert item_type in [
            "entities",
            "relations",
            "class",
        ], f"item_type has to be either `entities`, `relations` or `class`, got {item_type}..."

        if item_type == "entities":
            path = self.path_to_knowledge_graph / "entity2id.txt"

        elif item_type == "relations":
            path = self.path_to_knowledge_graph / "relation2id.txt"

        if item_type == "class":
            path = self.path_to_knowledge_graph / "entity2id2class.txt"
            items = pd.read_csv(
                path, sep="\t", header=None, names=["entity", "id", "items"],
                index_col="id", dtype={"items": str, "id": int}
            )
        else:
            if self.knowledge_graph_name in ['NELL-995']:
                items = pd.read_csv(
                    path, sep="\t", header=None, names=["id", "items"], index_col="id", dtype={"items": str, "id": int}
                )
            else:
                items =pd.read_csv(
                    path, sep="\t", header=None, names=["items", "id"], index_col="id", dtype={"items": str, "id": int}
                )

        if item_type == "relations" and self.inverse_relations:
            inverse_relations = [f"{item}^-1" for item in items["items"].tolist()]
            inverse_relations_df = pd.DataFrame({"items": inverse_relations})
            inverse_relations_df.index.name = "id"
            items = pd.concat([items, inverse_relations_df], ignore_index=True)
            items.index.name = "id"

        # index should start at 1, such that we can use sparse matrix
        items.reset_index(inplace=True)
        items["id"] += 1
        items.set_index("id", inplace=True)

        return items.to_dict()["items"]

    def load_knowledge_graph(self, source: str) -> torch.Tensor:
        """loads knowledge graph from source. Populates one_to_n_actions. Sets self.padding.

        Args:
            source (str): either train/test/valid

        Returns:
            torch.Tensor: knowledge graph with the shape [num_entites, max_degree, (head, relation, tail)]
        """

        # load source text file
        path = self.path_to_knowledge_graph / f"{source}.txt"
        source_df = pd.DataFrame()
        try:
            if self.knowledge_graph_name in ['DBpedia50', 'DBpedia500', 'WN18']:
                source_df = pd.read_csv(path, sep="\t", names=["head", "tail","relation"], dtype={"head": str, "tail": str,"relation": str})
                source_df = source_df[["head", "relation", "tail"]]

            else:
                source_df = pd.read_csv(path, sep="\t", names=["head", "relation", "tail"], dtype={"head": str, "relation": str,"tail": str})
        except FileNotFoundError:
            print(
                f'The path "{path}" cannot be found. Please check if "{source}" is a valid source.'
            )
            sys.exit()

        if self.inverse_relations:
            inverse_df = source_df.copy()
            inverse_df["relation"] = inverse_df["relation"].map(lambda x: f"{x}^-1")
            inverse_df = inverse_df.rename(columns={"head": "tail", "tail": "head"})
            source_df = pd.concat([source_df, inverse_df], ignore_index=True)

        # map item names to id's
        source_df["head"] = source_df["head"].map(lambda x: self.entity_to_id[x])
        source_df["tail"] = source_df["tail"].map(lambda x: self.entity_to_id[x])
        source_df["relation"] = source_df["relation"].map(
            lambda x: self.relation_to_id[x]
        )

        actions_by_node = defaultdict(set)
        source_df.apply(
            lambda x: actions_by_node[x["head"]].add((x["relation"], x["tail"])), axis=1
        )

        # populate one_to_n_actions
        source_df.apply(
            lambda x: self.one_to_n_actions_dict[x["head"], x["relation"]].add(
                x["tail"]
            ),
            axis=1,
        )

        pruned_actions = {}
        for node, actions in actions_by_node.items():
            if len(actions) > self.max_degree and self.prune:
                pruned_actions[node] = set(random.sample(list(actions), self.max_degree))
            else:
                pruned_actions[node] = actions

        # update max_degree -> add 1 for loop-relation
        self.max_degree = max(len(actions) for actions in pruned_actions.values()) + 1

        knowledge_graph = torch.zeros(
            (self.num_nodes, self.max_degree, 3),
            dtype=torch.int32,
            requires_grad=False,
            device=self.device,
        )

        padding = torch.zeros_like(knowledge_graph)

        knowledge_graph[:, :, 1] = self.padding_relation_key
        knowledge_graph[:, :, 2] = self.padding_node_key

        knowledge_graph[self.padding_node_key, :, 0] = self.padding_node_key

        padding[:, :, 1] = self.padding_relation_key
        padding[:, :, 2] = self.padding_node_key
        padding[self.padding_node_key, :, 0] = self.padding_node_key

        # fill knowledge graph
        for key, values in pruned_actions.items():
            knowledge_graph[key, :, 0] = key
            padding[key, :, 0] = key
            knowledge_graph[key, 0, 1] = self.loop_relation_key
            knowledge_graph[key, 0, 2] = key
            action_idx = 1
            for value in values:
                knowledge_graph[key, action_idx, 1] = value[0]
                knowledge_graph[key, action_idx, 2] = value[1]
                action_idx += 1
        self.padding = padding

        return knowledge_graph

    def load_triples(self, source: str) -> torch.Tensor:
        """loads the triples form the dataset. Populates one_to_n_actions.

        Args:
            source (str): either train/test/valid

        Returns:
            torch.Tensor: the triples with the shape (num_instances, 3(head, relation, tail))
        """

        # load source text file
        path = self.path_to_knowledge_graph / f"{source}.txt"
        source_df = pd.DataFrame()
        try:
            if self.knowledge_graph_name in ['DBpedia50','DBpedia500', 'WN18']:
                source_df = pd.read_csv(path, sep="\t", names=["head", "tail","relation"], dtype={"head": str, "tail": str,"relation": str})
                source_df = source_df[["head", "relation", "tail"]]

            else:
                source_df = pd.read_csv(path, sep="\t", names=["head", "relation", "tail"], dtype={"head": str, "relation": str,"tail": str})
        except FileNotFoundError:
            print(
                f'The path "{path}" cannot be found. Please check if "{source}" is a valid source.'
            )
            sys.exit()

        # drop triples with relationstypes we do not care about
        if self.relation_types:
            source_df = source_df.loc[source_df['relation'].isin(self.relation_types)]
        # map item names to id's
        source_df["head"] = source_df["head"].map(lambda x: self.entity_to_id[x])
        source_df["tail"] = source_df["tail"].map(lambda x: self.entity_to_id[x])
        source_df["relation"] = source_df["relation"].map(
            lambda x: self.relation_to_id[x]
        )

        # populate one_to_n_actions
        source_df.apply(
            lambda x: self.one_to_n_actions_dict[x["head"], x["relation"]].add(
                    x["tail"]
            ),
                axis=1,
            )
        triples = torch.tensor(
            source_df.values, device=self.device, requires_grad=False, dtype=torch.int32
        )

        return triples

    def set_one_to_n_actions(self) -> torch.Tensor:
        """Transforms the one_to_n_actions_dict into a tensor. Deletes self.one_to_n_actions_dict afterwards.

        Returns:
            torch.Tensor: with the shape [num_nodes, num_relations, max_n]
        """

        max_n = 0

        for values in self.one_to_n_actions_dict.values():
            max_n = max(len(values), max_n)

        indicies = []
        vals = []
        for key, values in self.one_to_n_actions_dict.items():
            for idx, value in enumerate(values):
                indicies.append([key[0], key[1], idx])
                vals.append(value)

        del self.one_to_n_actions_dict
        one_to_n_actions = torch.sparse_coo_tensor(list(zip(*indicies)), vals,
                                                   size=(self.num_nodes, self.num_relations, max_n), 
                                                   dtype=torch.int32, device=self.device, requires_grad=False).coalesce()
        return one_to_n_actions

    def set_knowledge_graph_mask(self) -> torch.Tensor:
        """Setter for the padding mask of the knowledge graph.

        Returns:
            torch.Tensor: is true if action exists and false if action is a padding. Has the shape (num_entites, max_degree).
        """

        knowledge_graph_mask = torch.logical_not(
            self.knowledge_graph[:, :, 2] == self.padding_node_key
        )

        return knowledge_graph_mask

    def set_loop_relation_key(self) -> int:
        """calulates the key of the loop relation. Loop relation key is
        max relation key + 1.

        Returns:
            key(int): the key of the loop relation
        """
        return max(self.relation_to_id.values()) + 1

    def set_num_relations(self) -> int:
        """sets the number of relations via the max relation key. 

        Returns:
            key(int): the number of relations
        """
        return max(self.relation_to_id.values()) + 2

    def set_padding_relation_key(self) -> int:
        """calulates the key of the padding relation. Padding relation key is
        0.

        Returns:
            key(int): the key of the padding relation
        """
        return 0 

    def set_num_nodes(self) -> int:
        """sets the number of nodes via the max node key. 

        Returns:
            key(int): the number of nodes
        """
        return max(self.entity_to_id.values())+1

    def set_padding_node_key(self) -> int:
        """calulates the key of the padding node. Padding node key is
        0.

        Returns:
            key(int): the key of the padding node
        """
        return 0 

    def set_available_actions(
        self, action_dropout_rate: float = 0.0, penultimate_hop: bool = False
    ) -> None:
        """Sets self.available_actions and self.padding_mask. Is called by reset() and step().

        Args:
            action_dropout_rate (float, optional): Propability to dropout an action. Use only while training. Defaults to 0.0.
            penultimate_hop (bool, optional): true at the penultimate hop.
        """

        # get actions by node
        available_actions = self.knowledge_graph[self.states[-1, :, 0]] # what happens here?

        padding_mask = torch.logical_not(
             available_actions[:, :, 2] == self.padding_node_key
        )

        # get padding
        padding = self.padding[self.states[-1, :, 0]]

        # make supervision edge to padding
        supervision_triples = torch.cat(
            (self.states[0, :, 1:3], self.goals), dim=1
        ).unsqueeze(dim=1)

        is_supervision_triple = torch.all(
            available_actions == supervision_triples, dim=2
        )
        padding_mask = torch.where(is_supervision_triple, False, padding_mask)
        is_supervision_triple = is_supervision_triple.unsqueeze(dim=2).repeat_interleave(
            3, dim=2
        )

        available_actions = torch.where(
            is_supervision_triple, padding, available_actions
        )
        del is_supervision_triple 
        if penultimate_hop:

            batch_slices = gen_batches(n=supervision_triples.size(0), batch_size=512) # why batch size 512 ?
            mask = []
            for slice in batch_slices:

                one_to_n_slice = self.one_to_n_actions.index_select(
                    dim = 0, index = self.states[-1, slice, 1].long().to(self.device)
                )

                one_to_n = one_to_n_slice.to(self.device).to_dense()
                one_to_n = one_to_n[torch.arange(one_to_n.size(0), device=self.device),
                                    supervision_triples[slice, 0, 1],
                                    ]

                goal_mask = one_to_n == supervision_triples[slice, :, 2]

                one_to_n = torch.where(goal_mask, -1, one_to_n).unsqueeze(dim=1)

                available_actions_temp = (
                    available_actions[slice, :, 2]
                    .unsqueeze(dim=2)
                )
                mask_temp = torch.any(one_to_n == available_actions_temp, dim=2)
                mask.append(mask_temp)

            mask = torch.cat(mask, dim=0)
            padding_mask = torch.where(mask, False, padding_mask)
            mask = mask.unsqueeze(dim=2).repeat((1, 1, 3))
            available_actions = torch.where(mask, padding, available_actions)

        if action_dropout_rate > 0.:
            num_actions = padding_mask.sum(dim=1) 
            num_actions_after_dropout = torch.floor(padding_mask.sum(dim=1)*(1-action_dropout_rate)) 
            num_actions_after_dropout_mask = torch.logical_or((num_actions_after_dropout > 0), (num_actions == 0))
            num_actions_after_dropout = torch.where(num_actions_after_dropout_mask, num_actions_after_dropout, 1)
            # roll the dice
            rolls = torch.rand(padding_mask.size())
            # "drop" paddings
            rolls = torch.where(padding_mask, rolls, 0.)
            # force include correct target
            rolls = torch.where(available_actions[:,:,2]==supervision_triples[:,:,2], 1., rolls)
            rolls_mask = (rolls >= action_dropout_rate)
            # apply rolls
            padding_mask = torch.where(rolls_mask, padding_mask, False)
            rolls_mask = rolls_mask.unsqueeze(dim=2).repeat((1, 1, 3))
            available_actions = torch.where(rolls_mask, available_actions, padding)       

        # set available_actions
        self.available_actions = available_actions[:, :, 1:3]
        # set padding_mask
        self.padding_mask = padding_mask

        return

    def reset(
        self,
        sources: list[str],
        num_rollouts: int = 5,
        action_dropout_rate: float = 0.0,
        batch_slice: slice = None,
    ) -> None:
        """create batch of starting points, given sources (train, test, vaildation) and for specific relations(s).
        Calls set_availabel_actions.

        Args:
            sources (list[str]): A list with the sources the following batch is build of (i.e., [train, test, validation])
            num_rollouts (int, optional): The number of rollouts ber instance of the batch. Defaults to 5.
            action_dropout_rate (float), optional): The dropout rate of actions. Defaults to 0.0.
            batch_slice (slice, optional): The batch slice. Defaults to None.
        """
        assert len(sources) > 0, "`sources` has to contain at least one source type!"
        for source in sources:
            assert source in [
                "train",
                "test",
                "validation",
            ], f"source has to be either of type `train`, `test` or `validation`, got {source}..."

        # extract start_node, missing_relation, goal_node form the adjaceny matrix
        sources_list = []
        for source in set(sources):
            if source == "train":
                sources_list.append(self.training_triples)
            if source == "test":
                sources_list.append(self.test_triples)
            if source == "validation":
                sources_list.append(self.validation_triples)
        sources_instances = torch.cat(sources_list, dim=0)

        # create batch
        batch = sources_instances[batch_slice]
        # make rollouts
        batch = batch.repeat_interleave(repeats=num_rollouts, dim=0)

        # make states and actions
        current_nodes = torch.unsqueeze(batch[:, 0], 1)
        missing_relations = torch.unsqueeze(batch[:, 1], 1)

        states = torch.cat(
            (current_nodes, current_nodes, missing_relations), dim=1
        )  # (current_node, start_node, missing_relation)
        actions = torch.cat(
            (missing_relations, current_nodes), dim=1
        )  # (relation, node)

        self.states = torch.unsqueeze(states, 0)
        self.actions = torch.unsqueeze(actions, 0)
        self.goals = torch.unsqueeze(batch[:, 2], 1)

        self.set_available_actions(action_dropout_rate)

        return

    def step(
        self,
        action_idx: torch.Tensor,
        one_to_n_trigger: bool,
        action_dropout_rate: float = 0.0,
        ultimate_hop: bool = False,
    ) -> None:
        """executes the action choosen by the agent. This is achieved by updating self.actions, self.states,
        and self.available_actions. Calls set_availabel_actions.

        Args:
            action_idx (torch.Tensor): 2d tensor with the indicies of the actions.
            action_dropout_rate (float, optional): The dropout rate of actions. Defaults to 0.0.
            ultimate_hop (bool): Is this the step of the final hop? Defaults to False.
        """
        # get the actual actions the agent took
        actions = self.available_actions[
            torch.arange(self.available_actions.size(0), device=self.device),
            action_idx.squeeze(),
        ]

        # update actions
        self.actions = torch.cat((self.actions, torch.unsqueeze(actions, dim=0)), dim=0)
        # update states
        new_states = torch.cat(
            (torch.unsqueeze(actions[:, 1], dim=1), self.states[0, :, 1:3]), dim=1
        )
        self.states = torch.cat((self.states, torch.unsqueeze(new_states, dim=0)))
        # update available actions
        if not ultimate_hop:
            self.set_available_actions(action_dropout_rate, one_to_n_trigger)
        return

    def shuffel_training_instances(self) -> None:
        """shuffels the training instances. For usage at the beginning of training epochs."""
        self.training_triples = self.training_triples[
            torch.randperm(self.training_triples.size(0))
        ]
        return

    def adapt_environment(self, batch_idx: torch.Tensor) -> None:
        """mutates environment, such that self.states, self.actions, self.available_actions,
        fit to the instances selected by the beam search.

        Args:
            idx (torch.Tensor): The indices of the instances selected by the beam search
        """
        self.states = self.states[:, batch_idx]
        self.actions = self.actions[:, batch_idx]
        self.goals = self.goals[batch_idx]
        self.available_actions = self.available_actions[batch_idx]
        return

    def collect_paths(self) -> torch.Tensor:
        """
        Given the environment, returns a tensor of all predicted paths produced by the model, specifically used for localization scores.
        """
        paths = torch.stack(
            tensors=(
                self.states[:-1, :, 0],
                self.actions[1:, :, 0],
                self.states[1:, :, 0]
            ),
            dim=-1
        ) # has shape: (steps, test_triples x num_rollouts, 3)
        return paths

    def prepare_roi(self) -> Dict:
        """
        Reads in directory of ROI (regions of interest) files into a dictionary.
        
        Returns:
            roi_dict (Dict): ROI dictionary with relation types as keys
        """
        roi_dict = {}
        roi_files = [x for x in Path(self.roi_dir).iterdir() if (x.is_file() and x.suffix =='.json')]
        for roi_file in roi_files:
            with open(roi_file, mode='r') as file:
                loaded_json = json.load(file)

            all_rois = {}
            for k, v in loaded_json.items():
                roi_path = []
                in_flag = False
                for sublist in v['formattedPath']:
                    relation = 'none'
                    if 'in' in sublist:
                        in_flag = True
                        relation = f'{sublist[1]}^-1'
                    elif 'out' in sublist:
                        relation = sublist[1]
                    entity = sublist[0]
                    roi_path.extend([entity, relation] if len(sublist)== 3 else [entity])
                if self.inverse_relations or not in_flag:
                    all_rois[k] = roi_path

            roi_dict[roi_file.stem] = all_rois
        return roi_dict

    def delexicalize_paths(
            self,
            paths: torch.Tensor,
            num_rollouts: int
        ) -> List[List[str]]:
        """
        Delexicalize predicted paths by reducing entities to their classes.

        For each alternating entity–relation path produced by the model, entities
        are replaced with their corresponding class labels. This abstracts concrete
        reasoning chains into higher‐level patterns.

        Example:
            Given the model output for query (Bavaria, inContinent, ?):
                Bavaria -> regionOf -> Germany -> inContinent -> Europe
            Delexicalization yields:
                region -> regionOf -> country -> inContinent -> continent

        Args:
            paths (Tensor): A tensor of shape (steps, rollouts, 3) containing the predicted [entity, relation, entity] sequences for each rollout step.
            num_rollouts (int): Number of rollouts (i.e., parallel reasoning paths) per query.

        Returns:
            List[List[str]]: A list of delexicalized paths grouped by rollout. Each inner list represents the sequence of abstract tokens (classes and relations) for one rollout.
        """

        num_paths = paths.shape[1]
        delexicalized_paths = []
        rollout_paths = []
        for path_idx in range(num_paths):
            path = paths[:, path_idx, :]
            head_ids = path[:, 0].tolist()
            rel_ids = path[:, 1].tolist()
            tail_ids = path[:, 2].tolist()

            delex_path = [self.id_to_class[head_ids[0]]]
            for rel, tail in zip(rel_ids, tail_ids):
                if rel != self.loop_relation_key and rel != self.padding_relation_key:
                    delex_path.append(self.id_to_relation[rel])
                    delex_path.append(self.id_to_class[tail])
            rollout_paths.append(delex_path)
            if (path_idx+1) % num_rollouts == 0: # to isolate all the paths belonging to one query
                delexicalized_paths.append(rollout_paths)
                rollout_paths = []
        return delexicalized_paths

    def set_reward(self, reward_type: str) -> None:
        """
        Configure the agent’s reward estimator based on the specified type.

        Args:
            reward_type (str): Type of reward estimator to initialize. Supported values:
                - "interactive": Sets up a PreferenceBasedReward using parameters
                from self.reward_config for human- or automated-preference feedback.
                - "basic": Sets up a BasicReward.

        Returns:
            None
        """
        if reward_type == "interactive":
            self.reward_estimator = PreferenceBasedReward(
                lambda_factor=self.reward_config["lambda_factor"],
                embedding_size=self.reward_config["embedding_size"],
                mlp_size=self.reward_config["mlp_size"],
                dropout=self.reward_config["dropout"],
                learning_rate=self.reward_config["learning_rate"],
                use_entities=self.reward_config["use_entities"],
                num_nodes=self.reward_config["num_nodes"],
                padding_node_key=self.reward_config["padding_node_key"],
                num_relations=self.reward_config["num_relations"],
                padding_relation_key=self.reward_config["padding_relation_key"],
                gamma=self.reward_config["gamma"],
                top_k=self.reward_config["top_k"],
                feedback_engine=self.reward_config["feedback_engine"],
                error_rate=self.reward_config["error_rate"],
                goal_loss_impact=self.reward_config["goal_loss_impact"],
                batch_size=self.reward_config["batch_size"],
                kwargs=self.reward_config["kwargs"],
            )
        if reward_type == "basic":
            self.reward_estimator = BasicReward(
                lambda_factor=self.reward_config["lambda_factor"],
                gamma=self.reward_config["gamma"],
            )
        return
