import numpy as np
import torch
import torch.nn as nn
from torch import optim
from prettytable import PrettyTable
from sklearn.utils import gen_batches
import time
from lstms import MultiLayerLSTM, LayerNormLSTM, LSTM
from src.logger import Logger
import torch.nn.functional as F
import random

logger = Logger()

from src.reward_estimator import Reward
from src.interactive_reward_modules.pairs_storage import PairsStorage


class PreferenceBasedReward(Reward):
    def __init__(self, lambda_factor, embedding_size, mlp_size, dropout, learning_rate, use_entities, num_nodes, padding_node_key, num_relations, padding_relation_key,
                 gamma=0.99, num_predictors=3, top_k=10, max_num_epochs=500, batch_size=64, early_stopper_patience=4, 
                 early_stopper_delta=0.05, feedback_engine = "automated", error_rate=-1, goal_loss_impact=0.1, kwargs={}):
        super().__init__(lambda_factor, gamma)
        # use_entities = True
        self.paris_storage = PairsStorage(top_k, use_entities, feedback_engine, error_rate, kwargs)

        self.learning_rate = learning_rate
        self.predictor_ensemble = [PreferenceBasedEstimator(self.device, embedding_size, mlp_size, num_nodes, padding_node_key, num_relations, padding_relation_key, dropout, learning_rate, use_entities) for _ in range(num_predictors)]
        self.early_stopper = EarlyStopper(patience=early_stopper_patience, min_delta=early_stopper_delta) 

        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size

        self.use_entities = use_entities
        self.training_flag = False
        self.goal_loss_impact = goal_loss_impact

    def __call__(self, previous_actions:torch.Tensor, current_states:torch.Tensor, hop_counter:int, ultimate_hop:bool) -> None:
        """ Save the reward and uncertainty of the reward ensemble

        Args:
            previous_actions (torch.Tensor): the executed actions 
            current_state (torch.Tensor): and the current states
            hop_counter (int): the number of the current hop. 
        """     

        [predictor.eval() for predictor in self.predictor_ensemble]
        ensemble_estimates = self.ensemble_predict(previous_actions, current_states, hop_counter)
        # ensemble_estimates = [self.normalize_reward(reward) for reward in ensemble_estimates]
        estimates = torch.cat((ensemble_estimates),dim=1).detach()
        estimates.requires_grad = False
        # TODO: do not use uncertainty but discrepancy between estimate and correct result
        uncertainty = torch.var(estimates, dim=1)
        if False:
            rewards = torch.eq(current_states[:,0].unsqueeze(1)).to(torch.float32)
            # self.uncertainty = torch.abs(rewards - torch.mean(estimates, dim=1).unsqueeze(dim=1)).squeeze(dim=1)
        else:     
            rewards = torch.mean(estimates, dim=1).unsqueeze(dim=1)
        # save reward and uncertainty
        if ultimate_hop:
            self.rewards = torch.cat((self.rewards, rewards), axis=1)
        else:
            self.rewards = torch.cat((self.rewards, torch.zeros_like(rewards)), axis=1)
        self.uncertainty += uncertainty

        return

    def ensemble_predict(self, actions: torch.Tensor, states:torch.Tensor, hop:int) -> tuple[torch.Tensor]:
        """ predicts the reward over a batch for every member of the ensemble

        Args:
            actions (torch.Tensor): the previously choosen actions
            states (torch.Tensor): the current states
            hop (int): the number of the current hop. 

        Returns:
            tuple(torch.Tensor): the rewards calculated by each member of the predictor ensemble. 
        """
        hop = torch.empty(actions.size(0)).fill_(hop)
        rewards= [predictor(actions, states, hop) for predictor in self.predictor_ensemble]
        return rewards

    def normalize_reward(self, reward, epsilon = 1e-7):
        # rewards_stacked = torch.stack(rewards, dim=0)
        mean = torch.mean(reward)
        var = torch.var(reward)
        reward = (reward - mean)/torch.sqrt(var + epsilon) 
        return reward

    def save_paths(self, states:torch.Tensor, actions:torch.Tensor, goals:torch.Tensor)-> None:
        """ saves paths for PreferenceBasedReward training

        Args:
            actions (torch.Tensor): actions over all hops
            states (torch.Tensor): states over all hops
            goals (torch.Tensor): The goals of all instances of a batch - only used if feedback is automated.
        """
        # kick out the first entries in states and actions, as they are not relevant
        # to calculate the reward
        self.paris_storage(states[1:], actions[1:], goals, self.uncertainty)
        return

    def ensemble_predict_pairs(self, pairs_actions, pairs_states):

        # original shape
        original_size= pairs_actions.size() # [batch_dim, pair_dim, step dim, action]
        # exit()
        # actions = pairs_actions.reshape(-1, 2)
        # states = pairs_states.reshape(-1, 3)
        # predict
        pairs_temp = []
        for path in range(original_size[1]):
            hops_temp = []
            for hop in range(original_size[2]):
                ensemble_estimates = self.ensemble_predict(pairs_actions[:,path,hop,:], pairs_states[:,path,hop,:], hop)
                ensemble_estimates = torch.cat((ensemble_estimates),dim=1)
                hops_temp.append(ensemble_estimates)    
            pairs_temp.append(torch.stack(hops_temp,dim=2))    
        ensemble_estimates = torch.stack(pairs_temp, dim=2)
        ensemble_estimates = ensemble_estimates.sum(dim=3)

        return ensemble_estimates

    def is_training_feasable(self):

        self.training_flag = self.paris_storage.training_flag

        return self.training_flag

    def training(self):
        # collect preferences
        self.paris_storage.collect_and_store_feedback()
        # if we get the signal that training is not feasable, than do not train.
        if not self.is_training_feasable():
            return
        random.shuffle(self.predictor_ensemble)
        for num_predictor, predictor in enumerate(self.predictor_ensemble):
            # init early_stopper
            # logger().debug(num_predictor)
            self.early_stopper.reset()
            # init epoch
            iteration = 0
            for _ in range(self.max_num_epochs):
                logger().debug(f"Predictor {num_predictor} Epoch {iteration} started")
                # set predictors to training
                # predictor.train()
                [pred.eval() if (pred != predictor) else pred.train() for pred in self.predictor_ensemble]
                # [pred.train() for pred in self.predictor_ensemble]
                # init running_train_loss
                running_train_loss = torch.tensor(0, device=self.device)

                self.paris_storage.shuffel_training_instances()
                if self.paris_storage.train_split_temp.size(0) == 0:
                    return 
                # init batching
                num_slices = 0
                for slice in gen_batches(n=self.paris_storage.train_split_temp.size(0), batch_size=self.batch_size):

                    pairs_states = self.paris_storage.get_pairs_states(slice, "train")
                    pairs_actions = self.paris_storage.get_pairs_actions(slice, "train")
                    pairs_goals = self.paris_storage.get_pairs_goals(slice, "train")
                    preferences = self.paris_storage.get_preferences(slice, "train")

                    pairs_temp = []
                    for path in range(pairs_actions.size(1)):
                        self.reset_hidden_states()
                        hops_temp = torch.zeros(pairs_actions.size(0), 1)
                        for hop in range(pairs_actions.size(2)):
                            hop_tensor = torch.empty(pairs_actions.size(0)).fill_(hop)
                            # estimates = predictor(pairs_actions[:,path,hop,:], pairs_states[:,path,hop,:], hop_tensor)
                            estimates = [pred(pairs_actions[:,path,hop,:], pairs_states[:,path,hop,:], hop_tensor) for pred in self.predictor_ensemble]
                            estimates_detached = [estimate.detach() if (num_estimate != num_predictor) else estimate for num_estimate, estimate in enumerate(estimates)]
                            estimates_mean = (
                                torch.cat(estimates_detached, dim=1)
                                .mean(dim=1)
                                .unsqueeze(dim=1)
                            )
                            hops_temp += estimates_mean 
                        pairs_temp.append(hops_temp)    
                    estimates = torch.cat(pairs_temp, dim=1)

                    # calculate loss
                    train_loss = self.calculate_loss(
                        estimates,
                        preferences,
                        pairs_states[:, :, -1, 0],
                        pairs_goals[:, :],
                    ) 
                    # update running training loss
                    running_train_loss = torch.add(running_train_loss, train_loss) 
                    # train the ensemble
                    # [pred.optimizer.zero_grad() for pred in self.predictor_ensemble]
                    # w0 = predictor.linear5.weight.clone()
                    predictor.optimizer.zero_grad()
                    train_loss.backward() 
                    # for name, p in predictor.named_parameters():
                    #    if p.grad is None:
                    #        print(f"[WARN] {name} has no grad")
                    #    else:
                    #        print(f"{name:30s} grad norm = {p.grad.norm().item():.3e}")
                    predictor.optimizer.step()
                    # [pred.optimizer.step() for pred in self.predictor_ensemble]
                    # w1 = predictor.linear5.weight.clone()
                    # print("Δnorm( linear5.weight ) =", (w1 - w0).norm().item())

                    num_slices += 1
                iteration +=1
                running_train_loss = running_train_loss / num_slices
                # logger().debug(f"Train loss: {running_train_loss} at iteration {iteration}")
                # set predictor to validation
                # predictor.eval()
                [pred.eval() for pred in self.predictor_ensemble]    
                # init running validation loss
                running_validation_loss = torch.tensor(0, device=self.device)
                num_slices = 0

                with torch.no_grad():
                    self.paris_storage.balance_validation_instances()
                    for slice in gen_batches(n=self.paris_storage.valid_split_temp.size(0), batch_size=self.batch_size):
                        pairs_states = self.paris_storage.get_pairs_states(slice, "valid")
                        pairs_actions = self.paris_storage.get_pairs_actions(slice, "valid")
                        pairs_goals = self.paris_storage.get_pairs_goals(slice, "valid")
                        preferences = self.paris_storage.get_preferences(slice, "valid")
                        pairs_temp = []
                        for path in range(pairs_actions.size(1)):
                            self.reset_hidden_states()
                            hops_temp = torch.zeros(pairs_actions.size(0), 1)
                            for hop in range(pairs_actions.size(2)):
                                hop_tensor = torch.empty(pairs_actions.size(0)).fill_(hop)
                                # estimates = predictor(pairs_actions[:,path,hop,:], pairs_states[:,path,hop,:], hop_tensor)
                                # hops_temp += estimates
                                estimates = [pred(pairs_actions[:,path,hop,:], pairs_states[:,path,hop,:], hop_tensor) for pred in self.predictor_ensemble]
                                estimates_mean = torch.cat(estimates, dim=1).mean(dim=1).unsqueeze(dim=1)
                                # estimates_mean = estimates[num_predictor]
                                hops_temp += estimates_mean
                            pairs_temp.append(hops_temp)
                        estimates = torch.cat(pairs_temp, dim=1)
                        # calculate loss
                        validation_loss = self.calculate_loss(
                            estimates,
                            preferences,
                            pairs_states[:, :, -1, 0],
                            pairs_goals[:, :],
                        )
                        # update running validation loss
                        running_validation_loss = torch.add(running_validation_loss, validation_loss)
                        num_slices += 1
                    running_validation_loss = running_validation_loss / num_slices
                # logger().debug(f"Validation loss: {running_validation_loss} at iteration {iteration}")
                # stop if pattern of increasing validation loss detected
                if self.early_stopper.early_stop(running_validation_loss):
                    # print("valid", running_validation_loss)
                    break
        # self.paris_storage.reset_storage()
        return

    def calculate_loss(self, reward_estimates:torch.Tensor, preferences: torch.Tensor, states, goals)-> torch.Tensor:
        """
        Calculates the preference-based loss (bradley terry model).

        Args:
            reward_estimates (torch.Tensor): with shape [batch_dim, pair_dim(=2)]
            preferences (torch.Tensor): the preferences with the shape [batch_size]
        Returns:
            loss(torch.Tensor): the loss of the current ensemble
        """

        goals = goals.squeeze() # get rid of unncessary 2 dim
        goal_mask = states == goals
        valid = goal_mask.sum(dim=1) == 1
        re_valid    = reward_estimates[valid]  # [n_valid, 2]
        gm_valid    = goal_mask   [valid]      # [n_valid, 2]

        # +1 if path0 hit, –1 if path1 hit
        signs = gm_valid[:,0] * 2 - 1          # [n_valid] in {+1, –1}

        # margin-ranking loss

        goal_loss = F.margin_ranking_loss(
            re_valid[:,0],   # scores for path 0
            re_valid[:,1],   # scores for path 1
            signs.float(),   # +1 or –1
            margin=1.0,
            reduction='mean'
        )

        # goals = torch.concat((goals[:,0], goals[:,1]), axis =0) # untie pairs
        # states = torch.concat((states[:, 0], states[:, 1]), axis=0)  # untie pairs
        # goal_mask = (states == goals)
        # untied_reward = torch.concat(
        #    (reward_estimates[:, 0], reward_estimates[:, 1]), axis=0
        # )  # untie pairs
        # bce = nn.BCEWithLogitsLoss()
        # goal_loss = bce(untied_reward, goal_mask.float())

        # calculate logsoftmax for pair_dim
        log_softmax = nn.functional.log_softmax(reward_estimates, dim=1)
        # log_softmax = torch.log(torch.pow(nn.functional.softmax(reward_estimates, dim=1),2))
        loss = torch.add(torch.mul(preferences, log_softmax[:, 0]), torch.mul(1-preferences, log_softmax[:,1]))
        preference_loss = torch.neg(torch.mean(loss))
        loss = preference_loss + self.goal_loss_impact * goal_loss # Todo: make 0.1 to hyperparameter
        return loss

    def reset_hidden_states(self)->None:
        for predictor in self.predictor_ensemble:
            predictor.hidden_state = None
        return


class PreferenceBasedEstimator(nn.Module):
    def __init__(self, device, embedding_size, mlp_size, num_nodes, padding_node_key, num_relations, padding_relation_key, dropout=0.1, learning_rate=0.001 ,use_entities=False):
        super(PreferenceBasedEstimator, self).__init__()
        self.use_entites = use_entities
        self.padding_relation_key = padding_relation_key
        # embeddings
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_size, padding_idx=padding_node_key, dtype=torch.float32, device=device, scale_grad_by_freq=True)
        self.relation_embedding = nn.Embedding(num_embeddings=num_relations, embedding_dim=embedding_size, padding_idx=padding_relation_key ,dtype=torch.float32, device=device, scale_grad_by_freq=True)
        if use_entities:
            input_size_factor = 4
        else: 
            input_size_factor = 2
        # define layers
        self.hop_embedding = nn.Linear(in_features=1, out_features=12)

        self.lstm = MultiLayerLSTM(
            input_size=(input_size_factor * embedding_size + 12),
            layer_type=LayerNormLSTM,
            layer_sizes=(
                input_size_factor * embedding_size * 2 + 12,
                # input_size_factor * embedding_size * 2,
                input_size_factor * embedding_size,
            ),
            bias=True,
            dropout=dropout,
            dropout_method="pytorch",
        )

        self.linear0 = nn.Linear(in_features=(input_size_factor+1)*embedding_size, out_features=mlp_size*2)
        self.gelu_0 = nn.GELU()
        self.linear1 = nn.Linear(in_features=mlp_size*2, out_features=int(mlp_size/2))
        self.gelu_1 = nn.GELU()
        self.linear5 = nn.Linear(in_features=int(mlp_size/2), out_features=1)

        self.optimizer = optim.AdamW(self.parameters(), 
                           lr=learning_rate, weight_decay=0.001)
        self.hidden_state = None 

        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_weights)

    def forward(self, actions, states, hop):

        if not self.hidden_state:
            batch_size = actions.size(0)
            self.hidden_state = self.lstm.create_hiddens(batch_size)
        actions = self.relation_embedding(actions[:,0])
        missing_relations = self.relation_embedding(states[:,2])

        observation = torch.cat((actions, missing_relations), dim=1)
        if self.use_entites: 
            current = self.node_embedding(states[:,0])                    
            start = self.node_embedding(states[:,1])
            observation = torch.cat((observation, current, start),dim=1)

        hop_embedding = self.hop_embedding(hop.unsqueeze(dim=1))
        observation = torch.cat((observation, hop_embedding), dim=1).unsqueeze(dim=0))
        x, self.hidden_state = self.lstm(observation, self.hidden_state)
        x = x.squeeze(dim = 0)
        x = self.linear0(torch.cat((x, missing_relations), dim=1))
        x = self.gelu_0(x)
        x = self.linear1(x)
        x = self.gelu_1(x)
        x = self.linear5(x)

        return x 


class EarlyStopper:
    """EarlyStopper implementation from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    Returns:
        EarlyStopper: early stopper instance 
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False

        self.counter += 1
        return (self.counter >= self.patience)

    def reset(self):
        self.counter = 0
        self.min_validation_loss = torch.inf
