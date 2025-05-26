import torch
from torch import optim
from sklearn.utils import gen_batches
import random
from tqdm import tqdm


from src.evaluation import Evaluation
from src.policy_network import PolicyNetwork
from src.environment import Environment
from src.reward_estimator import BasicReward
from src.utility import delete_tensors

class Trainer:
    """The trainer for the model"""
    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        num_hops: int,
        rollouts_train: int,
        rollouts_test: int,
        action_dropout_rate: float = 0.4,
        beta: float = 0.05,
        decay_rate: float = 0.9,
        learning_rate: float = 0.01,
        pretraining_epochs: int = 5, # number of epochs for pretraining
        patients: int = 5, # number of epochs for gathering pairs before feedback-based training starts
        fit_reward_epochs: str = "zero"
    ):
        """Inits the trainer for the model.

        Args:
            batch_size (int): the batch size
            num_epochs (int): the number of training epochs
            num_hops (int): the number of hops (steps)
            rollouts_train (int): number of rollouts per training instance
            rollouts_test (int): number of rollouts per test instance
            action_dropout_rate (float, optional): Propability to dropout an action. Use only while training. Defaults to 0.0.
            beta (float, optional): beta factor to discount entropy in reinforcement loss. Defaults to 0.05.
            decay_rate (float, optional): decay rate to discount entropy in reinforcement loss. Defaults to 0.9.
            learning_rate (float, optional): learning rate of the model. Defaults to 0.01.
            pretraining (int, optional): Number of pretraining epochs. Defaults to 5.
            patients (int, optional): Patients before starting to train to collect paths for feedback. Defaults to 5.
            fit_reward_epochs (str, optional): strategy on when to train the reward. Defaults to zero.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dropout_rate = action_dropout_rate
        self.learning_rate = learning_rate
        self.beta = beta
        self.decay_rate = decay_rate

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.rollouts_train = rollouts_train
        self.num_hops = num_hops

        self.evaluation = Evaluation(num_hops, rollouts_test)

        # put metrices here
        self.epoch_counter = 0
        self.global_step = 1
        self.loss_by_epoch = []
        self.hits_by_epoch = []
        self.reward_by_epoch = []
        self.feedback_counter = 0

        # init mode
        self.stalling_flag = True
        self.pretraining_epochs = pretraining_epochs
        self.patients = patients

        fit_reward_epochs_choices = {
            'zero': (0,), 
            'zero-five-ten': (0, 5, 10),
            'zero-five-ten-fifteen':(0, 5, 10, 15),
            'zero-to-four':(0, 1, 2, 3, 4),
            'five-steps':tuple(range(0, 150, 5)),
            'ten-steps':tuple(range(0, 150, 10)),
            'exponential':tuple(i * i for i in range(22))
            }
        self.fit_reward_epochs = fit_reward_epochs_choices[fit_reward_epochs]

    def __call__(
        self,
        model: PolicyNetwork,
        environment: Environment,
        eval_epochs: list[int] = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
    ):
        """trains the model in the environment. The exploration strategy is a biased random search. Calls self.epoch().

        Args:
            model (PolicyNetwork): the model.
            environment (Environment): the environment.
            eval_epochs (list, optional): Indicates in which epochs to evaluate the model.
                                          Defaults to [0, 100, 200, 300, 400, 500, 600, 700, 800, 900].

        Returns:
            (PolicyNetwork, list[float], list[float], list[float], dict): returns the model
                                                                          and some metrices about
                                                                          the training and evaluation.
        """

        # init optimizer
        optimizer_dense = optim.AdamW(list(model.parameters())[:-2], lr=self.learning_rate)
        optimizer_sparse = optim.SparseAdam(list(model.parameters())[-2:], lr=self.learning_rate*10)
        optimizers = [optimizer_dense, optimizer_sparse]
        print("Lets start with learning!")

        if self.pretraining_epochs > 0 and environment.interactive:
            environment.set_reward(reward_type='basic')

        for _ in range(self.num_epochs):

            if self.pretraining_epochs == self.epoch_counter and environment.interactive:

                environment.set_reward(reward_type="interactive")

            self.epoch(
                    environment=environment, model=model, optimizers=optimizers
                    )

            if not isinstance(environment.reward_estimator, BasicReward) and (
                (self.epoch_counter - self.pretraining_epochs)
                in self.fit_reward_epochs
            ):
                self.feedback_counter += 1
                environment.reward_estimator.training()

            if self.epoch_counter in eval_epochs:
                # set model to eval mode
                model.eval()
                with torch.no_grad():
                    self.evaluation(
                        epoch=self.epoch_counter, model=model, environment=environment
                    )
            self.epoch_counter += 1

        return (
            model,
            self.reward_by_epoch,
            self.loss_by_epoch,
            self.hits_by_epoch,
            dict(self.evaluation.metrics),
        )

    def epoch(
        self,
        environment: Environment,
        model: PolicyNetwork,
        optimizers: [torch.optim.Adam, torch.optim.SparseAdam]
    ):
        """trains the model for one epoch. Updates self.loss_by_epoch, self.reward_by_epoch, and self.hits_by_epoch. Calls self.batch().

        Args:
            environment (Environment): the environment.
            model (PolicyNetwork): the model.
            optimizer ([torch.optim.Adam, torch.optim.SparseAdam]): Adam optimizer for sparse and dense gradient.
        """
        print(f"Policy network training epoch {self.epoch_counter} started")
        # set model to train mode
        model.train()
        # shuffel training instances
        environment.shuffel_training_instances()

        # slices training set to batches
        batch_slices = list(
            gen_batches(
                n=environment.training_triples.size(0), batch_size=self.batch_size
            )
        )
        random.shuffle(batch_slices)

        first_batch_flag = True
        epoch_log = tqdm(total=0, position=1, bar_format='{desc}')
        for batch_slice in tqdm(batch_slices):
            hits, loss, reward = self.batch(
                batch_slice=batch_slice,
                environment=environment,
                model=model,
                optimizers=optimizers,
                epoch_log=epoch_log
            )
            if first_batch_flag:
                running_loss = loss
                running_reward = reward
                first_batch_flag = False
                absolute_hits = hits
            else:
                running_loss = running_loss * 0.9 + loss * 0.1
                running_reward = running_reward * 0.9 + reward * 0.1
                absolute_hits += hits

        self.loss_by_epoch.append(running_loss)
        self.reward_by_epoch.append(running_reward)
        relativ_hits = absolute_hits / environment.training_triples.size(0)
        self.hits_by_epoch.append(relativ_hits)
        del epoch_log
        print(
            f"Ep {self.epoch_counter}: \n \
                    Average reward: {running_reward} \n \
                    Absolut hits: {absolute_hits}  \n \
                    Relative hits: {relativ_hits} \n \
                    Running loss: {running_loss} \n"
        )
        print("_____________________________________________")

        return 

    def batch(
        self,
        batch_slice: slice,
        environment: Environment,
        model: PolicyNetwork,
        optimizers: [torch.optim.Adam, torch.optim.SparseAdam],
        epoch_log: tqdm
    ):
        """Trains the model on one batch. This includes calculating the loss for the batch and backpropagation. Calls self.hop() and self.calculate_reinforcement_loss().

        Args:
            batch_slice (slice): the batch slice.
            environment (Environment): the environment
            model (PolicyNetwork): the model.
            optimizer ([torch.optim.Adam, torch.optim.SparseAdam]): Adam optimizer for sparse and dense gradient.

        Returns:
            (int, int, int):  the number of hits, the loss, and average_reward of the batch
        """
        real_batch_size = batch_slice.stop - batch_slice.start
        # reset environment
        environment.reset(
            sources=["train"],
            num_rollouts=self.rollouts_train,
            action_dropout_rate=self.action_dropout_rate,
            batch_slice=batch_slice,
        )
        # reset reward
        environment.reward_estimator.reset()
        # init hidden states
        hidden_state = None
        missing_relation_embeddings = None

        for hop_counter in range(self.num_hops):
            (
                hidden_state,
                missing_relation_embeddings,
                logits,
                loss,
            ) = self.hop(
                hidden_state=hidden_state,
                missing_relation_embeddings=missing_relation_embeddings,
                hop_counter=hop_counter,
                model=model,
                environment=environment,
            )

            if hop_counter == 0:
                logits_steps = torch.unsqueeze(logits, 1)
                loss_steps = torch.unsqueeze(loss, 1)

            else:
                logits_steps = torch.cat(
                    (logits_steps, torch.unsqueeze(logits, 1)), dim=1
                )
                loss_steps = torch.cat((loss_steps, torch.unsqueeze(loss, 1)), dim=1)

        if not isinstance(environment.reward_estimator, BasicReward):
            environment.reward_estimator.save_paths(environment.states, environment.actions, environment.goals)

        # discount reward
        environment.reward_estimator.discount_rewards()

        # calculate loss
        loss = self.calculate_reinforcement_loss(
            discounted_rewards=environment.reward_estimator.discounted_rewards,
            baseline=environment.reward_estimator.baseline.get_baseline_value(),
            logits_steps=logits_steps,
            loss_steps=loss_steps,
        )

        if (
            (self.pretraining_epochs - self.patients) >= self.epoch_counter
            or (self.pretraining_epochs + self.patients) <= self.epoch_counter
            or not environment.interactive
        ):
            # update_baseline
            environment.reward_estimator.update_baseline()
            # Calculate gradients
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, foreach=False)
            # Apply gradients
            for optimizer in optimizers:
                # print("optimizer step is done")
                optimizer.step()
            self.global_step += 1

        # caluclate metrices
        hits = environment.reward_estimator.get_hits(goals=environment.goals,
                current_states=environment.states[-1, :, 0], shape=(real_batch_size, self.rollouts_train))
        loss = loss.item()
        average_reward = torch.mean(
            environment.reward_estimator.rewards
        ).item()
        
        return (
            hits,
            loss,
            average_reward,
        )

    def hop(
        self,
        hidden_state: tuple[torch.Tensor],
        missing_relation_embeddings: torch.Tensor,
        model: PolicyNetwork,
        environment: Environment,
        hop_counter: int,
    ):
        """Executes one hop of the model.

        Args:
            hidden_state (tuple[torch.Tensor]): the hidden state of the policy network.
            missing_relation_embeddings (torch.Tensor): the embedding of the missing relations.
            model (PolicyNetwork): the model.
            environment (Environment): the environment.
            hop_counter (int): the number of the current hop.

        Returns:
            (tuple[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor) : returns the hidden_state, missing_relation_embeddings,
                                                                                logits, and loss of a hop.
        """
        states = environment.states[-1]
        actions = environment.actions[-1]
        available_actions = environment.available_actions
        padding_mask = environment.padding_mask
        logits, loss, action_idx, hidden_state, missing_relation_embeddings = model(
            actions,
            states,
            available_actions,
            padding_mask,
            hidden_state,
            missing_relation_embeddings,
        )
        delete_tensors([actions, states, available_actions, padding_mask])

        ultimate_hop = (hop_counter + 1) == self.num_hops
        penultimate_hop = (hop_counter + 2) == self.num_hops
        environment.step(
            action_idx,
            penultimate_hop,
            action_dropout_rate=self.action_dropout_rate,
            ultimate_hop=ultimate_hop,
        )
        # estimates the reward of the hop
        if not isinstance(environment.reward_estimator, BasicReward):
            if hop_counter == 0:
                environment.reward_estimator.reset_hidden_states()

            environment.reward_estimator(
                    previous_actions=environment.actions[-1],
                    current_states=environment.states[-1],
                    hop_counter = hop_counter,
                    ultimate_hop = ultimate_hop
                )
        else:
            environment.reward_estimator(
                goals=environment.goals,
                current_states=environment.states[-1, :, 0],
                ultimate_hop=ultimate_hop,
            )
        return (
            hidden_state,
            missing_relation_embeddings,
            logits,
            loss,
        )

    def calculate_reinforcement_loss(
        self,
        discounted_rewards: torch.Tensor,
        baseline: torch.Tensor,
        logits_steps: torch.Tensor,
        loss_steps: torch.Tensor,
    ):
        """Calulates the reinforcement loss. Calls self.entropy_regularized_loss

        Args:
            discounted_rewards (torch.Tensor): the discounted rewards over all hops(steps).
            baseline (torch.Tensor): the baseline of the reward.
            logits_steps (torch.Tensor): the logits over all hops(steps).
            loss_steps (torch.Tensor): the loss over all hops(steps).
        Returns:
            (torch.Tensor): returns the regularized loss.
        """
        # subtract the cumulative discounted reward with baseline
        final_reward = (
            discounted_rewards - baseline #torch.mean(discounted_rewards)#- baseline
        )  # is positive while reward increases

        # get reward mean, std
        mean_reward = torch.mean(
            final_reward, dim=(0, 1), keepdim=False
        )  # is negative if reward decreases
        std_reward = torch.std(final_reward, dim=(0, 1), keepdim=False)

        # calculate final reward by subtracting the final_reward from the mean_reward and than
        # elementwise division by the std_reward to normalize reward
        final_reward = torch.div(final_reward - mean_reward, std_reward + 1e-6)

        # relate reward to the loss to create a differentialbe function with regard to the input of the network.
        loss = torch.mul(loss_steps, final_reward)

        decaying_beta = self.beta * pow(self.decay_rate, (self.global_step / 200))

        total_loss = torch.mean(loss) - (
            decaying_beta * self.entropy_regularized_loss(logits_steps)
        )

        return total_loss

    def entropy_regularized_loss(self, logits_steps: torch.Tensor):
        """calcualtes the entropy regularized loss.

        Args:
            logits_steps (torch.Tensor): the logits over all hops(steps).

        Returns:
            torch.Tensor: scalar with the entorpy regularized loss.
        """
        entropy_policy = -torch.mean(
            torch.sum(torch.mul(torch.exp(logits_steps), logits_steps), axis=1)
        )  # scalar
        return entropy_policy

