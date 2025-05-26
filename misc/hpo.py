#!/usr/bin/env python3
import argparse
import toml
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
from src.environment import Environment
from src.policy_network import PolicyNetwork
from src.training import Trainer

backend = JournalFileBackend("./hpo_journal.log")
storage = JournalStorage(backend)

def load_config(path):
    return toml.load(path)


def run_training(config):
    # instantiate environment
    env = Environment(
        knowledge_graph_name=config["environment"]["knowledge_graph_name"],
        inverse_relations=config["environment"]["inverse_relations"],
        relation_types=config["environment"]["relation_types"],
        max_degree=config["environment"]["max_degree"],
        prune=config["environment"]["prune"],
        gamma=config["reward"]["gamma"],
        lambda_factor=config["reward"]["lambda_factor"],
        interactive=config["interaction"]["interactive"],
        use_entities=config["interaction"]["use_entities"],
        top_k=config["interaction"]["top_k"],
        feedback_engine=config["interaction"]["feedback_engine"],
        error_rate=config["interaction"]["error_rate"],
        localization=config["evaluation"]["use_localization"],
        roi_dir=config["evaluation"]["roi_dir"],
        embedding_size=config["interaction"]["embedding_size"],
        mlp_size=config["interaction"]["mlp_size"],
        dropout=config["interaction"]["dropout"],
        learning_rate=config["interaction"]["learning_rate"],
        goal_loss_impact=config["interaction"]["goal_loss_impact"],
        batch_size=config["interaction"]["batch_size"]
    )

    # instantiate policy network
    net = PolicyNetwork(
        hidden_size=config["policy_network"]["hidden_size"],
        embedding_size=config["policy_network"]["embedding_size"],
        mlp_size=config["policy_network"]["mlp_size"],
        use_entities=config["policy_network"]["use_entites"],
        dropout=config["policy_network"]["dropout"],
        num_node_embeddings=env.num_nodes,
        padding_node_key=env.padding_node_key,
        num_relation_embeddings=env.num_relations,
        padding_relation_key=env.padding_relation_key,
    )

    # instantiate trainer
    trainer = Trainer(
        batch_size=config["training"]["batch_size"],
        num_epochs=config["training"]["num_epochs"],
        num_hops=config["training"]["num_hops"],
        rollouts_train=config["training"]["num_rollouts_training"],
        rollouts_test=config["evaluation"]["num_rollouts_test"],
        action_dropout_rate=config["training"]["action_dropout_rate"],
        beta=config["training"]["beta"],
        decay_rate=config["training"]["decay_rate"],
        learning_rate=config["training"]["learning_rate"],
        pretraining_epochs=config["interaction"]["pretraining_epochs"],
        patients=config["interaction"]["patients"],
        fit_reward_epochs=config["interaction"]["fit_reward_epochs"],
    )

    # define evaluation epochs
    eval_epochs = range(
        0,
        config["training"]["num_epochs"],
        config["evaluation"]["evaluation_frequency"],
    )

    # run training + evaluation
    try:
        _, average_rewards, losses, hits, metrics = trainer(
            model=net, environment=env, eval_epochs=eval_epochs
        )
    except Exception as error:
        print(error)
        return 0, dict()
        

    # build a DataFrame of test metrics and compute max MRR
    df_test = pd.DataFrame.from_records(list(metrics.values()))
    return df_test["roi_mrr"].max(), metrics


def objective(trial, base_config):
    # deep copy base config so each trial starts fresh
    config = toml.loads(toml.dumps(base_config))

    # === define your search space below ===

    config["environment"]["inverse_relations"] = trial.suggest_categorical(
        "inverse_relations", [False, True]
    )

    config["reward"]["gamma"] = trial.suggest_float("gamma", 0.90, 0.999)
    config["reward"]["lambda_factor"] = trial.suggest_float(
        "lambda_factor", 0.02, 0.4
    )

    config["interaction"]["use_entities"] = trial.suggest_categorical(
        "interaction_use_entities", [False, True]
    )
    config["interaction"]["top_k"] = trial.suggest_int("top_k", 20, 500)
    config["interaction"]["pretraining_epochs"] = trial.suggest_int("pretraining_epochs", 0, 200)
    config["interaction"]["patients"] = trial.suggest_int("patients", 1, 10)

    config["interaction"]["fit_reward_epochs"] = trial.suggest_categorical(
        "fit_reward_epochs",
        [
            "zero",
            "zero-five-ten",
            "zero-five-ten-fifteen",
            "zero-to-four",
            "five-steps",
            "ten-steps",
            "exponential"
        ],
    )
    config["interaction"]["embedding_size"] = trial.suggest_categorical(
        "interaction_embedding_size", [16, 32, 64, 128]
    )
    config["interaction"]["mlp_size"] = trial.suggest_categorical(
        "interaction_mlp_size", [32, 64, 128, 256]
    )
    config["interaction"]["dropout"] = trial.suggest_float(
        "interaction_dropout", 0.0, 0.5
    )
    config["interaction"]["learning_rate"] = trial.suggest_float(
        "interaction_lr", 1e-5, 1e-1, log=True
    )
    config["interaction"]["goal_loss_impact"] = trial.suggest_float(
        "goal_loss_impact", 0.0, 1.0
    )
    config["interaction"]["batch_size"]  = trial.suggest_categorical(
        "interaction_batch_size", [32, 64, 128, 256]
    )

    config["policy_network"]["hidden_size"] = trial.suggest_categorical(
        "hidden_size", [32, 64, 128]
    )
    config["policy_network"]["embedding_size"] = trial.suggest_categorical(
        "policy_embedding_size", [16, 32, 64, 128]
    )
    config["policy_network"]["mlp_size"] = trial.suggest_categorical(
        "policy_mlp_size", [32, 64, 128]
    )
    config["policy_network"]["use_entities"] = trial.suggest_categorical(
        "policy_use_entities", [False, True]
    )
    config["policy_network"]["dropout"] = trial.suggest_float(
        "policy_dropout", 0.0, 0.5
    )

    config["training"]["num_epochs"] = trial.suggest_int("num_epochs", 5, 200) + config["interaction"]["pretraining_epochs"] + config["interaction"]["patients"] 
    config["training"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256]
    )
    config["training"]["action_dropout_rate"] = trial.suggest_float(
        "action_dropout_rate", 0.0, 0.5
    )
    config["training"]["num_rollouts_training"] = trial.suggest_categorical(
        "num_rollouts_training", [20, 40]
    )
    config["training"]["beta"] = trial.suggest_float("beta", 0.0, 1.0)
    config["training"]["decay_rate"] = trial.suggest_float("decay_rate", 0.5, 1.0)
    config["training"]["learning_rate"] = trial.suggest_float(
        "training_lr", 1e-5, 1e-1, log=True
    )
    # ========================================

    # run training and fetch max MRR
    max_mrr, metrics = run_training(config)

    # report intermediate MRRs for pruning
    for step, rec in enumerate(metrics.values()):
        trial.report(rec["roi_mrr"], step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return max_mrr


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for your KG agent")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base TOML config (e.g. config/Countries_S2.toml)",
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="kg_hpo",
        help="Optuna study name (for RDB storage or reuse)",
    )

    args = parser.parse_args()

    # load base config
    base_config = load_config(args.config)


    study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True),
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(),
    )

    # optimize
    study.optimize(lambda t: objective(t, base_config), n_trials=args.trials)

    # report best
    print("Best ROI MRR: {:.4f}".format(study.best_value))
    print("Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
