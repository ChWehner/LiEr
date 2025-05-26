from src.policy_network import PolicyNetwork
from src.environment import Environment
from src.training import Trainer
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import toml
import sys
import os
from src.logger import Logger

logger = Logger()

if __name__ == "__main__":
    all_kgs = os.listdir('config')
    try:
        config = toml.load(f"config/{sys.argv[1]}.toml")
    except IOError:
        print(
            f"'{sys.argv[1]}' is not available. Please choose one of the following knowledge graphs: {all_kgs}"
        )
        exit()

    mrr = []
    hit1 = []
    hit3 = []
    hit5 = []
    loc1 = []
    loc3 = []
    loc5 = []

    for run in range(config["general"]["number_of_runs"]):
        environment = Environment(
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

        network = PolicyNetwork(
            hidden_size=config["policy_network"]["hidden_size"],
            embedding_size=config["policy_network"]["embedding_size"],
            mlp_size=config["policy_network"]["mlp_size"],
            use_entities=config["policy_network"]["use_entites"],
            dropout=config["policy_network"]["dropout"],
            num_node_embeddings = environment.num_nodes,
            padding_node_key=environment.padding_node_key,
            num_relation_embeddings = environment.num_relations,
            padding_relation_key=environment.padding_relation_key,
        )

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

        eval_epochs = range(
            0,
            config["training"]["num_epochs"],
            config["evaluation"]["evaluation_frequency"],
        )

        network, average_rewards, losses, hits, metrics = trainer(
            model=network, environment=environment, eval_epochs=eval_epochs
        )
        logger().debug(f"Metrics: {metrics}")

        # plot and save metrics
        df_train = pd.DataFrame()
        df_train["average_rewards"] = average_rewards
        df_train["losses"] = losses
        df_train["hits"] = hits

        df_test = pd.DataFrame.from_records(list(metrics.values()))

        fig, axes = plt.subplots(14)

        df_test["hit@1"].plot(ax=axes[0], legend=True)  # title="hit@1")
        df_test["hit@3"].plot(ax=axes[1], legend=True)  # title="hit@3")
        df_test["hit@5"].plot(ax=axes[2], legend=True)  # title="hit@5")
        df_test["hit@10"].plot(ax=axes[3], legend=True)  # title="hit@10")
        df_test["hit@20"].plot(ax=axes[4], legend=True)  # title="hit@20")
        df_test["loc@1"].plot(ax=axes[5], legend=True)  # title="hit@1")
        df_test["loc@3"].plot(ax=axes[6], legend=True)  # title="hit@3")
        df_test["loc@5"].plot(ax=axes[7], legend=True)  # title="hit@5")
        df_test["loc@10"].plot(ax=axes[8], legend=True)  # title="hit@10")
        df_test["loc@20"].plot(ax=axes[9], legend=True)  # title="hit@20")
        df_test["mrr"].plot(ax=axes[10], legend=True)  # title="mrr")
        df_train["average_rewards"].plot(
            ax=axes[11], legend=True
        )  # title="average rewards")
        df_train["losses"].plot(
            ax=axes[12], legend=True
        )  # title="average training loss")
        df_train["hits"].plot(ax=axes[13], legend=True)  # title="average training loss")
        # df_train.plot(subplots=True)

        if not os.path.exists(f'results/{config["environment"]["knowledge_graph_name"]}/{config["general"]["prefix"]}'):
            os.makedirs(
                f'results/{config["environment"]["knowledge_graph_name"]}/{config["general"]["prefix"]}'
            )

        plt.savefig(
            f'results/{config["environment"]["knowledge_graph_name"]}/{config["general"]["prefix"]}/{run}.png'
        )
        df_test.to_json(
            f'results/{config["environment"]["knowledge_graph_name"]}/{config["general"]["prefix"]}/{run}_test.json'
        )
        df_train.to_json(
            f'results/{config["environment"]["knowledge_graph_name"]}/{config["general"]["prefix"]}/{run}_train.json'
        )
        mrr.append(df_test["mrr"].max())
        hit1.append(df_test["hit@1"].max())
        hit3.append(df_test["hit@3"].max())
        hit5.append(df_test["hit@5"].max())
        loc1.append(df_test["hit@1"].max())
        loc3.append(df_test["loc@3"].max())
        loc5.append(df_test["loc@5"].max())

    np_mrr = np.array(mrr)
    np_hit1 = np.array(hit1)
    np_hit3 = np.array(hit3)
    np_hit5 = np.array(hit5)
    np_loc1 = np.array(loc1)
    np_loc3 = np.array(loc3)
    np_loc5 = np.array(loc5)

    results = []
    results.append({"mrr": {"mean": np.mean(np_mrr), "var": np.var(np_mrr)}})
    results.append({"hit1": {"mean": np.mean(np_hit1), "var": np.var(np_hit1)}})
    results.append({"hit3": {"mean": np.mean(np_hit3), "var": np.var(np_hit3)}})
    results.append({"hit5": {"mean": np.mean(np_hit5), "var": np.var(np_hit5)}})
    results.append({"loc1": {"mean": np.mean(np_loc1), "var": np.var(np_loc1)}})
    results.append({"loc3": {"mean": np.mean(np_loc3), "var": np.var(np_loc3)}})
    results.append({"loc5": {"mean": np.mean(np_loc5), "var": np.var(np_loc5)}})

    with open(
        f'results/{config["environment"]["knowledge_graph_name"]}/{config["general"]["prefix"]}/summary.txt',
        "a",
    ) as output:
        output.write(str(mrr))
        output.write(str(hit1))
        output.write(str(hit3))
        output.write(str(hit5))
        output.write(str(loc1))
        output.write(str(loc3))
        output.write(str(loc5))
        output.write(str(results))
