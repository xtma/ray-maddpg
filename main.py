import argparse

import ray
from config import MADDPG_CONFIG
from env import SimpleMultiAgentEnv
from maddpg_policy import MADDPGTorchPolicy
from ray import tune
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from ray.rllib.optimizers import SyncBatchReplayOptimizer
from ray.tune import register_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MADDPG with Ray')
    parser.add_argument('--scenario_name',
                        type=str,
                        choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_tag'],
                        default='simple_push',
                        help="Scenario name of the multiagent-particle-envs")

    args = parser.parse_args()

    # Define the trainer
    def make_sync_batch_optimizer(workers, config):
        return SyncBatchReplayOptimizer(workers,
                                        learning_starts=config["learning_starts"],
                                        buffer_size=config["buffer_size"],
                                        train_batch_size=config["train_batch_size"])

    MADDPGTrainer = GenericOffPolicyTrainer.with_updates(name="MADDPG",
                                                         default_config=MADDPG_CONFIG,
                                                         default_policy=MADDPGTorchPolicy,
                                                         make_policy_optimizer=make_sync_batch_optimizer)

    # Registry Environment
    register_env("simple_multiagent", lambda config: SimpleMultiAgentEnv(config))
    single_env = SimpleMultiAgentEnv(env_config={"scenario_name": args.scenario_name})

    # Policy Mapping
    policies = {
        agent: (None, single_env.observation_space[agent], single_env.action_space[agent], {
            "observation_spaces": single_env.observation_space,
            "action_spaces": single_env.action_space,
            "agent_id": agent
        }) for agent in single_env.agent_ids
    }

    # Start training
    ray.init()
    tune.run(
        MADDPGTrainer,
        stop={
            "timesteps_total": 1000000,
        },
        config={
            "env": "simple_multiagent",
            "env_config": {
                "scenario_name": args.scenario_name,
                "time_limit": 100
            },
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": tune.function(lambda agent_id: agent_id),
            },
            #  "observation_filter": "NoFilter",
        })
