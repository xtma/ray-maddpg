from __future__ import absolute_import, division, print_function

import logging
import os
from threading import Lock

import numpy as np

import torch
import torch.nn.functional as F
from model import MADDPGActor, MADDPGCritic
from ray.rllib.policy.policy import LEARNER_STATS_KEY, Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tracking_dict import UsageTrackingDict

logger = logging.getLogger(__name__)


class MADDPGTorchPolicy(Policy):
    """Template for a PyTorch policy and loss to use with RLlib.

    This is similar to TFPolicy, but for PyTorch.

    Attributes:
        observation_space (gym.Space): observation space of the policy.
        action_space (gym.Space): action space of the policy.
        lock (Lock): Lock that must be held around PyTorch ops on this graph.
            This is necessary when using the async sampler.
    """

    def __init__(self, observation_space, action_space, config):
        """Build a policy from policy and loss torch modules.

        Note that model will be placed on GPU device if CUDA_VISIBLE_DEVICES
        is set. Only single GPU is supported for now.

        Arguments:
            observation_space (gym.Space): observation space of the policy.
            action_space (gym.Space): action space of the policy.
            model (nn.Module): PyTorch policy module. Given observations as
                input, this module must return a list of outputs where the
                first item is action logits, and the rest can be any value.
            loss (func): Function that takes (policy, batch_tensors)
                and returns a single scalar loss.
            action_distribution_cls (ActionDistribution): Class for action
                distribution.
        """
        # Spaces Specification
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_obs = np.product(observation_space.shape)
        self.n_actions = np.product(action_space.shape)
        self.n_global_obs = np.sum(
            [np.product(space.shape) for agent, space in config.get("observation_spaces").spaces.items()])
        self.n_other_actions = np.sum([
            np.product(space.shape)
            for agent, space in config.get("action_spaces").spaces.items()
            if not agent == config.get("agent_id")
        ])  # sum of others action

        # PyTorch Models
        self.lock = Lock()
        self.device = (torch.device("cuda")
                       if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None)) else torch.device("cpu"))
        self._actor, self._target_actor = [
            MADDPGActor(
                self.n_obs,
                self.n_actions,
                options=config["actor_model"],
            ).to(self.device) for _ in range(2)
        ]
        self._critic_1, self._target_critic_1, self._critic_2, self._target_critic_2 = [
            MADDPGCritic(
                self.n_global_obs,
                self.n_actions,
                self.n_other_actions,
                options=config["critic_model"],
            ).to(self.device) for _ in range(4)
        ]

        # Hyperparameters
        self.cur_noise_scale = 1.0
        self.target_noise_scale = config.get("target_noise_scale")
        self.tau = config.get("tau")
        self.gamma = config.get("gamma")
        self.actor_lr = config.get("actor_lr")
        self.critic_lr = config.get("critic_lr")

        self._training_step = 0

        self._actor_optimizer = torch.optim.Adam(params=self._actor.parameters(), lr=self.actor_lr)
        self._critic_1_optimizer = torch.optim.Adam(params=self._critic_1.parameters(), lr=self.critic_lr)
        self._critic_2_optimizer = torch.optim.Adam(params=self._critic_2.parameters(), lr=self.critic_lr)

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        with self.lock:
            with torch.no_grad():
                input_dict = self._lazy_tensor_dict({
                    "obs": obs_batch,
                })
                if prev_action_batch:
                    input_dict["prev_actions"] = prev_action_batch
                if prev_reward_batch:
                    input_dict["prev_rewards"] = prev_reward_batch
                actions_mean, _, _ = self._actor(input_dict, state_batches)
                actions = torch.normal(actions_mean, self.cur_noise_scale)
                return (actions.cpu().numpy(), [], {})

    @override(Policy)
    def learn_on_batch(self, postprocessed_batch):
        batch_tensors = self._lazy_tensor_dict(postprocessed_batch)

        with self.lock:
            obs = batch_tensors["obs"]
            global_obs = batch_tensors["global_obs"]
            next_obs = batch_tensors["new_obs"]
            next_global_obs = batch_tensors["global_new_obs"]
            actions = batch_tensors["actions"]
            other_actions = batch_tensors["other_actions"]
            next_other_actions = batch_tensors["other_new_actions"]
            terminated = batch_tensors["dones"].view(-1, 1)
            rewards = batch_tensors["rewards"].view(-1, 1)

            with torch.no_grad():
                # Target Policy Smoothing
                next_actions_mean, _, _ = self._target_actor({"obs": next_obs}, [])
                next_actions = torch.normal(next_actions_mean, self.target_noise_scale)
                next_input_dict = {
                    "global_obs": next_global_obs,
                    "actions": next_actions,
                    "other_actions": next_other_actions
                }
                next_q_1, _, _ = self._target_critic_1(next_input_dict, [])
                next_q_2, _, _ = self._target_critic_2(next_input_dict, [])

            # Clipped Double-Q Learning
            target_q = rewards + self.gamma * (1 - terminated) * torch.min(next_q_1, next_q_2)
            cur_input_dict = {"global_obs": global_obs, "actions": actions, "other_actions": other_actions}
            cur_q_1, _, _ = self._critic_1(cur_input_dict, [])
            cur_q_2, _, _ = self._critic_2(cur_input_dict, [])
            value_loss_1 = F.smooth_l1_loss(cur_q_1, target_q)
            value_loss_2 = F.smooth_l1_loss(cur_q_2, target_q)

            actions_mean, _, _ = self._actor({"obs": obs}, [])
            expected_q, _, _ = self._critic_1(
                {
                    "global_obs": global_obs,
                    "actions": actions_mean,
                    "other_actions": other_actions
                }, [])
            action_loss = -expected_q.mean()

            self._critic_1_optimizer.zero_grad()
            value_loss_1.backward()
            self._critic_1_optimizer.step()
            self._critic_2_optimizer.zero_grad()
            value_loss_2.backward()
            self._critic_2_optimizer.step()
            # Delayed Policy Updates
            if self._training_step % 2 == 0:
                self._actor_optimizer.zero_grad()
                action_loss.backward()
                self._actor_optimizer.step()

            stats = {
                "action_loss": action_loss.item(),
                "value_loss_1": value_loss_1.item(),
                "value_loss_2": value_loss_2.item(),
                "cur_noise_scale": self.cur_noise_scale,
            }

            self._training_step += 1

            return {LEARNER_STATS_KEY: stats}

    @override(Policy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        """Implements algorithm-specific trajectory postprocessing.

        This will be called on each trajectory fragment computed during policy
        evaluation. Each fragment is guaranteed to be only from one episode.

        Arguments:
            sample_batch (SampleBatch): batch of experiences for the policy,
                which will contain at most one episode trajectory.
            other_agent_batches (dict): In a multi-agent env, this contains a
                mapping of agent ids to (policy, agent_batch) tuples
                containing the policy and experiences of the other agent.
            episode (MultiAgentEpisode): this provides access to all of the
                internal episode state, which may be useful for model-based or
                multi-agent algorithms.

        Returns:
            SampleBatch: postprocessed sample batch.
        """
        other_obs_batch = np.hstack([batch[1]["obs"] for batch in other_agent_batches.values()])
        other_next_obs_batch = np.hstack([batch[1]["new_obs"] for batch in other_agent_batches.values()])
        global_obs_batch = np.hstack([sample_batch["obs"], other_obs_batch])
        global_next_obs_batch = np.hstack([sample_batch["new_obs"], other_next_obs_batch])
        other_action_batch = np.hstack([batch[1]["actions"] for batch in other_agent_batches.values()])
        other_next_action_batch = np.vstack([other_action_batch[1:, :], np.zeros_like(other_action_batch[:1, :])])
        sample_batch["global_obs"] = global_obs_batch
        sample_batch["global_new_obs"] = global_next_obs_batch
        sample_batch["other_actions"] = other_action_batch
        sample_batch["other_new_actions"] = other_next_action_batch
        return sample_batch

    @override(Policy)
    def get_weights(self):
        with self.lock:
            return {
                "actor": self._actor.state_dict(),
                "target_actor": self._target_actor.state_dict(),
                "critic_1": self._critic_1.state_dict(),
                "target_critic_1": self._target_critic_1.state_dict(),
                "critic_2": self._critic_2.state_dict(),
                "target_critic_2": self._target_critic_2.state_dict(),
            }

    @override(Policy)
    def set_weights(self, weights):
        with self.lock:
            self._actor.load_state_dict(weights.get("actor"))
            self._target_actor.load_state_dict(weights.get("target_actor"))
            self._critic_1.load_state_dict(weights.get("critic_1"))
            self._target_critic_1.load_state_dict(weights.get("target_critic_1"))
            self._critic_2.load_state_dict(weights.get("critic_2"))
            self._target_critic_2.load_state_dict(weights.get("target_critic_2"))

    def set_epsilon(self, epsilon):
        # set_epsilon is called by optimizer to anneal exploration as
        # necessary, and to turn it off during evaluation. The "epsilon" part
        # is a carry-over from DQN, which uses epsilon-greedy exploration
        # rather than adding action noise to the output of a policy network.
        self.cur_noise_scale = epsilon

    def update_target(self):
        for we, wt in zip(self._actor.parameters(), self._target_actor.parameters()):
            wt.data += self.tau * (we.data - wt.data)
        for we, wt in zip(self._critic_1.parameters(), self._target_critic_1.parameters()):
            wt.data += self.tau * (we.data - wt.data)
        for we, wt in zip(self._critic_2.parameters(), self._target_critic_2.parameters()):
            wt.data += self.tau * (we.data - wt.data)
        logger.debug("Updated target networks")

    def _lazy_tensor_dict(self, postprocessed_batch):
        batch_tensors = UsageTrackingDict(postprocessed_batch)

        def convert(arr):
            return torch.as_tensor(np.asarray(arr), dtype=torch.float, device=self.device)

        batch_tensors.set_get_interceptor(convert)
        return batch_tensors
