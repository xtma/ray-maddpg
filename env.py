from gym.spaces import Dict

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv as GymMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv as RayMultiAgentEnv
from collections import OrderedDict


class SimpleMultiAgentEnv(RayMultiAgentEnv):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    def __init__(self, env_config):
        self.config = env_config
        # load scenario from script
        scenario = scenarios.load(self.config.get("scenario_name") + ".py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        if self.config.get("benchmark", False):
            self._env = GymMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                         scenario.benchmark_data)
        else:
            self._env = GymMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        self._agents = self._env.agents
        self.agent_ids = [agent.name.replace(' ', '_') for agent in self._agents]
        self.observation_space = Dict(dict(zip(self.agent_ids, self._env.observation_space)))
        self.action_space = Dict(dict(zip(self.agent_ids, self._env.action_space)))

    def reset(self):
        self.time = 0
        return self._list_to_dict(self._env.reset())

    def step(self, action_dict):
        self.time += 1
        obs_list, rewards_list, dones_list, _ = self._env.step(list(action_dict.values()))
        obs_dict = self._list_to_dict(obs_list)
        rewards_dict = self._list_to_dict(rewards_list)
        dones_dict = self._list_to_dict(dones_list)
        dones_dict["__all__"] = all(dones_list)
        if self.time >= self.config.get("time_limit", 100):
            for k in dones_dict.keys():
                dones_dict[k] = True
        infos = {}
        return obs_dict, rewards_dict, dones_dict, infos

    def _list_to_dict(self, _list):
        assert len(self.agent_ids) == len(_list)
        return OrderedDict(zip(self.agent_ids, _list))


if __name__ == "__main__":
    # simple_adversary, simple_crypto, simple_push, simple_tag
    env = SimpleMultiAgentEnv(env_config={"scenario_name": "simple_adversary"})
    env.reset()
    for i in range(1000):
        s, r, d, info = env.step(env.action_space.sample())
        print(r)
        if d["__all__"]:
            env.reset()
