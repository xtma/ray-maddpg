# MADDPG in Ray
Status: developing

MADDPG implementation with [Ray](https://github.com/ray-project/ray). Models are written in [PyTorch](https://github.com/pytorch/pytorch).
Tricks proposed in TD3 are added to the original method.

## Installation
Install the dependencies first.
```
pip install -r requirements.txt
``` 

You need to install the [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs).
```
git clone https://github.com/openai/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .
```

To use the sction space continuous and make it suitable for MADDPG, you need to modify line 29 in `multiagent-particle-envs/multiagent/environment.py` manually (waiting for better way...):
```
        self.discrete_action_space = False  # Continuous Action Space
```

You can run the code with `python main.py --scenario_name simple_push`.

## Reference
- [Original paper of MADDPG](https://arxiv.org/abs/1706.02275) 
- [Paper about Tricks in TD3](https://arxiv.org/abs/1802.09477)
- [Ray Documents](https://ray.readthedocs.io/en/latest/index.html)