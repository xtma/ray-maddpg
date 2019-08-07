from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn

from ray.rllib.models.pytorch.misc import normc_initializer, SlimFC, \
    _get_activation_fn
from ray.rllib.utils.annotations import override

logger = logging.getLogger(__name__)


class MADDPGActor(nn.Module):
    """MADDPG Actor (fully connected network)."""

    def __init__(self, n_obs, n_actions, options):
        nn.Module.__init__(self)
        hiddens = options.get("fcnet_hiddens")
        activation = _get_activation_fn(options.get("fcnet_activation"))
        logger.debug("Constructing MADDPG Actor {} {}".format(hiddens, activation))
        layers = []
        last_layer_size = n_obs
        for size in hiddens:
            layers.append(
                SlimFC(in_size=last_layer_size,
                       out_size=size,
                       initializer=normc_initializer(1.0),
                       activation_fn=activation))
            last_layer_size = size

        self._hidden_layers = nn.Sequential(*layers)

        self._logits = SlimFC(in_size=last_layer_size,
                              out_size=n_actions,
                              initializer=normc_initializer(0.01),
                              activation_fn=None)

    @override(nn.Module)
    def forward(self, input_dict, hidden_state):
        obs = input_dict["obs"]
        features = self._hidden_layers(obs)
        logits = self._logits(features)
        return logits, features, hidden_state


class MADDPGCritic(nn.Module):
    """MADDPG Critic (fully connected network)."""

    def __init__(self, n_global_obs, n_actions, n_other_actions, options):
        nn.Module.__init__(self)
        hiddens = options.get("fcnet_hiddens")
        activation = _get_activation_fn(options.get("fcnet_activation"))
        logger.debug("Constructing MADDPG Critic {} {}".format(hiddens, activation))
        layers = []
        last_layer_size = n_global_obs + n_actions + n_other_actions
        for size in hiddens:
            layers.append(
                SlimFC(in_size=last_layer_size,
                       out_size=size,
                       initializer=normc_initializer(1.0),
                       activation_fn=activation))
            last_layer_size = size

        self._hidden_layers = nn.Sequential(*layers)

        self._value = SlimFC(in_size=last_layer_size,
                             out_size=1,
                             initializer=normc_initializer(0.01),
                             activation_fn=None)

    @override(nn.Module)
    def forward(self, input_dict, hidden_state):
        global_obs = input_dict["global_obs"]
        actions = input_dict["actions"]
        other_actions = input_dict["other_actions"]
        features = self._hidden_layers(torch.cat([global_obs, actions, other_actions], dim=1))
        value = self._value(features)
        return value, features, hidden_state
