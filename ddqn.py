import torch
import torch.nn as nn
import numpy as np
import variable as v
import utils
import random
from collections import namedtuple, deque
from copy import deepcopy
from PIL import Image


class DDQN:
    def __init__(self, ddqn_init):
        torch.manual_seed(ddqn_init['seed'])
        self.policy_type = ddqn_init["policy_type"]
        self.temperature = ddqn_init["temperature"]
        self.num_action = ddqn_init["num_action"]
        self.epsilon = ExplorationRateDecay(**ddqn_init["exploration_rate"])
        self.discount_factor = ddqn_init["discount_factor"]
        self.update_target_rate = ddqn_init["update_target_rate"]
        self.update_after = ddqn_init["update_after"]
        self.update_every = ddqn_init["update_every"]
        self.update_count = 0.0
        self.hard_update_target_every = ddqn_init["hard_update_target_every"]

        self.random_generator = np.random.RandomState(seed=ddqn_init['seed'])
        self.primary_q_network = QNetwork(ddqn_init['q_network']).to(v.device)
        self.target_q_network = QNetwork(ddqn_init['q_network']).to(v.device)
        self.replay_buffer = ReplayBuffer(ddqn_init["buffer"])

        self.state = None
        self.action = None

        self.init_optimizers(q_network_optimizer=ddqn_init['q_network']['optimizer'])
        self.hard_update_target_weights()

    def init_optimizers(self, q_network_optimizer={}):
        self.primary_q_network.init_optimizer(q_network_optimizer)

    def hard_update_target_weights(self):
        self.target_q_network.load_state_dict(self.primary_q_network.state_dict())

    def soft_update_target_weights(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.primary_q_network.parameters()):
            target_param.data.copy_(target_param.data * self.update_target_rate +
                                    param.data * (1.0 - self.update_target_rate))

    def set_test(self):
        self.primary_q_network.eval()

    def set_train(self):
        self.primary_q_network.train()

    @staticmethod
    def preprocess_image(state_image, im_size=84):
        state_image = state_image[34:194]
        state_image = np.round(np.dot(state_image, [0.2989, 0.587, 0.114])).astype(np.float)
        state_image = np.array(Image.fromarray(state_image).resize((im_size, im_size)))
        return np.expand_dims(state_image, axis=0).astype(np.float)

    def policy(self, state):

        state_tensor = utils.to_tensor(state).view((1, ) + state.shape)
        with torch.no_grad():
            values = self.primary_q_network.predict(state_tensor).squeeze().cpu().numpy()

        if self.policy_type == 'e-greedy':
            action = self.e_greedy(values)
        elif self.policy_type == 'softmax':
            action = self.softmax(values)
        else:
            raise ValueError(f"Agent does not handle {self.policy_type} policy")

        return action

    def softmax(self, values):
        values = values

        values_temp = values / self.temperature
        exp_values_temp = np.exp(values_temp - np.max(values_temp, axis=0))
        softmax_values = exp_values_temp / np.sum(exp_values_temp)

        action = self.random_generator.choice(self.num_action, p=softmax_values)

        return action

    def e_greedy(self, values):

        if self.random_generator.rand() < self.epsilon():
            action = self.random_generator.randint(self.num_action)
        else:
            action = np.argmax(values)

        return action

    def episode_init(self, state):
        self.state = state
        action = self.policy(state)
        self.action = action

        return action

    def update(self, next_state, reward, done):
        self.replay_buffer.append(deepcopy(self.state), deepcopy(self.action), reward, next_state, done)

        if len(self.replay_buffer) > self.update_after and self.update_count % self.update_every == 0.0:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()

            next_actions = torch.argmax(self.target_q_network.predict(next_states), axis=1).unsqueeze(-1)
            next_actions_values = self.primary_q_network.action_values_predict(next_states, next_actions)

            expected_q_values = rewards + self.discount_factor * (1 - dones) * next_actions_values
            self.primary_q_network.update(expected_q_values, states, actions)
            self.soft_update_target_weights()

            if self.update_count % self.hard_update_target_every == 0.0:
                self.hard_update_target_weights()

        self.update_count += 1.0

        next_action = self.policy(next_state)
        self.action = next_action
        self.state = next_state
        return next_action


class QNetwork(torch.nn.Module):
    def __init__(self, q_network):
        super(QNetwork, self).__init__()
        net = q_network["network_init"]
        self.optimizer = None
        self.loss = torch.nn.MSELoss()
        self.loss_history = list()
        self.input_image = q_network["input_image"]

        if self.input_image:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_channels=net["cl1_channels"],
                          out_channels=net["cl2_channels"],
                          kernel_size=net["cl1_kernel_size"],
                          stride=net["cl1_stride"],
                          padding=net["cl1_padding"]
                          ),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=net["cl2_channels"],
                          out_channels=net["out_channels"],
                          kernel_size=net["cl2_kernel_size"],
                          stride=net["cl2_stride"],
                          padding=net["cl2_padding"]
                          ),
                nn.ReLU()
            )

        self.relu = nn.ReLU()
        self.l1 = nn.Linear(net["l1_shape"], net["l2_shape"])
        self.l2 = nn.Linear(net["l2_shape"], net["l3_shape"])
        self.o = nn.Linear(net["l3_shape"], net["o_shape"])

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):
        if self.input_image:
            x = self.conv_layer(x)
            x = x.view(x.shape[0], -1)

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.o(x)

        return x

    def predict(self, states):
        return self.forward(states)

    def action_values_predict(self, states, actions):
        states_values = self.forward(states)
        return torch.gather(input=states_values, dim=1, index=actions.long()).squeeze()

    def update(self, expected_q_values, states, actions):
        current_q_values = self.action_values_predict(states, actions)
        loss = self.loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class ReplayBuffer:
    def __init__(self, replay_buffer_init, seed=42):
        random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.max_len = replay_buffer_init["max_len"]
        self.batch_size = replay_buffer_init["batch_size"]
        self.buffer = deque(maxlen=self.max_len)

    def append(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = utils.to_tensor(np.vstack([e.state for e in experiences if e is not None]))
        actions = utils.to_tensor(np.vstack([e.action for e in experiences if e is not None]))
        rewards = utils.to_tensor(np.vstack([e.reward for e in experiences if e is not None])).squeeze()
        next_states = utils.to_tensor(np.vstack([e.next_state for e in experiences if e is not None]))
        dones = utils.to_tensor(np.vstack([e.done for e in experiences if e is not None])).squeeze()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class ExplorationRateDecay:
    def __init__(self, rate, max_rate, min_rate, decay_rate, is_constant):
        self.rate = rate
        self.max_er = max_rate
        self.min_er = min_rate
        self.decay_er = decay_rate
        self.episode_count = 0.0
        self.is_constant = is_constant
        self.history = list()

    def __call__(self):
        if self.is_constant:
            self.history.append(self.rate)
            return self.rate

        self.rate = self.min_er + ((self.max_er - self.min_er) * (np.exp(-self.decay_er * self.episode_count)))
        self.episode_count += 1.0

        self.history.append(self.rate)

        return self.rate

    def reset(self):
        self.episode_count = 0.0
