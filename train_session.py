import torch
import variable as var
import numpy as np
from itertools import product, count
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import os
import utils as ut
import gym
from datetime import datetime
import time


class TrainSession:
    def __init__(self, agents, env, seed):
        env.seed(seed)
        plt.style.use('ggplot')
        self.agents = agents
        self.env = env

        self.rewards_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}

        self.time_steps_per_episode = {agent_name: np.array([]) for agent_name, _ in agents.items()}

        self.line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
        self.num_lines_style = len(self.line_styles)
        self.cm = plt.get_cmap('tab10')
        self.max_diff_colors = 8

    def append_agents(self, agents, overwrite=False):

        assert not any(item in agents for item in self.agents) or overwrite, "You are trying to overwrite agents dictionary"
        agent_names = list(agents.keys())

        self.agents.update(agents)
        self.rewards_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})
        self.time_steps_per_episode.update({agent_name: np.array([]) for agent_name, _ in agents.items()})

        return agent_names

    def pop_agents(self, agents):
        valid_agent_name = set(agents).intersection(self.agents.keys())
        for agent_name in valid_agent_name:
            self.agents.pop(agent_name)

    def parameter_grid_append(self, agent_object, base_agent_init, parameters_dict):

        agents = {}
        parameter_grid = list(dict(zip(parameters_dict, x)) for x in product(*parameters_dict.values()))
        for parameters_dict in parameter_grid:
            agent_init_tmp = deepcopy(base_agent_init)
            agent_name = f"{agent_object.__name__}: "
            for name, value in parameters_dict.items():
                ut.set_in_dict(agent_init_tmp, name, value)
                agent_name += f"{'_'.join(name)}:{value};"

            agents.update({agent_name: agent_object(agent_init_tmp)})
            self.rewards_per_episode.update({agent_name: np.array([])})
            self.time_steps_per_episode.update({agent_name: np.array([])})

        self.agents.update(agents)

        return list(agents.keys())

    def plot_train(self, window=200, agent_subset=None, std=True):

        if not agent_subset:
            agent_subset = self.agents.keys()

        series_to_plot = {'cumulative rewards': {agent_name: self.rewards_per_episode[agent_name] for agent_name in agent_subset},
                          'time steps': {agent_name: self.time_steps_per_episode[agent_name] for agent_name in agent_subset}
                          }

        agents_to_plot = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        loss_per_agents = {'loss': {agent_name: (np.array(agent.primary_q_network.loss_history) if 'primary_q_network'
                                                                                                   in agent.__dict__.keys()
                                                 else np.array([]))
                                    for agent_name, agent
                                    in agents_to_plot.items()}
                           }

        series_to_plot.update(loss_per_agents)

        self.plot(series_to_plot, window=window, std=std)

    def plot(self, series_to_plot, window=200, std=True):

        nb_graph = len(series_to_plot)
        fig, axs = plt.subplots(nb_graph, 1, figsize=(10, 5*nb_graph), facecolor='w', edgecolor='k')
        axs = axs.ravel()

        for idx, (series_name, dict_series) in enumerate(series_to_plot.items()):
            for jdx, (agent_name, series) in enumerate(dict_series.items()):
                if series.size == 0:
                    axs[idx].plot([0.0], [0.0], label=agent_name)
                    continue

                cm_idx = jdx % self.max_diff_colors
                # jdx // self.num_lines_style * float(self.num_lines_style) / self.max_diff_colors (upward)
                ls_idx = min(jdx // self.max_diff_colors, self.num_lines_style)  # jdx % self.num_lines_style

                series_mvg = ut.rolling_window(series, window=window)
                series_mvg_avg = np.mean(series_mvg, axis=1)

                lines = axs[idx].plot(range(len(series_mvg_avg)), series_mvg_avg, label=agent_name)

                lines[0].set_color(self.cm(cm_idx))
                lines[0].set_linestyle(self.line_styles[ls_idx])

                if std:
                    series_mvg_std = np.std(series_mvg, axis=1)
                    area = axs[idx].fill_between(range(len(series_mvg_avg)), series_mvg_avg - series_mvg_std,
                                                 series_mvg_avg + series_mvg_std, alpha=0.15)
                    area.set_color(self.cm(cm_idx))
                    area.set_linestyle(self.line_styles[ls_idx])

            box = axs[idx].get_position()
            axs[idx].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[idx].set_title(f"{series_name} per episode", fontsize=15)
            axs[idx].set_ylabel(f"avg {series_name}", fontsize=10)
            axs[idx].set_xlabel(f"episodes", fontsize=10)
            axs[idx].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.tight_layout()

    def train(self, n_episode=500, t_max_per_episode=200, graphical=False, agent_subset=None):

        if agent_subset:
            agents = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        else:
            agents = self.agents

        for agent_name, agent in agents.items():

            time_steps_per_episode = list()
            rewards_per_episode = list()

            for _ in tqdm(range(n_episode)):

                rewards = 0.0
                state = self.env.reset()
                next_action = agent.episode_init(state)

                for t in count():
                    if graphical:
                        self.env.render()
                    state, reward, done, _ = self.env.step(next_action)
                    next_action = agent.update(state, reward, done)
                    rewards += reward

                    if done or t > t_max_per_episode:
                        break

                time_steps_per_episode.append(t)
                rewards_per_episode.append(rewards)

            self.time_steps_per_episode[agent_name] = np.concatenate([self.time_steps_per_episode[agent_name],
                                                                      np.array(time_steps_per_episode)])
            self.rewards_per_episode[agent_name] = np.concatenate([self.rewards_per_episode[agent_name],
                                                                   np.array(rewards_per_episode)])
        if graphical:
            self.env.close()

    def test(self, n_episode=500, t_max_per_episode=200, graphical=False, agent_subset=None, plot=True,
             window=200, std=True, save_video=False):
        if agent_subset:
            agents = {agent_name: self.agents[agent_name] for agent_name in agent_subset}
        else:
            agents = self.agents

        time_steps_per_episode_per_agent = {}
        rewards_per_episode_per_agent = {}

        for agent_name, agent in agents.items():

            agent.set_test()
            time_steps_per_episode = list()
            rewards_per_episode = list()

            if save_video:
                env = gym.wrappers.Monitor(self.env, os.path.join(var.PATH, 'video_test', f"{agent_name}_{datetime.now()}"),
                                           force=True)
            else:
                env = self.env

            for _ in tqdm(range(n_episode)):

                rewards = 0.0
                state = env.reset()
                next_action = agent.episode_init(state)
                for t in count():

                    if graphical:
                        time.sleep(.05)
                        env.render()

                    state, reward, done, _ = env.step(next_action)
                    next_action = agent.policy(state)
                    rewards += reward

                    if done or t > t_max_per_episode:
                        break

                time_steps_per_episode.append(t)
                rewards_per_episode.append(rewards)

            time_steps_per_episode_per_agent[agent_name] = np.array(time_steps_per_episode)
            rewards_per_episode_per_agent[agent_name] = np.array(rewards_per_episode)
            agent.set_train()

        if graphical:
            self.env.close()
            env.close()

        if plot:
            series_to_plot = {
                'cumulative rewards': rewards_per_episode_per_agent,
                'time steps': time_steps_per_episode_per_agent
            }
            self.plot(series_to_plot, window=window, std=std)

    def save_model(self, suffix=''):
        model_dir = os.path.join(var.PATH, 'saved_model/')
        for agent_name, agent in self.agents.items():
            torch.save(agent.primary_q_network.state_dict(),
                       os.path.join(model_dir, f"{agent_name}_q_network_{suffix}.pth"))

    def load_model(self, agent_name, suffix=''):
        model_dir = os.path.join(var.PATH, 'saved_model/')
        self.agents[agent_name].primary_q_network.load_state_dict(
            torch.load(os.path.join(model_dir, f"{agent_name}_q_network_{suffix}.pth"))
        )
        self.agents[agent_name].target_q_network.load_state_dict(
            torch.load(os.path.join(model_dir, f"{agent_name}_q_network_{suffix}.pth"))
        )
