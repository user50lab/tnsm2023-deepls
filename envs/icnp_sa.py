import os
from typing import Tuple

import gym
import gym.spaces
import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors

from envs.utils import johnson_all_sp

# 定义python字典，用于初始化图的边的默认属性集合。
DEFAULT_EDGE_ATTRIBUTES = {
    "increments": 1,
    "reductions": 1,
    "weight": 1,
    "traffic": 0.0,
}


class ICNP2021Env(gym.Env):
    ### __init__: 构造函数，用于初始化环境。
    def __init__(
        self,
        env_type: str = "NSFNet",
        episode_length: int = 10,
        traffic_profile: str = "uniform",
        weight_change: int = 1,
        sim_annealing: bool = False,
        starting_temperature: float = 100,
        end_temperature: float = 10,
        base_dir: str = "./MARL-GNN-TE/datasets",
        seed: int = 42,
        eval: bool = False,
    ) -> None:
        self.env_type = env_type
        self.episode_length = episode_length
        self.traffic_profile = traffic_profile
        self.weight_change = weight_change
        self.sim_annealing = sim_annealing
        self.starting_temperature = starting_temperature
        self.end_temperature = end_temperature
        self.base_dir = base_dir
        self.dataset_dir = os.path.join(base_dir, env_type, traffic_profile)
        self.eval = eval

        self.G, self.link_id_dict = self._load_topology()

        self.test_index = 99

        # gym.spaces.Discrete(n)创建了一个离散的动作空间，这里的n表示可能的动作的数量。在离散动作空间中，每个动作都被分配一个唯一的整数编号，从0到n-1。
        # self.G.number_of_edges()是图G中边的数量。所以，动作空间的大小被设定为图中边的数量。
        # 这意味着在这个环境中，每个动作对应于图中的一条边。智能体选择的动作将影响到对应边的权重或其他属性。
        self.action_space = gym.spaces.Discrete(self.G.number_of_edges())
        # gym.spaces.Box创建了一个连续的多维空间，其中每个观测都是一个在指定范围内的实数向量。
        # low和high参数定义了这个空间的上下限。在这里，它们被设置为0和1，表示观测空间中每个元素的最小和最大值。因为后面使用了归一化，所以这个范围是合适的。
        # shape=(self.G.number_of_edges(), 2)定义了观测空间的形状，这里是一个二维数组。第一个维度的大小是图中边的数量，第二个维度的大小是2，这表示每条边有两种观测数据。
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.G.number_of_edges(), 2), dtype=np.float32)

        self.seed(seed)

        self.reset()

    ### step: 执行一个行动(action)，返回新的状态观测(observation)，奖励(reward)，布尔值表示是否终止(done)，以及调试信息(info)。
    # 实现环境中的一次动作和状态转换的核心。
    # 定义函数step的行，它接受一次动作action作为输入，并返回一个包含四个元素的元组(Tuple)：一个NumPy数组(npt.NDArray)表示新的观测值，一个浮点数(float)表示奖励，一个布尔值(bool)表示是否结束这一回合（或称为episode），以及一个字典(dict)包含额外的调试信息。
    def step(self, action) -> Tuple[npt.NDArray, float, bool, dict]:
        # for idl, value in enumerate(action):
        # 将传入的action转换为整数，并使用这个整数作为self.link_id_dict字典的键来得到对应的链接。此处假定动作(action)代表着一个序号或ID，该序号或ID将被映射到一个特定的网络链接上。
        link = self.link_id_dict[int(action)]
        # 根据所选的动作(即特定的链路)更新环境中的权重。
        self._update_weights(link)
        # 获取更新后的权重信息。
        self._get_weights()
        # 可能用于收集每个链接的流量信息。
        self._get_link_traffic()
        # 计算并获取当前动作后的即时奖励。
        reward = self.reward()
        # 获取当前环境状态的观测值，它的具体实现会提供智能体需要的状态信息。
        observation = self.observation()
        self.iter_count += 1
        # 对self.temperature变量进行更新，这里用的是模拟退火算法的通用更新公式。这一行减少了self.temperature的值，模拟退火过程中的温度下降。退火系数是基于end_temperature, starting_temperature, 和episode_length计算得来。
        self.temperature -= self.temperature * np.power(self.end_temperature / self.starting_temperature, self.episode_length)
        # 创建一个字典info，包含了一些可能用于调试和分析的额外信息，如starting_max_load, max_load和n_accepted等。这些都是与环境的性能指标相关的值。
        info = {
            "starting_max_load": self.start_reward_measure,
            "max_load": self.best_reward_measure,
            "n_accepted": self.n_accepted,
        }
        # 返回一个元组，包括新的观测值(observation)、计算得到的奖励(reward)、一个布尔值表示当前回合是否应该结束（当iter_count大于等于episode_length时表明结束）、以及包含额外调试信息的字典(info)。
        return observation, reward, self.iter_count >= self.episode_length, info

    ### reset: 重置环境的状态到初始状态，并返回初始观测。
    def reset(self) -> npt.NDArray:
        if self.eval:
            self.num_sample = self.test_index
        else:
            self.num_sample = self.rng.integers(low=0, high=99)
        self.test_index += 1
        if self.test_index >= 200:
            self.test_index = 100
        self._load_capacities()
        self._load_traffic_matrix()
        self._reset_edge_attributes()
        self._get_weights()
        self._get_link_traffic()
        self.reward_measure = np.max(self.link_traffic)
        self.best_reward_measure = self.reward_measure
        self.start_reward_measure = self.reward_measure
        self.best_weights = np.copy(self.weights)
        self.iter_count = 0
        self.temperature = self.starting_temperature
        self.n_accepted = 0
        return self.observation()

    def observation(self) -> npt.NDArray:
        # best_reward_measure = np.full(shape=(self.G.number_of_edges(), 1), fill_value=self.best_reward_measure)
        return np.concatenate([self.link_traffic.reshape(-1, 1), self.weights.reshape(-1, 1)], axis=-1, dtype=np.float32)

    def reward(self) -> float:
        current_reward_measure = np.max(self.link_traffic)
        reward = self.reward_measure - current_reward_measure
        # reward = max(0, (self.best_reward_measure - current_reward_measure))
        if self.best_reward_measure > current_reward_measure:
            self.best_reward_measure = current_reward_measure
            self.best_weights = np.copy(self.weights)
            self.n_accepted += 1
        elif current_reward_measure > self.reward_measure:
            if self.rng.random() > self._metropolis_acceptance(current_reward_measure):
                self.weights = self.best_weights
                self._set_weights()
                self._get_link_traffic()
                current_reward_measure = np.max(self.link_traffic)
            else:
                self.n_accepted += 1
        else:
            self.n_accepted += 1
        self.reward_measure = current_reward_measure
        return reward

    def seed(self, seed: int) -> None:
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.rng = np.random.default_rng(seed)

    def _load_topology(self) -> Tuple[nx.DiGraph, dict]:
        try:
            nx_file = os.path.join(self.base_dir, self.env_type, "graph_attr.txt")
            topology = nx.DiGraph(nx.read_gml(nx_file, destringizer=int))
        except:
            topology = nx.DiGraph()
            capacity_file = os.path.join(self.dataset_dir, "capacities", "graph.txt")
            with open(capacity_file) as fd:
                for line in fd:
                    if "Link_" in line:
                        camps = line.split(" ")
                        topology.add_edge(int(camps[1]), int(camps[2]))
                        topology[int(camps[1])][int(camps[2])]["bandwidth"] = int(camps[4])
        link_id_dict = {}
        idx = 0
        for i, j in topology.edges():
            topology[i][j]["id"] = idx
            link_id_dict[idx] = (i, j)
            idx += 1
        return topology, link_id_dict

    def _load_capacities(self) -> None:
        if self.traffic_profile == "gravity_full":
            capacity_file = os.path.join(self.dataset_dir, "capacities", "graph-TM-" + str(self.num_sample) + ".txt")
        else:
            capacity_file = os.path.join(self.dataset_dir, "capacities", "graph.txt")
        with open(capacity_file) as fd:
            for line in fd:
                if "Link_" in line:
                    camps = line.split(" ")
                    self.G[int(camps[1])][int(camps[2])]["capacity"] = int(camps[4])

    def _load_traffic_matrix(self) -> None:
        tm_file = os.path.join(self.dataset_dir, "TM", "TM-" + str(self.num_sample))
        self.traffic_demand = np.zeros((self.G.number_of_nodes(), self.G.number_of_nodes()))
        with open(tm_file) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                self.traffic_demand[int(camps[1]), int(camps[2])] = float(camps[3])

    def _get_weights(self) -> None:
        weights = np.zeros(shape=(self.G.number_of_edges()), dtype=np.float32)
        for i, j in self.G.edges():
            weights[self.G[i][j]["id"]] = self.G[i][j]["weight"]
        self.weights = np.divide(weights, np.max(weights))  # otherwise does not give type hints

    def _update_weights(self, link) -> None:
        i, j = link
        self.G[i][j]["weight"] += self.weight_change

    def _set_weights(self) -> None:
        for i, j in self.G.edges():
            self.G[i][j]["weight"] = self.weights[self.G[i][j]["id"]]

    def _reset_edge_attributes(self, attributes=None) -> None:
        if attributes is None:
            attributes = list(DEFAULT_EDGE_ATTRIBUTES.keys())
        if type(attributes) != list:
            attributes = [attributes]
        for i, j in self.G.edges():
            for attribute in attributes:
                self.G[i][j][attribute] = DEFAULT_EDGE_ATTRIBUTES[attribute]

    def _normalize_traffic(self) -> None:
        for i, j in self.G.edges():
            self.G[i][j]["traffic"] /= self.G[i][j]["capacity"]

    def _successive_equal_cost_multipaths(self, src, dst, traffic) -> None:
        new_srcs = self.next_hop_dict[src][dst]
        traffic /= len(new_srcs)
        for new_src in new_srcs:
            self.G[src][new_src]["traffic"] += traffic
            if new_src != dst:
                self._successive_equal_cost_multipaths(new_src, dst, traffic)

    # def _distribute_link_traffic(self):
    #     self._reset_edge_attributes('traffic')
    #     visited_pairs = set()
    #     self.next_hop_dict = {i : {j : set() for j in range(self.G.number_of_nodes()) if j != i} for i in range(self.G.number_of_nodes())}
    #     for src in range(self.G.number_of_nodes()):
    #         for dst in range(self.G.number_of_nodes()):
    #             if src == dst: continue
    #             if (src, dst) not in visited_pairs:
    #                 routings = set([item for sublist in [[(routing[i], routing[i+1]) for i in range(len(routing)-1)] for routing in nx.all_shortest_paths(self.G, src, dst, 'weight')] for item in sublist])
    #                 for (new_src, next_hop) in routings:
    #                     self.next_hop_dict[new_src][dst].add(next_hop)
    #                     visited_pairs.add((new_src, dst))
    #             traffic = self.traffic_demand[src][dst]
    #             self._successive_equal_cost_multipaths(src, dst, traffic)
    #     self._normalize_traffic()

    def _distribute_link_traffic(self):
        self._reset_edge_attributes("traffic")
        self.next_hop_dict = {
            i: {j: set() for j in range(self.G.number_of_nodes()) if j != i} for i in range(self.G.number_of_nodes())
        }
        pred = johnson_all_sp(self.G)
        visited_pairs = set()
        self.next_hop_dict = {
            i: {j: set() for j in range(self.G.number_of_nodes()) if j != i} for i in range(self.G.number_of_nodes())
        }
        for src in range(self.G.number_of_nodes()):
            for dst in range(self.G.number_of_nodes()):
                if src == dst:
                    continue
                if (src, dst) not in visited_pairs:
                    routings = set(
                        [
                            item
                            for sublist in [
                                [(routing[i], routing[i + 1]) for i in range(len(routing) - 1)]
                                for routing in _build_paths_from_predecessors({src}, dst, pred=pred[src])
                            ]
                            for item in sublist
                        ]
                    )
                    for new_src, next_hop in routings:
                        self.next_hop_dict[new_src][dst].add(next_hop)
                        visited_pairs.add((new_src, dst))
                traffic = self.traffic_demand[src][dst]
                self._successive_equal_cost_multipaths(src, dst, traffic)
        self._normalize_traffic()

    def _get_link_traffic(self):
        self._distribute_link_traffic()
        link_traffic = np.zeros(shape=(self.G.number_of_edges()))
        for i, j in self.G.edges():
            link_traffic[self.G[i][j]["id"]] = self.G[i][j]["traffic"]
        self.link_traffic = link_traffic

    def _metropolis_acceptance(self, current_reward_measure) -> float:
        if self.sim_annealing:
            return np.exp(-(current_reward_measure - self.reward_measure) / self.temperature)
        else:
            return 1
