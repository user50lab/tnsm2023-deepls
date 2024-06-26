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
    # 返回一个NumPy数组（npt.NDArray）类型的对象，这个数组代表环境的初始状态的观测。
    def reset(self) -> npt.NDArray:
        if self.eval:
            # 如果处于评估模式，设置测试样本的编号为当前的test_index值。
            self.num_sample = self.test_index
        else:
            # 使用环境独有的随机数生成器对象self.rng来选择一个整数作为样本编号，范围从0 (包括) 到99 (不包括)。
            self.num_sample = self.rng.integers(low=0, high=99)
        # test_index值递增1，用于追踪评估过程中的当前样本编号。
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

    # 返回网络的当前状态的观察值，观察值通常用于提供环境的当前信息
    def observation(self) -> npt.NDArray:
        # best_reward_measure = np.full(shape=(self.G.number_of_edges(), 1), fill_value=self.best_reward_measure)
        return np.concatenate([self.link_traffic.reshape(-1, 1), self.weights.reshape(-1, 1)], axis=-1, dtype=np.float32)   # 首先将类变量 self.link_traffic 和 self.weights 转换为列向量；然后，它使用 np.concatenate 函数按照最后一个轴（axis=-1），也就是列，将这两个列向量拼接起来形成一个新的二维数组。

    ### 用于计算和更新当前的奖励值。
    def reward(self) -> float:
        current_reward_measure = np.max(self.link_traffic)   # 计算当前网络中所有链路流量的最大值，并将其存储在变量 current_reward_measure 中。
        reward = self.reward_measure - current_reward_measure   # 计算当前奖励值，方法是从某个存储在实例中的基线奖励测量 self.reward_measure 中减去 current_reward_measure。
        # reward = max(0, (self.best_reward_measure - current_reward_measure))
        if self.best_reward_measure > current_reward_measure:
            self.best_reward_measure = current_reward_measure   # 更新 self.best_reward_measure 为新的 current_reward_measure
            self.best_weights = np.copy(self.weights)   # 用当前链路的权重 self.weights 更新 self.best_weights
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
        # 调用了环境的动作空间对象self.action_space的seed方法，并将传入的种子seed值传递给它。这是为了确保后续基于动作空间的随机选择是可重现的，因为相同的种子值会产生相同的随机序列。
        self.action_space.seed(seed)
        # 设置观测空间对象self.observation_space的随机数生成器的种子。这样做可以确保任何依赖于观测空间的随机性都是可重现的。
        self.observation_space.seed(seed)
        self.rng = np.random.default_rng(seed)

    ### 负责加载网络拓扑的数据结构和识别每条边的唯一标识。
    # 这个函数的声明指明了它返回一个元组，包含一个networkx的有向图(DiGraph)和一个字典(dict)。
    def _load_topology(self) -> Tuple[nx.DiGraph, dict]:
        try:
            # 使用os.path.join方法构建拓扑图文件的路径。
            nx_file = os.path.join(self.base_dir, self.env_type, "graph_attr.txt")
            # 尝试通过读取GML文件（图建模语言文件）来构建一个networkx的有向图对象。nx.read_gml函数用于读取GML文件，并且destringizer=int参数指示将所有的属性值转换为整数类型。
            topology = nx.DiGraph(nx.read_gml(nx_file, destringizer=int))
        # 如果在试图加载GML文件时发生错误（比如文件不存在），则执行except子句。
        except:
            # 如果加载GML文件失败，则创建一个空的DiGraph对象。
            topology = nx.DiGraph()
            # 构建graph.txt文件的路径，这个文件包含边缘容量信息。
            capacity_file = os.path.join(self.dataset_dir, "capacities", "graph.txt")
            # 打开容量文件，并将其内容赋值给fd变量。
            with open(capacity_file) as fd:
                # 遍历文件中的每一行。
                for line in fd:
                    # 检查这一行是否包含"Link_"字符串，这可能是文件中标识链路数据的方式。
                    if "Link_" in line:
                        # 将行按空格分割成多个部分，并将这些部分存储在列表camps中。
                        camps = line.split(" ")
                        # 在图topology中添加一条边，从camps[1]到camps[2]，这两个值在列表中位置分别是第二和第三，通过int函数转化为整数。
                        topology.add_edge(int(camps[1]), int(camps[2]))
                        # 为刚刚添加的边设置一个属性“bandwidth”，它是camps列表中第五个元素的值，转换为整数。
                        topology[int(camps[1])][int(camps[2])]["bandwidth"] = int(camps[4])
        # 初始化一个空字典，用于存储链接ID与它们的节点对应关系。
        link_id_dict = {}
        # 初始化索引变量idx为0，用于给边分配唯一的标识符。
        idx = 0
        # 遍历图topology中所有的边。
        for i, j in topology.edges():
            # 给每条边分配一个唯一的ID。
            topology[i][j]["id"] = idx
            # 在link_id_dict中记录边的ID与其起点和终点的映射。
            link_id_dict[idx] = (i, j)
            # 将索引递增，为下一条边准备新的ID。
            idx += 1
        # 返回拓扑图(topology)和链接ID字典(link_id_dict)的元组。
        return topology, link_id_dict

    ### 用于从文件加载网络链路的容量信息。
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
            # 这里连续调用两次fd.readline()来跳过文件的前两行。这通常用于跳过文件头部的信息，例如列标题或者注释。
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                self.traffic_demand[int(camps[1]), int(camps[2])] = float(camps[3])

    def _get_weights(self) -> None:
        # 这个数组用于存储图中每条边的权重。
        weights = np.zeros(shape=(self.G.number_of_edges()), dtype=np.float32)
        for i, j in self.G.edges():
            # 该行将边 (i, j) 的权重 weight 赋值给之前创建的 weights 数组。权重是从图的边属性 ["weight"] 中获取的，而数组中对应的索引位置由边的另一个属性 ["id"] 确定。
            weights[self.G[i][j]["id"]] = self.G[i][j]["weight"]
        # 这一行把 weights 数组中的所有权重值除以数组中的最大权重值，实现了权值的归一化处理。
        self.weights = np.divide(weights, np.max(weights))  # otherwise does not give type hints

    def _update_weights(self, link) -> None:
        i, j = link
        self.G[i][j]["weight"] += self.weight_change

    def _set_weights(self) -> None:
        for i, j in self.G.edges():
            self.G[i][j]["weight"] = self.weights[self.G[i][j]["id"]]

    # 用于重置网络图中所有边的属性到默认值。
    def _reset_edge_attributes(self, attributes=None) -> None:
        if attributes is None:
            attributes = list(DEFAULT_EDGE_ATTRIBUTES.keys())
        if type(attributes) != list:
            # 如果attributes不是列表，这行代码会创建一个包含attributes的新列表。这是为了确保attributes是可迭代的，下面的循环不会因类型不符而出错。
            attributes = [attributes]
        for i, j in self.G.edges():
            for attribute in attributes:
                # 将所有选定的边属性重置为其默认值。
                self.G[i][j][attribute] = DEFAULT_EDGE_ATTRIBUTES[attribute]

    def _normalize_traffic(self) -> None:
        for i, j in self.G.edges():
            self.G[i][j]["traffic"] /= self.G[i][j]["capacity"]   # 在循环内部，边上的流量 (traffic) 被除以相应边的容量 (capacity)。这是流量归一化的过程，它确保了每条边上的流量不会超过其容量。这步处理对于保持网络的稳定性至关重要。

    def _successive_equal_cost_multipaths(self, src, dst, traffic) -> None:
        new_srcs = self.next_hop_dict[src][dst]   # 从之前创建的字典 self.next_hop_dict 中，获取从 src 到 dst 的所有下一跳节点，并将它们存储在 new_srcs 中。
        traffic /= len(new_srcs)   # 将流量 traffic 均等分配给所有的下一跳节点，即通过将总流量除以下一跳节点的数量 len(new_srcs)。
        for new_src in new_srcs:   # 遍历路由下一跳节点集 new_srcs。
            self.G[src][new_src]["traffic"] += traffic   # 对每个下一跳节点 new_src，在边 (src, new_src) 上增加它的流量份额。
            if new_src != dst:   # 如果下一跳节点不是目的节点 dst，
                self._successive_equal_cost_multipaths(new_src, dst, traffic)   # 递归调用 _successive_equal_cost_multipaths 方法处理从下一跳节点 new_src 到目的节点 dst 的流量。这种递归确保了流量能够沿着多条路径向下传播。

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

    ### 用于在网络图中分配流量，依照源节点和目的节点对间的所有最短路径。
    def _distribute_link_traffic(self):
        # 为每条边的 "traffic" 属性设置到默认值。
        self._reset_edge_attributes("traffic")
        # 初始化一个叫做 next_hop_dict 的字典，其中包含所有节点对，排除同一个节点作为源节点和目的节点的情况。字典用于存储可能的下一跳节点。
        self.next_hop_dict = {
            i: {j: set() for j in range(self.G.number_of_nodes()) if j != i} for i in range(self.G.number_of_nodes())
        }
        # 调用 johnson_all_sp 函数，计算图 self.G 中所有对节点间的所有最短路径的前驱节点。这个函数的返回值可能是一个字典，其键是图中的每个节点，值是另一个字典，映射到达其他所有节点的前驱节点。
        pred = johnson_all_sp(self.G)
        # 初始化一个集合 visited_pairs 来跟踪已经访问过的节点对，以避免重复处理。
        visited_pairs = set()
        # 再次初始化 next_hop_dict 字典。这可能是重复的操作，如果是，那它只是确保 next_hop_dict 是空的。
        self.next_hop_dict = {
            i: {j: set() for j in range(self.G.number_of_nodes()) if j != i} for i in range(self.G.number_of_nodes())
        }
        for src in range(self.G.number_of_nodes()):   # 开始第一个循环，遍历图中所有的源节点（src）。
            for dst in range(self.G.number_of_nodes()):   # 开始嵌套循环，遍历所有可能的目的节点（dst）。
                if src == dst:
                    continue
                if (src, dst) not in visited_pairs:   # 检查节点对 (src, dst) 是否已经被访问过。
                    # 为每对未访问的节点对 (src, dst) 计算所有路径上的下一跳，并将它们添加到 routings 集合中。
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
                    for new_src, next_hop in routings:   # 遍历 routings 中的元素，每个元素是一个（源节点，下一跳）对。
                        self.next_hop_dict[new_src][dst].add(next_hop)   # 更新 next_hop_dict 字典，将下一跳 next_hop 添加到对应的（源节点，目的节点）条目中。
                        visited_pairs.add((new_src, dst))   # 将（源节点，目的节点）对添加到 visited_pairs 集合中，标记为已访问。
                traffic = self.traffic_demand[src][dst]   # 获取从源节点 src 到目的节点 dst 的流量需求。
                self._successive_equal_cost_multipaths(src, dst, traffic)
        self._normalize_traffic()

    ### 收集网络图中每条边上的流量信息并将其存储。
    def _get_link_traffic(self):
        self._distribute_link_traffic()
        # 这个数组将被用来存储每条边的流量信息
        link_traffic = np.zeros(shape=(self.G.number_of_edges()))
        for i, j in self.G.edges():
            # 对于每一对节点(i, j)，这行代码从图的边属性中获取"traffic"的值，并赋给link_traffic数组的相应索引位置。索引是由边的属性["id"]给出的。此操作将每条边的流量值存储在数组中，从而使得每个边的唯一标识符id与其流量值相关联。
            link_traffic[self.G[i][j]["id"]] = self.G[i][j]["traffic"]
        # 这行代码将之前创建并填充了数据的link_traffic数组赋值给类的实例变量self.link_traffic。这样，链接流量的信息就可以在类的其他部分被调用和使用了。
        self.link_traffic = link_traffic

    def _metropolis_acceptance(self, current_reward_measure) -> float:
        if self.sim_annealing:
            return np.exp(-(current_reward_measure - self.reward_measure) / self.temperature)
        else:
            return 1
