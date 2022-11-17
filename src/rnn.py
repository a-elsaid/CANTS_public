""" RNN class """
import sys
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
from helper import ACTIVATIONS, LOSS


class Node:
    """
    RNN node
    :param int id: node id
    :param Point point: search space point
    :param bias: node bias
    :param str activation_type: type of the activation function
    """

    counter = 0

    def __init__(
        self,
        point,
        lag: int,
        activation_type: str = "relu",
    ) -> None:
        self.id = self.counter
        self.activation = ACTIVATIONS[activation_type]
        self.lag = lag
        self.point = point
        self.type = self.point.type
        self.bias = 0.0
        """
        if self.type == 0:
            self.bias = torch.tensor(
                np.random.random(), dtype=torch.float64, requires_grad=True
            )
        """
        self.fan_in = []  # Coming in nodes
        self.fan_out = {}  # Going out edges and their nodes
        self.value = torch.tensor(
            0.0, dtype=torch.float64, requires_grad=False
        )  # value in node
        self.out = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
        self.signals_to_receive: int = 0  # number of fan_in
        self.waiting_signals: int = 0
        Node.counter += 1
        self.fired = False

    def add_fan_out_node(self, in_node, out_node, wght: float) -> None:
        if out_node not in self.fan_out:
            self.fan_out[out_node] = Edge(in_node, out_node, wght)

    def add_fan_in_node(self, in_node) -> None:
        if in_node not in self.fan_in:
            self.fan_in.append(in_node)
            self.signals_to_receive += 1

    def fire(
        self,
    ) -> None:
        """
        firing the node when activated
        """
        self.waiting_signals = self.signals_to_receive
        logger.debug(
            f"Node({self.id}) [Point({self.point.id})] Fired: Sig({self.value} \
            + {self.bias}) = {self.activation(self.value + self.bias)}"
        )
        self.out = self.activation(self.value + self.bias)
        self.fired = True
        logger.debug(f"Node({self.id}) fired: {self.out}")
        for edge in self.fan_out.values():
            logger.debug(
                f"\t Node({self.id}) Sent Signal {self.out} * " +
                f"{edge.weight} ({self.out * edge.weight}) to " +
                f"Node({edge.out_node.id})"
            )
            edge.out_node.synaptic_signal(self.out * edge.weight)

    def reset_node(
        self,
    ) -> None:
        with torch.no_grad():
            self.value = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
            self.out = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)

    def synaptic_signal(self, signal: float) -> None:
        """
        receiving a synaptic signal and firing when all signals are received
        """
        self.fired = False
        logger.debug(
            f"Node({self.id}) [Point:{self.point.id}, Type: {self.type}] revieved signal: {signal}"
        )
        logger.debug(
            f"Node({self.id}) [Point({self.point.id})] value before receiving signal: {self.value}"
        )
        self.value += signal
        logger.debug(
            f"Node({self.id}) [Point({self.point.id})] value after received " +
            f"signal: {self.value} -- Waiting {self.waiting_signals} Signals"
        )
        self.waiting_signals -= 1

        if self.waiting_signals <= 0:
            logger.debug(f"Node({self.id}) is going to fire: {self.value}")
            self.fire()


class LSTM_Node(Node):
    def __init__(
        self,
        point,
        lag: int,
        activation_type: str = "relu",
    ) -> None:
        super().__init__(point, lag, activation_type)
        self.wf = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.uf = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.wi = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.ui = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.wo = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.uo = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.wg = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.ug = torch.tensor(
            np.random.random() * 5, dtype=torch.float64, requires_grad=True
        )
        self.ct = 0.0
        self.ht = 0.0

    def fire(self) -> None:
        self.waiting_signals = self.signals_to_receive
        ft = self.activation(self.wf * self.value + self.uf * self.ht)
        it = self.activation(self.wi * self.value + self.ui * self.ht)
        ot = self.activation(self.wo * self.value + self.uo * self.ht)
        c_ = self.activation(self.wg * self.value + self.ug * self.ht)
        ct = ft * self.ct + it * c_
        self.ct = ct.item()
        self.out = ot * self.activation(ct)
        self.ht = self.out.item()
        logger.debug(f"Node({self.id}) fired: {self.out}")
        """
        if self.out > 2.0:
            ipdb.sset_trace()
        """
        for edge in self.fan_out.values():
            logger.debug(
                f"\t Node({self.id}) Sent Signal {self.out} * {edge.weight} " +
                f"({self.out * edge.weight}) to Node({edge.out_node.id})"
            )
            edge.out_node.synaptic_signal(self.out * edge.weight)
        self.fired = True


class Edge:
    """
    RNN Edge
    :param Node from: source node
    :param Node to: sink node
    :param float weight: weight fro source node to sink node
    """

    counter = 0

    def __init__(self, in_node: Node, out_node: Node, weight: float = None):
        self.id = self.counter
        self.in_node = in_node
        self.out_node = out_node
        if not weight:
            weight = np.random.random() * 5
        self.weight = torch.tensor(
            float(weight), dtype=torch.float64, requires_grad=True
        )


class RNN:
    """
    RNN class
    """

    def __init__(
        self,
        paths,
        lags: int,
        loss_fun: str = "mse",
        act_fun: str = "sigmoid",
        centeroids_clusters: Dict[int, np.ndarray] = None,
    ):
        self.fitness: float = None
        self.err: torch.float64 = torch.tensor(0.0, dtype=torch.float64)
        self.total_err = 0.0
        self.nodes: Dict[int, Node] = {}
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        logger.debug(f"LOSS type: {loss_fun}  ")
        self.act_fun = act_fun
        self.loss_fun = LOSS[loss_fun]
        self.centeroids_clusters = centeroids_clusters
        self.lags = lags

        for point in [pnt for path in paths for pnt in path]:
            if point.id in self.nodes:
                continue
            node = LSTM_Node(point, point.pos_l, activation_type=self.act_fun)
            self.nodes[point.id] = node
            if point.type == 0:
                self.input_nodes.append(node)
            elif point.type == 2:
                self.output_nodes.append(node)

        for path in paths:
            for i, curr_p in enumerate(path[:-1]):
                curr_node = self.nodes[curr_p.id]
                next_p = path[i + 1]
                next_node = self.nodes[next_p.id]
                if curr_node.point.id in next_node.fan_out:
                    continue
                if self.centeroids_clusters:
                    curr_node.add_fan_out_node(curr_node, next_node, curr_p.pos_w)
                else:
                    curr_node.add_fan_out_node(
                        curr_node, next_node, curr_p.fan_out[next_p].weight
                    )
                next_node.add_fan_in_node(curr_node)

        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            logger.debug(
                f"Node({node.id}) Point({node.point.id}) Type: {node.type} " +
                f"WaitingSignals: {node.waiting_signals}"
            )
            for e in node.fan_out.values():
                logger.debug(f"\t Out Node({e.out_node.id})")
            for n in node.fan_in:
                logger.debug(f"\t In Node({n.id})")

    def build_fully_connected_rnn(
        self, input_names, output_names, lags, hid_layers, hid_nodes
    ) -> None:
        from search_space_ants import RNNSearchSpaceANTS

        self.lags = lags
        Point = RNNSearchSpaceANTS.Point
        for i, name in enumerate(input_names):
            for lag in range(lags):
                node = LSTM_Node(Point(None, None, 0, name, i), lag, self.act_fun)
                self.nodes[node.id] = node
                self.input_nodes.append(node)

        for name in output_names:
            node = LSTM_Node(Point(None, None, 2, name, i), lags - 1, self.act_fun)
            self.nodes[node.id] = node
            self.output_nodes.append(node)

        next_layer = self.output_nodes
        for _ in range(hid_layers):
            curr_layer = []
            for _ in range(hid_nodes):
                for lag in range(lags):
                    node = LSTM_Node(Point(None, None, 1, None, None), lag, self.act_fun)
                    self.nodes[node.id] = node
                    curr_layer.append(node)
            for in_node in curr_layer:
                for out_node in next_layer:
                    in_node.add_fan_out_node(in_node, out_node, np.random.random())
                    out_node.add_fan_in_node(in_node)
            next_layer = curr_layer

        for in_node in self.input_nodes:
            for out_node in next_layer:
                in_node.add_fan_out_node(in_node, out_node, np.random.random())
                out_node.add_fan_in_node(in_node)

        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            logger.debug(
                f"Node({node.id}): Type: {node.type}  Name: {node.point.name} " +
                f"Point: {node.point.id}"
            )
            logger.debug(
                f"\tOut Nodes: {[edge.out_node.id for edge in node.fan_out.values()]}"
            )
            logger.debug(f"\tIn Nodes: {[node.id for node in node.fan_in]}")

    def feedforward(
        self,
        inputs: np.ndarray,
    ) -> List[float]:
        """
        feeding forward a data point to get an output
        """
        """
        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            node.fired = False
        """
        for node in self.input_nodes:
            node.synaptic_signal(inputs[self.lags - node.lag - 1][node.point.inout_num])

        raise_err = False
        for n in self.nodes.values():
            if not n.fired:
                logger.error(f"Node {n.id} Didn't Fire  Point: {n.point.id}")
                raise_err = True
        if raise_err:
            sys.exit()

        res = [node.out for node in self.output_nodes]
        """
        for node in self.output_nodes:
            node.value = 0.0
        """
        return res

    def do_epoch(
        self, inputs: np.ndarray, outputs: np.ndarray, loss_fun=None, do_feedbck=True
    ) -> None:
        """
        perform one epoch using the whole dataset
        """
        if not loss_fun:
            loss_fun = self.loss_fun
        self.total_err = 0.0
        err = None
        # for i in tqdm(range(len(inputs) - self.lags)):
        for i in range(len(inputs) - self.lags):
            res = self.feedforward(inputs[i : i + self.lags])
            logger.debug(f"feedforward return (output nodes values): {res}")
            err = [loss for loss in loss_fun(res, outputs[i])]
            err = sum(err) / len(err)
            self.total_err += err
            if do_feedbck:
                self.feedbackward(err)
            for node in self.nodes.values():
                node.reset_node()

        self.total_err /= i
        logger.info(f"Training Epoch average Total Error: {self.total_err}")

    def test_rnn(self, inputs: np.ndarray, outputs: np.ndarray, loss_fun=None) -> None:
        """
        feeding forward a all testing data points and getting
        the output to calculate the error between the output
        and the groundtruth
        """
        with torch.no_grad():
            if not loss_fun:
                loss_fun = self.loss_fun
            err = 0.0
            for i in range(len(inputs) - self.lags):
                res = self.feedforward(inputs[i : i + self.lags])
                e = [loss for loss in loss_fun(res, outputs[i])]
                e = sum(e) / len(e)
                err += e
            self.fitness = err.item() / i

    def feedbackward(self, err) -> None:
        """
        calculating gradients
        """
        logger.debug(f"ERR: {err}")
        err.backward()
        with torch.no_grad():
            for node in self.nodes.values():
                for edge in node.fan_out.values():
                    logger.debug(
                        f"From Point {node.point.id} To Point \
                            {edge.out_node.point.id}: dweight = {edge.weight.grad}"
                    )
                    logger.debug(f"\t Weight before update: {edge.weight}")
                    edge.weight += edge.weight.grad
                    logger.debug(f"\t Weight after update: {edge.weight}")
                    edge.weight.grad.zero_()
                if isinstance(node, LSTM_Node):
                    node.wf -= node.wf.grad
                    node.uf -= node.uf.grad
                    node.wi -= node.wi.grad
                    node.ui -= node.ui.grad
                    node.wo -= node.wo.grad
                    node.uo -= node.uo.grad
                    node.wg -= node.wg.grad
                    node.ug -= node.ug.grad
                    if torch.isnan(node.wf.grad):
                        exit()
                    node.wf.grad.zero_()
                    node.uf.grad.zero_()
                    node.wi.grad.zero_()
                    node.ui.grad.zero_()
                    node.wo.grad.zero_()
                    node.uo.grad.zero_()
                    node.wg.grad.zero_()
                    node.ug.grad.zero_()
