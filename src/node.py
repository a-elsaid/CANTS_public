import numpy as np
import torch
from loguru import logger
from helper import ACTIVATIONS

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
                np.random.normal(), dtype=torch.float64, requires_grad=True
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
            self.fan_out[out_node] = Edge(in_node=in_node, out_node=out_node, weight=wght)

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
        logger.trace(
            f"Node({self.id}) [Point({self.point.id})] Fired: Sig({self.value} \
            + {self.bias}) = {self.activation(self.value + self.bias)}"
        )
        self.out = self.activation(self.value + self.bias)
        self.fired = True
        logger.trace(f"Node({self.id}) fired: {self.out}")
        for edge in self.fan_out.values():
            logger.trace(
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
        logger.trace(
            f"Node({self.id}) [Point:{self.point.id}, Type: {self.type}, Class:{type(self)}] revieved signal: {signal}"
        )
        logger.trace(
            f"Node({self.id}) [Point({self.point.id})] value before receiving signal: {self.value}"
        )
        self.value += signal
        logger.trace(
            f"Node({self.id}) [Point({self.point.id})] value after received " +
            f"signal: {self.value} -- Waiting {self.waiting_signals} Signals"
        )
        self.waiting_signals -= 1

        if self.waiting_signals <= 0:
            logger.trace(f"Node({self.id}) is going to fire: {self.value}")
            self.fire()


class LSTM_Node(Node):
    def __init__(
        self,
        point,
        lag: int,
        activation_type: str = "relu",
    ) -> None:
        super().__init__(point, lag, activation_type)
        self.wf, self.uf, self.wi, self.ui, self.wo, self.uo, self.wg, self.ug = self.gen_lstm_hid_state()
        self.ct = 0.0
        self.ht = 0.0

    def gen_lstm_hid_state(self):
        wf = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        uf = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        wi = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        ui = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        wo = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        uo = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        wg = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        ug = torch.tensor(
            np.random.normal() * 5, dtype=torch.float64, requires_grad=True
        )
        return wf, uf, wi, ui, wo, uo, wg, ug

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
        logger.trace(f"Node({self.id}) fired: {self.out}")
        """
        if self.out > 2.0:
            ipdb.sset_trace()
        """
        for edge in self.fan_out.values():
            logger.trace(
                f"\t Node({self.id}) Sent Signal {self.out} * {edge.weight} " +
                f"({self.out * edge.weight}) to Node({edge.out_node.id})"
            )
            edge.out_node.synaptic_signal(self.out * edge.weight)
        self.fired = True

class BNN_LSTM_Node(LSTM_Node):
    def __init__(
        self,
        point,
        lag,
        activation_type: str = "relu",
        prior_mu=0,
        prior_sigma=5,
    ) -> None:
        super().__init__(point, lag, activation_type)
        self.ct = 0.0
        self.ht = 0.0
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        (
        self.wf_ro, 
        self.uf_ro, 
        self.wi_ro, 
        self.ui_ro, 
        self.wo_ro, 
        self.uo_ro, 
        self.wg_ro, 
        self.ug_ro ) = self.gen_lstm_hid_state()
        
        (
        self.wf_mu, 
        self.uf_mu, 
        self.wi_mu, 
        self.ui_mu, 
        self.wo_mu, 
        self.uo_mu, 
        self.wg_mu, 
        self.ug_mu ) = self.gen_lstm_hid_state()

        self.gates_ro = [self.wf_ro,
                         self.uf_ro,
                         self.wi_ro,
                         self.ui_ro,
                         self.wo_ro,
                         self.uo_ro,
                         self.wg_ro,
                         self.ug_ro]
        self.gates_mu = [self.wf_mu,
                         self.uf_mu,
                         self.wi_mu,
                         self.ui_mu,
                         self.wo_mu,
                         self.uo_mu,
                         self.wg_mu,
                         self.ug_mu]

        # self.wf, self.uf, self.wi, self.ui, self.wo, self.uo, self.wg, self.ug = self.gen_lstm_hid_state()

    def add_fan_out_node(
                         self, 
                         e_id, 
                         out_node
    ) -> None:
        if out_node not in self.fan_out:
            edge = BNN_Edge(
                            e_id=e_id,
                            in_node=self, 
                            out_node=out_node,
            )
            self.fan_out[out_node] = edge

    def add_fan_in_node(self, in_node) -> None:
        if in_node not in self.fan_in:
            self.fan_in.append(in_node)
            self.signals_to_receive += 1

    def fire(self) -> None:
        self.waiting_signals = self.signals_to_receive

        ft_mu = self.activation(self.wf_mu * self.value + self.uf_mu * self.ht)
        ft_ro = self.activation(self.wf_ro * self.value + self.uf_ro * self.ht)
        ft_epsilon = torch.randn_like(ft_ro)
        ft_sample = ft_mu + torch.exp(ft_ro) * ft_epsilon
        ft = self.activation(ft_sample)

        it_mu = self.activation(self.wi_mu * self.value + self.ui_mu * self.ht)
        it_ro = self.activation(self.wi_ro * self.value + self.ui_ro * self.ht)
        it_epsilon = torch.randn_like(it_ro)
        it_sample = it_mu + torch.exp(it_ro) * it_epsilon
        it = self.activation(it_sample)

        ot_mu = self.activation(self.wo_mu * self.value + self.uo_mu * self.ht)
        ot_ro = self.activation(self.wo_ro * self.value + self.uo_ro * self.ht)
        ot_epsilon = torch.randn_like(ot_ro)
        ot_sample = ot_mu + torch.exp(ot_ro) * ot_epsilon
        ot = self.activation(ot_sample)

        c_mu = self.activation(self.wg_mu * self.value + self.ug_mu * self.ht)
        c_ro = self.activation(self.wg_ro * self.value + self.ug_ro * self.ht)
        c_epsilon = torch.randn_like(c_ro)
        c_sample = c_mu + torch.exp(c_ro) * c_epsilon
        c_ = self.activation(c_sample)

        ct = ft * self.ct + it * c_

        self.ct = ct.item()
        self.out = ot * self.activation(ct)
        self.ht = self.out.item()

        logger.trace(f"Node({self.id}) fired: {self.out}")
        
        for edge in self.fan_out.values():
            mu = self.out * edge.weight_mu
            ro = self.out * edge.weight_ro
            epsilon = torch.randn_like(ro)
            r = mu + torch.exp(ro) * epsilon

            edge.out_node.synaptic_signal(r)
            logger.trace(
                f"\t Node({self.id}) Sent Signal " +
                f"({r}) to Node({edge.out_node.id})"
            )
        self.fired = True




class BNN_Edge:

    counter = 0

    def __init__(self, 
                 e_id, 
                 in_node: Node=None, 
                 out_node: Node=None,
                 prior_mu=0,
                 prior_sigma=5,
    ):
        if e_id==None:
            self.id = self.counter
            BNN_Edge+=1
        else:
            self.id = e_id
        self.in_node = in_node
        self.out_node = out_node
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.weight_mu = torch.tensor(
            float((np.random.normal()+self.prior_mu) * self.prior_sigma), 
            dtype=torch.float64, requires_grad=True
        )
        self.weight_ro = torch.tensor(
            float((np.random.normal()+self.prior_mu) * self.prior_sigma), 
            dtype=torch.float64, requires_grad=True
        )

class Edge:
    """
    RNN Edge
    :param Node from: source node
    :param Node to: sink node
    :param float weight: weight fro source node to sink node
    """

    counter = 0

    def __init__(self, e_id = None, in_node: Node=None, out_node: Node=None, weight: float = None):
        if e_id==None:
            self.id = self.counter
        else:
            self.id = e_id
            Edge+=1

        self.in_node = in_node
        self.out_node = out_node
        if not weight:
            weight = np.random.normal() * 5
        self.weight = torch.tensor(
            float(weight), dtype=torch.float64, requires_grad=True
        )
