""" RNN class """
import sys
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
from helper import ACTIVATIONS, LOSS
from node import *

class RNN:
    """
    RNN class
    """
    counter = 0

    def __init__(
        self,
        paths,
        lags: int,
        loss_fun: str = "mse",
        act_fun: str = "sigmoid",
        centeroids_clusters: Dict[int, np.ndarray] = None,
    ):
        self.id = self.counter
        RNN.counter+=1
        self.fitness: float = None
        self.uncertainity = 0.
        self.mean_bnn_fit = 0.
        self.score = 0.

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
            logger.trace(
                f"Node({node.id}) Point({node.point.id}) Type: {node.type} " +
                f"WaitingSignals: {node.waiting_signals}"
            )
            for e in node.fan_out.values():
                logger.trace(f"\t Out Node({e.out_node.id})")
            for n in node.fan_in:
                logger.trace(f"\t In Node({n.id})")

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
                    in_node.add_fan_out_node(in_node, out_node, np.random.normal())
                    out_node.add_fan_in_node(in_node)
            next_layer = curr_layer

        for in_node in self.input_nodes:
            for out_node in next_layer:
                in_node.add_fan_out_node(in_node, out_node, np.random.normal())
                out_node.add_fan_in_node(in_node)

        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            logger.trace(
                f"Node({node.id}): Type: {node.type}  Name: {node.point.name} " +
                f"Point: {node.point.id}"
            )
            logger.trace(
                f"\tOut Nodes: {[edge.out_node.id for edge in node.fan_out.values()]}"
            )
            logger.trace(f"\tIn Nodes: {[node.id for node in node.fan_in]}")


    def generate_bnn_version(self,) -> None:

        self.bnn_nodes = {}
        self.bnn_input_nodes = []
        self.bnn_output_nodes = []
        for n_id, node in self.nodes.items():
            if type(node)==LSTM_Node:
                bnn_node = BNN_LSTM_Node(node.point, node.lag, activation_type='sigmoid')
                bnn_node.id = node.id
                self.bnn_nodes[node.id] = bnn_node
            else:
                raise("BNN for None LSTM is not implemented yet")

            if node in self.input_nodes:
                self.bnn_input_nodes.append(self.bnn_nodes[node.id])
            elif node in self.output_nodes:
                self.bnn_output_nodes.append(self.bnn_nodes[node.id])

        for node in self.nodes.values():
            for rnn_edge in node.fan_out.values():
                in_node  = self.bnn_nodes[rnn_edge.in_node.id]
                out_node = self.bnn_nodes[rnn_edge.out_node.id]
                # edge = BNN_Edge(e_id=rnn_edge.id, in_node=in_node, out_node=out_node)
                # in_node.fan_out[out_node] = edge

                out_node.add_fan_in_node(in_node)
                in_node.add_fan_out_node(e_id=rnn_edge.id,  out_node=out_node)

        for node in self.bnn_nodes.values():
            node.waiting_signals = node.signals_to_receive
            logger.trace(
                f"Node({node.id}) Point({node.point.id}) Type: {node.type} " +
                f"WaitingSignals: {node.waiting_signals}"
            )
                
    def feedforward(
        self,
        inputs: np.ndarray,
        active_inference: bool = False,
    ) -> List[float]:
        """
        feeding forward a data point to get an output
        """

        """
        for node in self.nodes.values():
            node.waiting_signals = node.signals_to_receive
            node.fired = False
        """
        if active_inference:
            nodes = self.bnn_nodes
            input_nodes  = self.bnn_input_nodes
            output_nodes = self.bnn_output_nodes
        else:
            nodes = self.nodes
            input_nodes  = self.input_nodes
            output_nodes = self.output_nodes

        for node in input_nodes:
            node.synaptic_signal(inputs[self.lags - node.lag - 1][node.point.inout_num])

        raise_err = False
        for n in nodes.values():
            if not n.fired:
                logger.error(f"Node {n.id} Didn't Fire  Point: {n.point.id}")
                raise_err = True
        if raise_err:
            raise("Stopping: One or more nodes DID NOT Fire")

        res = [node.out for node in output_nodes]

        return res

    def bnn_k_loss(self,):
        kl = 0
        for n in self.bnn_nodes.values():
            for ro,mu in zip(n.gates_mu, n.gates_ro):
                kl+=(ro.item()-np.log(n.prior_sigma)) + \
                    (ro.item()**2 + (mu.item() - n.prior_mu)**2) / (2*n.prior_sigma**2) - 0.5

            for e in n.fan_out.values():
                kl+=(e.weight_ro.item()-np.log(n.prior_sigma)) + \
                    (e.weight_ro.item()**2 + (e.weight_mu.item() - n.prior_mu)**2) / (2*n.prior_sigma**2) - 0.5
        
        return kl

    def do_epoch(
        self, 
        inputs: np.ndarray, 
        outputs: np.ndarray, 
        loss_fun=None, 
        do_feedbck=True, 
        active_inference=False,
        bnn_kl_param=.01,
    ) -> None:
        """
        perform one epoch using the whole dataset
        """
        if active_inference:
            nodes = self.bnn_nodes
        else:
            nodes = self.nodes
        if not loss_fun:
            loss_fun = self.loss_fun
        self.total_err = 0.0
        err = None
        # for i in tqdm(range(len(inputs) - self.lags)):
        for i in range(len(inputs) - self.lags):
            res = self.feedforward(inputs[i : i + self.lags], active_inference)
            logger.trace(f"feedforward return (output nodes values): {res}")
            kl = 0.0
            if active_inference:
                kl = self.bnn_k_loss()
            err = [loss+bnn_kl_param*kl for loss in loss_fun(res, outputs[i])]
            err = sum(err) / len(err)
            self.total_err += err
            if do_feedbck:
                self.feedbackward(err, active_inference)
            for node in nodes.values():
                node.reset_node()

        self.total_err /= i
        logger.info(f"Training Epoch average Total Error: {self.total_err}")

    def test_rnn(self, 
                 inputs: np.ndarray, 
                 outputs: np.ndarray, 
                 loss_fun=None, 
                 active_inference=False,
                 bnn_kl_param=.01,
        ) -> None:
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
                res = self.feedforward(inputs[i : i + self.lags], active_inference=active_inference)


                kl=0.0
                if active_inference:
                    kl = self.bnn_k_loss()

                e = [loss+bnn_kl_param*kl for loss in loss_fun(res, outputs[i])]
                e = sum(e) / len(e)
                err += e

            if not active_inference:
                self.fitness = err.item() / i
        return err.item()/i

    def feedbackward(self, err, active_inference=False) -> None:
        """
        calculating gradients
        """

        # torch.autograd.set_detect_anomaly(True)

        if active_inference:
            nodes = self.bnn_nodes
        else:
            nodes = self.nodes
        logger.debug(f"ERR: {err}")
        err.backward()
        with torch.no_grad():
            for node in nodes.values():
                for edge in node.fan_out.values():
                    if active_inference:
                        logger.trace(
                            f"From Point {node.point.id} To Point \
                                {edge.out_node.point.id}: \
                                dweight_mu = {edge.weight_mu.grad} \
                                dweight_ro = {edge.weight_ro.grad}"
                        )
                        logger.trace(f"\t Weight before update: dw_mu: {edge.weight_mu}, dw_ro: {edge.weight_ro}")
                        edge.weight_mu += edge.weight_mu.grad
                        edge.weight_ro += edge.weight_ro.grad
                        logger.trace(f"\t Weight after update: dw_mu: {edge.weight_mu}, dw_ro: {edge.weight_ro}")
                        edge.weight_mu.grad.zero_()
                        edge.weight_ro.grad.zero_()
                    else:
                        logger.trace(
                            f"From Point {node.point.id} To Point \
                                {edge.out_node.point.id}: dweight = {edge.weight.grad}"
                        )
                        logger.trace(f"\t Weight before update: {edge.weight}")
                        edge.weight += edge.weight.grad
                        logger.trace(f"\t Weight after update: {edge.weight}")
                        edge.weight.grad.zero_()

                if active_inference:
                    if isinstance(node, BNN_LSTM_Node):
                        node.wf_mu -= node.wf_mu.grad
                        node.uf_mu -= node.uf_mu.grad
                        node.wi_mu -= node.wi_mu.grad
                        node.ui_mu -= node.ui_mu.grad
                        node.wo_mu -= node.wo_mu.grad
                        node.uo_mu -= node.uo_mu.grad
                        node.wg_mu -= node.wg_mu.grad
                        node.ug_mu -= node.ug_mu.grad

                        node.wf_ro -= node.wf_ro.grad
                        node.uf_ro -= node.uf_ro.grad
                        node.wi_ro -= node.wi_ro.grad
                        node.ui_ro -= node.ui_ro.grad
                        node.wo_ro -= node.wo_ro.grad
                        node.uo_ro -= node.uo_ro.grad
                        node.wg_ro -= node.wg_ro.grad
                        node.ug_ro -= node.ug_ro.grad
                        if torch.isnan(node.wf_ro.grad):
                            exit()
                        node.wf_mu.grad.zero_()
                        node.uf_mu.grad.zero_()
                        node.wi_mu.grad.zero_()
                        node.ui_mu.grad.zero_()
                        node.wo_mu.grad.zero_()
                        node.uo_mu.grad.zero_()
                        node.wg_mu.grad.zero_()
                        node.ug_mu.grad.zero_()

                        node.wf_ro.grad.zero_()
                        node.uf_ro.grad.zero_()
                        node.wi_ro.grad.zero_()
                        node.ui_ro.grad.zero_()
                        node.wo_ro.grad.zero_()
                        node.uo_ro.grad.zero_()
                        node.wg_ro.grad.zero_()
                        node.ug_ro.grad.zero_()

                else:
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

    def copy_rnn(self,) -> List:
        nodes = self.nodes
        rnn_info = []
        rnn_info.append(self.id)
        rnn_info.append(self.fitness)
        rnn_info.append(self.score)
        with torch.no_grad():
            for node in nodes.values():
                for edge in node.fan_out.values():
                    rnn_info.append(edge.weight.detach().numpy())

                if isinstance(node, LSTM_Node):
                    rnn_info.append(node.wf.detach().numpy())
                    rnn_info.append(node.uf.detach().numpy())
                    rnn_info.append(node.wi.detach().numpy())
                    rnn_info.append(node.ui.detach().numpy())
                    rnn_info.append(node.wo.detach().numpy())
                    rnn_info.append(node.uo.detach().numpy())
                    rnn_info.append(node.wg.detach().numpy())
                    rnn_info.append(node.ug.detach().numpy())
        return rnn_info

    def assign_rnn(self, rnn_info) -> List:
        nodes = self.nodes
        self.fitness = rnn_info[1]
        self.score = rnn_info[2]
        i = 3
        with torch.no_grad():
            for node in nodes.values():
                for edge in node.fan_out.values():
                    edge.weight.detach_()
                    edge.weight = torch.from_numpy(rnn_info[i])
                    edge.weight.requires_grad_(True)
                    i+=1

                if isinstance(node, LSTM_Node):
                    node.wf.detach_()
                    node.wf = torch.from_numpy(rnn_info[i])
                    node.wf.requires_grad_(True) 
                    i+=1
                    node.uf.detach_()
                    node.uf = torch.from_numpy(rnn_info[i])
                    node.uf.requires_grad_(True) 
                    i+=1
                    node.wi.detach_()
                    node.wi = torch.from_numpy(rnn_info[i])
                    node.wi.requires_grad_(True) 
                    i+=1
                    node.ui.detach_()
                    node.ui = torch.from_numpy(rnn_info[i])
                    node.ui.requires_grad_(True) 
                    i+=1
                    node.wo.detach_()
                    node.wo = torch.from_numpy(rnn_info[i])
                    node.wo.requires_grad_(True) 
                    i+=1
                    node.uo.detach_()
                    node.uo = torch.from_numpy(rnn_info[i])
                    node.uo.requires_grad_(True) 
                    i+=1
                    node.wg.detach_()
                    node.wg = torch.from_numpy(rnn_info[i])
                    node.wg.requires_grad_(True) 
                    i+=1
                    node.ug.detach_()
                    node.ug = torch.from_numpy(rnn_info[i])
                    node.ug.requires_grad_(True) 
                    i+=1
