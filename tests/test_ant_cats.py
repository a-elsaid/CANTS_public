import sys
sys.path.append('..')
from ant_cants import Ant
import pytest
import numpy as np
from typing import Tuple, List
from unittest import mock
import torch
from search_space_cants import Point, RNNSearchSpace
from rnn import Node, Edge, RNN
from helper import LOSS
from data_prep import Timeseries

@pytest.fixture
def current_ant_position() -> Tuple[float]:
    '''Provid dummy ant locations'''
    x, y, w = np.random.random(3)
    l = np.random.randint(low = 0, high = 10)
    return (x,y,l,w)

@pytest.fixture
def ant() -> Ant:
    ant = Ant(sense_range = 0.2, explore_rate = 0.5, mortality = 0.2)
    ant.current_l = 2
    ant.current_x = 0.2
    ant.current_y = 0.2
    ant.current_w = 0.5
    return ant

@pytest.fixture
def search_space() -> RNNSearchSpace:
    return mock.Mock()

@pytest.fixture
def points_cluster() -> List[Point]:
    return points

@pytest.fixture
def search_space() -> RNNSearchSpace:
    space = RNNSearchSpace(lags=10)
    points  = [Point(0,0,2,0),
               Point(1,0,0,0),
               Point(1,1,0,0),
               Point(1,1,2,1),
               Point(1,0,0,1),
               Point(0,1,0,0),
               Point(0,1,0,1),
               Point(0,0,2,1)]
    for pnt in points:
        space.all_points[pnt.id] = pnt
    return space


class TestAnt:
    def test_create_point(self, ant: Ant, search_space: RNNSearchSpace):
        lag = 10
        ant.current_l = 2
        current_point = current_ant_position
        old_x = ant.current_x
        old_y = ant.current_y
        old_l = ant.current_l
        new_point = ant.create_point(search_space)
        assert ant.current_x>=0.0 and ant.current_x<1.0
        assert ant.current_y >= 0.2 and ant.current_y >=0.0 and ant.current_y <=1.0
        assert ant.current_w >=0.0 and ant.current_w<=1.0
        assert ant.current_x<=(old_x+ant.sense_range) and ant.current_x>= (old_x - ant.sense_range)
        assert ant.current_y<=(old_y+ant.sense_range) and ant.current_y>= (old_y - ant.sense_range)

    def test_move(self, ant: Ant):
        assert ant.sense_range >0.0 and ant.sense_range<0.5

    def test_ceter_of_mass(self, search_space: RNNSearchSpace, ant:Ant):
        new_point = ant.center_of_mass(search_space.all_points.values())
        assert new_point.pos_x == 0.5
        assert new_point.pos_y == 0.5
        assert new_point.pos_l == 1
        assert new_point.w == 0.5


    def test_pick_point(self, ant: Ant, search_space: RNNSearchSpace):
        pnt = Point(0.21, 0.21, 2, 0.51)
        search_space.all_points[pnt.id] = pnt
        picked_point = ant.pick_point(search_space)
        assert pnt.pos_x == picked_point.pos_x and pnt.pos_y == picked_point.pos_y and pnt.pos_l == picked_point.pos_l and pnt.w == picked_point.w
        search_space.all_points[pnt.id].pos_x = 0.31
        search_space.all_points[pnt.id].pos_y = 0.31
        search_space.all_points[pnt.id].pos_l = 6
        search_space.all_points[pnt.id].w = 0.55
        picked_point = ant.pick_point(search_space)
        assert pnt.pos_x != picked_point.pos_x and pnt.pos_y != picked_point.pos_y and pnt.pos_l != picked_point.pos_l and pnt.w != picked_point.w





@pytest.fixture
def single_node() -> Node:
    node = Node(Point(0,0,2,0), 0.2)
    node.bias = 0.1
    node_out1 = Node(Point(0,1,2,0.4), 2)
    node_out1.value = .8
    node_out1.bias = .3
    node_out2 = Node(Point(0,2,1,0.6), 3)
    node_out2.value = .6
    node_out2.bias = .4
    node.fan_out = {node_out1: Edge(node, node_out1, 0.2), node_out2: Edge(node, node_out2, 0.2)}
    return node


@pytest.fixture
def rnn2() -> RNN:
    p_in1 = Point(0.4,0,2,0.1,0,'in1')
    p_in1.id = 0

    p_in2 = Point(0,0,2,0.2,0,'in2')
    p_in2.id = 1

    p_hid1 = Point(0,0.1,2,0.2,1)
    p_hid1.id = 2

    p_hid2 = Point(0,0.2,1,0.3,1)
    p_hid2.id = 3

    p_out1 = Point(0,0.2,9,0.4,2, 'out1')
    p_out1.id = 4

    p_out2 = Point(0,0.4,9,0.6,2, 'out2')
    p_out2.id = 5

    nn = RNN([[p_in1, p_hid1, p_out1],
              [p_in1, p_hid1, p_out2],
              [p_in1, p_hid2, p_out1],
              [p_in1, p_hid2, p_out2],
              [p_in2, p_hid1, p_out1],
              [p_in2, p_hid1, p_out2],
              [p_in2, p_hid2, p_out1],
              [p_in2, p_hid2, p_out2]])
    for node in nn.nodes.values():
        node.bias = 0.0
        node.id = node.point.id
    return nn

class TestRNN:

    def test_rrnn_2(self, rnn2: RNN):
        def sig(x):
            return 1/(1+np.exp(-x))
        def dsig(x):
            return x*(1-x)
        output = rnn2.do_epoch([[0.3, 0.4]],  [[0.5, 0.7]], LOSS["mse"])
        in1 = np.float32(0.3)
        sig_in1 = sig(in1)
        wi1h1 = np.float32(0.1)
        wi1h2 = np.float32(0.1)

        in2 = np.float32(0.4)
        sig_in2 = sig(in2)
        wi2h1 = np.float32(0.2)
        wi2h2 = np.float32(0.2)

        hid1 = sig_in1 * wi1h1 + sig_in2 * wi2h1
        sig_hid1 = sig(hid1)

        hid2 = sig_in1 * wi1h2 + sig_in2 * wi2h2
        sig_hid2 = sig(hid2)

        wh1o1 = np.float32(0.2)
        wh1o2 = np.float32(0.2)

        wh2o1 = np.float32(0.3)
        wh2o2 = np.float32(0.3)

        out1 = sig_hid1 * wh1o1 + sig_hid2* wh2o1
        out2 = sig_hid1 * wh1o2 + sig_hid2* wh2o2
        sig_out1 = sig(out1)
        sig_out2 = sig(out2)
        #assert float("{:.7f}".format(rnn2.output_nodes[0].value.item())) == float("{:.7f}".format(sig_out1))
        #assert float("{:.7f}".format(rnn2.output_nodes[1].value.item())) == float("{:.7f}".format(sig_out2))

        gt1 = np.float32(0.5)
        gt2 = np.float32(0.7)
        de1 = sig_out1-gt1 
        de2 = sig_out2-gt2 
        dwh1o1 = de1 * dsig(sig_out1) * sig_hid1
        dwh1o2 = de2 * dsig(sig_out2) * sig_hid1
        dwh2o1 = de1 * dsig(sig_out1) * sig_hid2
        dwh2o2 = de2 * dsig(sig_out2) * sig_hid2
        dwi1h1 = de1 * dsig(sig_out1) * wh1o1 * dsig(sig_hid1) * sig_in1 + de2 * dsig(sig_out2) * wh1o2 * dsig(sig_hid1) * sig_in1
        dwi1h2 = de1 * dsig(sig_out1) * wh2o1 * dsig(sig_hid2) * sig_in1 + de2 * dsig(sig_out2) * wh2o2 * dsig(sig_hid2) * sig_in1
        dwi2h1 = de1 * dsig(sig_out1) * wh1o1 * dsig(sig_hid1) * sig_in2 + de2 * dsig(sig_out2) * wh1o2 * dsig(sig_hid1) * sig_in2
        dwi2h2 = de1 * dsig(sig_out1) * wh2o1 * dsig(sig_hid2) * sig_in2 + de2 * dsig(sig_out2) * wh2o2 * dsig(sig_hid2) * sig_in2
        rnn2.feedbackward()
        
        nodes = [rnn2.nodes[i] for i in range(6)]

        assert float("{:.7f}".format(dwh1o1)) == float('{:.7f}'.format(nodes[2].fan_out[nodes[4]].weight.item() - wh1o1))
        assert float("{:.7f}".format(dwh1o2)) == float('{:.7f}'.format(nodes[2].fan_out[nodes[5]].weight.item() - wh1o2))
        assert float("{:.7f}".format(dwh2o1)) == float('{:.7f}'.format(nodes[3].fan_out[nodes[4]].weight.item() - wh2o1))
        assert float("{:.7f}".format(dwh2o2)) == float('{:.7f}'.format(nodes[3].fan_out[nodes[5]].weight.item() - wh2o2))
        assert float("{:.7f}".format(dwi1h1)) == float('{:.7f}'.format(nodes[0].fan_out[nodes[2]].weight.item() - wi1h1))
        assert float("{:.7f}".format(dwi1h2)) == float('{:.7f}'.format(nodes[0].fan_out[nodes[3]].weight.item() - wi1h2))
        assert float("{:.7f}".format(dwi2h1)) == float('{:.7f}'.format(nodes[1].fan_out[nodes[2]].weight.item() - wi2h1))
        assert float("{:.7f}".format(dwi2h2)) == float('{:.7f}'.format(nodes[1].fan_out[nodes[3]].weight.item() - wi2h2))
