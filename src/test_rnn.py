from rnn import RNN
import sklearn
import sklearn.datasets # To generate the dataset
import numpy as np

def test_rnn(rnn, ) -> None:
    # Generate the dataset
    X, t = sklearn.datasets.make_circles(
        n_samples=100, shuffle=False, factor=0.3, noise=0.1)
    T = np.zeros((100, 2)) # Define target matrix
    T[t==1, 1] = 1
    T[t==0, 0] = 1

    weights = []
    for e in rnn.edges:
        weights.append(e.weight)

    '''
    for n in rnn.nodes.values():
        weights.append(n.wf)
        weights.append(n.uf)
        weights.append(n.wi)
        weights.append(n.ui)
        weights.append(n.wo)
        weights.append(n.uo)
        weights.append(n.wg)
        weights.append(n.ug)
    '''

    eps = 0.0001
    for w in weights:
        print("MAIN: EDGE ID:", rnn.edges[0].id)
        print(type(w))
        min_w = w.value() - eps
        max_w = w.value() + eps
        org_w = w.value()       

        rnn.do_epoch(X, T, "mse", True)
        grad_param = w.get_gradient()

        print("****")
        print("min_w", min_w)
        print("weight before:", w.value())
        w.set_value(min_w)
        print("weight after:", w.value())
        rnn.do_epoch(X, T, "mse", True)
        grad_min = w.get_gradient()
        print("----")

        w.set_value(max_w)
        rnn.do_epoch(X, T, "mse", True)
        grad_max = w.get_gradient()

        grad_num = (grad_max - grad_min) / (2*eps)

        print(type(grad_min))
        print(type(grad_param))
        if not np.isclose(grad_num, grad_param):
            raise ValueError((
                    f'Numerical gradient of {grad_num:.6f} '
                    'is not close to the backpropagation'
                    f'gradient of {grad_param:.6f}!'))
        w.set_value(org_w)
    print('No gradient errors found')
        

def create_rnn():
    rnn = RNN(paths=[], centeroids_clusters=None, lags=1)
    rnn.build_fully_connected_rnn(input_names=['i1', 'i2'], output_names=['o1', 'o2'], lags=1, hid_layers=1, hid_nodes=2) 
    return rnn


rnn = create_rnn()
test_rnn(rnn)
