from torch.nn.functional import tanh, relu, prelu, leaky_relu, sigmoid, elu, selu
from torch.nn.init import uniform, normal, eye, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal

def linear(x):
    return x

def activation(func_a):
    """Activation function wrapper
    """
    return eval(func_a)

def init_wrapper(init='xavier_uniform'):
    return eval(init)
