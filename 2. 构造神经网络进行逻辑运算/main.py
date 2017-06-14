import numpy as np

class Neuron(object):
    '''sigmoid神经元'''
    def __init__(self):
        self.weights = 0
        self.bias = 0
        self.inputs = 0
        self.output = 0
        self.len_inputs = -1
    
    def sigmoid(self, x):
        '''sigmoid实现 f = 1/(1 + e^(-x))'''
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, h):
        '''sigmoid导数 f' = f(1-f) '''
        tmp = self.sigmoid(h)
        return tmp(1 - tmp)
    
    def forward(self, inputs):
        '''利用inputs构造神经元，输出激活值。
        
        Args:
        	inputs: shape=(n, ) n表示输入的参数数量
        Return:
        	一个常量
        '''
        self.inputs = inputs
        if self.len_inputs == -1:
            # 使用均值为0，方差为0.1的值来初始化权重
            self.len_inputs = inputs.shape[0]
            self.weights = np.random.normal(
                loc=.0, scale=.1, size=self.len_inputs)
        self.output = self.sigmoid(np.dot(self.weights.T, self.inputs))
        return self.output


class Layer(object):
    '''实现神经网络的一个层'''
    def __init__(self, num_node):
        '''输入当前层的神经元个数，构造神经网络的一个层。'''
        self.num_node = num_node
        self.neurons = [Neuron() for _ in range(num_node)]
        self.inputs = 0
        self.output = 0
    
    def calc(self, x):
        self.inputs = x
        self.output = np.array([n.forward(x) for n in self.neurons])
        return self.output
    
    
class ANNNet(object):
    def __init__(self, layers_detail=[4, 4, 1]):
        self.layers_detail = layers_detail
        self.layers = [Layer(n) for n in self.layers_detail]
        
    def fit(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.calc(output)
        return output

def main():
    ann = ANNNet()
    res = ann.fit(np.array([1, 2, 3]))
    return res

if __name__ == '__main__':
    print(main())