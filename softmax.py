import numpy as np

class Softmax:

    def __init__(self,input_len,nodes):
        self.weights = np.random.randn(input_len,nodes) / input_len
        self.biases = np.zeros(nodes)
    
    def forward(self,input):

        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len,nodes = self.weights.shape
        
        totals = np.dot(input,self.weights) + self.biases
        self.last_totals = totals

        # Shift totals by subtracting the max for numerical stability
        shift_totals = totals - np.max(totals)
        exp = np.exp(shift_totals)
        return exp / np.sum(exp, axis=0)
    
    def backprop(self, d_L_d_out,learn_rate):
        
        d_L_d_t = d_L_d_out
    
        # Backprop through linear layer
        d_L_d_w = self.last_input[:, np.newaxis] @ d_L_d_t[np.newaxis]
        d_L_d_b = d_L_d_t
        d_L_d_inputs = self.weights @ d_L_d_t
    
        # Update ONCE per sample
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)
        
        





