import numpy as np
import  scipy
class NeuralNetwork :

    #initialize a network:
    def __init__(self,input_node=3,hidden_node=3,output_node=3,learningrate=0.3):
        self.inode = input_node
        self.onode = output_node
        self.hnode = hidden_node

        #learning rate:
        self.learn_r = learningrate

        #weights from i node to j node are w_ij
        self.wih = np.random.normal(0.0,pow(self.hnode,-0.5),(self.hnode,self.inode))
        self.who = np.random.normal(0.0,pow(self.onode,-0.5),(self.onode,self.hnode))

        self.activation_function = lambda x : 1/(1+scipy.exp(-x))
        pass

    def train(self,input,target):
        inputs = np.array(input,ndmin=2).T
        targets = np.array(target,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_error = targets-final_outputs
        hidden_error = np.dot(self.who.T,output_error)

        #refresh who & wih
        self.who += self.learn_r*np.dot((output_error * final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.learn_r*np.dot((hidden_error*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        pass

    def query(self,input):
        inputs =  np.array(input,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #print(final_outputs)
        return final_outputs


