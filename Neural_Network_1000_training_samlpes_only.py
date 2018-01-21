#Import Statements
import numpy as np
import mnist_loader
import random

class Network:

    #Construcutor
    def __init__(self,neurons,initialization):
        self.layers = len(neurons)
        self.neurons = neurons
        
        self.weights = []
        self.biases = []
        
        #Initialize weights
        if initialization.lower() == "xavier":
        	self.init_xavier()

        elif initialization.lower() in ["zeros","zero"] :
        	self.init_zero()

        else:
        	self.init_random()

        return

    #Weights Initialization strategies..
    def init_xavier(self):
    	self.weights = [np.random.randn(self.neurons[i],self.neurons[i-1]) * np.sqrt(6.0 /(self.neurons[i]+self.neurons[i-1])) for i in range(1,self.layers)]
    	self.biases = [np.zeros((self.neurons[i],1)) for i in range(1,self.layers)]
    	return

    def init_zero(self):
    	self.weights = [np.zeros((self.neurons[i],self.neurons[i-1])) for i in range(1,self.layers)]
    	self.biases = [np.zeros((self.neurons[i],1)) for i in range(1,self.layers)]
    	return
    
    #Random -> normal distribution and not uniform distribution
    def init_random(self):
    	self.weights = [np.random.randn(self.neurons[i],self.neurons[i-1]) for i in range(1,self.layers)]
    	self.biases = [np.random.randn(self.neurons[i],1) for i in range(1,self.layers)]
    	return



def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
    
def cost_gradient(output,label):
    return output-label

def feedforward(network,X):
    
    #Stores the Aj values
    Layer_Activations = []
    
    #Stores the Zj values
    Weighted_Sums = []
    
    for i in range(1,network.layers):
   
        if i==1:
            A = X
        else:
            A  = Layer_Activations[-1]
        
        W = network.weights[i-1]
        B = network.biases[i-1]
        Z = np.dot(W,A) + B
        Weighted_Sums.append(Z)
        A = sigmoid(Z)
        Layer_Activations.append(A)

    return (Weighted_Sums,Layer_Activations)

def backprop(network,delta_output_layer,weighted_sums):
    #Initialize error for all layers
    #delta is the backpropogation error
    
    #Number of output layers --> first layer is an input layer, hence excluded
    n_out = network.layers-1
    
    delta_all_layers = [None]*(n_out)
    
    delta_all_layers[-1] = delta_output_layer
    
    for i in range(n_out-2,-1,-1):
        delta_all_layers[i] = np.dot(network.weights[i+1].transpose(),delta_all_layers[i+1])*sigmoid_derivative(weighted_sums[i])
            
    return delta_all_layers

def delta_output_layer(a,y,z,cost):

	delta_out = cost_gradient(a,y)

	#If Quadritic Cost or MSE is used then, multiply with sigmoid_derivative(z)
	if cost.lower in ["mse","quadratic"]:
		delta_out *= sigmoid_derivative(z)

	return delta_out

def update_weights(network,X,layer_activations,delta_all_layers,alpha):
    
    #number of weights (weighted_layers)
    n_weights = network.layers-1

    for i in range(n_weights):
        
        if i==0:
            layer_input = X
        else:
            layer_input = layer_activations[i-1]
        
        dcdw = np.dot(delta_all_layers[i],layer_input.transpose())
        dcb = np.average(delta_all_layers[i],axis=1).reshape(delta_all_layers[i].shape[0],1)
        
        network.weights[i] -= alpha * dcdw
        network.biases[i] -= alpha * dcb

    return

def test(network,test_data):

    _,activations = feedforward(network,test_data[0])
    
    Output = activations[-1]
    Target = test_data[1]
    
    #Output --> (10,Number_of_training_Samples)
    #Target --> (Number_of_traing_Samples)
    
    Prediction = np.argmax(Output,axis=0)
    #Prediction --> (Number_of_training_Samples,)    
    
    if Target.ndim>1:
        Target = np.argmax(Target,axis=0)

    return sum((Prediction == Target))

def train_SGD(network,train_data,valid_data,cost,mini_batch_size,alpha,epochs):
    
    X = train_data[0] #(784,50000)
    y = train_data[1] #(10,50000)
    
    alpha = alpha/float(mini_batch_size)
    for e in range(epochs):

        Xtrans = X.transpose()
        ytrans = y.transpose()

        training_list = [ [ Xtrans[i],ytrans[i] ] for i in range(X.shape[1]) ]

        random.shuffle(training_list)
        
        for k1 in range(0,len(training_list),mini_batch_size):

            #list --> subset of training_list
            training_batch = training_list[k1:k1+mini_batch_size]
            
            Xbatch = np.array([sample[0] for sample in training_batch]).transpose()
            
            ybatch = np.array([sample[1] for sample in training_batch]).transpose()
            
            #Step 1: Calculate Output
            weighted_sums,activations = feedforward(network,Xbatch)

            #Step 2: Calculate Error at final layer
            delta_out = delta_output_layer(activations[-1],ybatch,weighted_sums[-1],cost)

            #Step 3: Backpropogate Error
            delta_all_layers = backprop(network,delta_out,weighted_sums)

            #Step 4: Update Weights
            update_weights(network,Xbatch,activations,delta_all_layers,alpha)

        #Step 5: Validation Testing

        #Test on Training_Data
        accuracy_train_data = (test(network,train_data)/train_data[1].shape[1])*100.0 

        #Test on Validation Data
        accuracy_test_data = (test(network,valid_data)/valid_data[1].shape[0])*100.0
        
        print("End of Epoch {0}: accuracy_on(train,test) = ({1:.4},{2:.4})".format(e,accuracy_train_data,accuracy_test_data))
    
    return

def main():

    #Load Data
    train_data,valid_data,test_data = mnist_loader.load_data_wrapper()
    
    #Use only 1000 training samples
    train_data = (train_data[0].transpose()[:1000].transpose(),train_data[1].transpose()[:1000].transpose())

    #Initilaize Neural network with following number of neurons in respective layers
    network = Network([784,30,10],initialization = "xavier")

    #Train the Network using Stochastic Gradient Descent
    train_SGD(network,train_data,valid_data,cost="entropy",mini_batch_size=50,alpha=0.5,epochs=100)
    
    #Test the network on Testing Data..
    accuracy = (test(network,test_data)/valid_data[1].shape[0])*100
    
    print("On Testing Data: Accuracy =",accuracy)


if __name__=='__main__':
    main()