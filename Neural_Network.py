
# coding: utf-8

# In[1]:


#Import Statements
import numpy as np
import mnist_loader
import random


# In[2]:


class Network:
    #Construcutor
    def __init__(self,neurons):
        self.layers = len(neurons)
        self.neurons = neurons
        
        self.weights = []
        self.biases = []
        
        #Initialize weights
        for i in range(1,self.layers):
            #layer_weight = None
            rows  = neurons[i]
            cols = neurons[i-1]
            #Creates a numpy array with dimensions rows x cols
            #At the same time, initializes them with random normal distribution b/w 0 and 1
            
            #layer_weight = np.zeros((rows,cols))
            layer_weight = np.random.randn(rows,cols)
            self.weights.append(layer_weight)

            #layer_bias = np.zeros((rows,1))
            layer_bias = np.random.randn(rows,1)
            self.biases.append(layer_bias)


# In[3]:


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


# In[4]:


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


# In[5]:


def test(network,test_data):

    _,activations = feedforward(network,test_data[0])
    
    Output = activations[-1]
    Target = test_data[1]
    
    #Output --> (10,Number_of_training_Samples)
    #Target --> (Number_of_traing_Samples)
    
    Prediction = np.argmax(Output,axis=0)
    #Prediction --> (Number_of_training_Samples,)    
        
    return sum((Prediction == Target))


# In[6]:


def sigmoid_derivative(z):
    #print("sigmoid_derivative =",sigmoid(z)*(1-sigmoid(z)) )
    return sigmoid(z)*(1-sigmoid(z))
    


# In[7]:


def cost_gradient(output,label):
    #print("cost_gradient =",output-label)
    return output-label


# In[8]:


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


# In[9]:


def update_weights(network,X,layer_activations,delta_all_layers,alpha):
    
    #before_weight = np.array(network.weights[0],copy=True)
    #print("alpha = ",alpha)

    n_weights = network.layers-1
    for i in range(n_weights):
        
        #print("shape of delta_all_layers[",i,"]:",delta_all_layers[i].shape)
        #print("shape of layer_activations[",i,"].transpose():",delta_all_layers[i].transpose().shape)            
        if i==0:
            layer_input = X
        else:
            layer_input = layer_activations[i-1]
        
       #print("delta for layer {0} = {1}".format(i,delta_all_layers))

        dcdw = np.dot(delta_all_layers[i],layer_input.transpose())
        dcb = np.average(delta_all_layers[i],axis=1).reshape(delta_all_layers[i].shape[0],1)
        #print("shape of dcdw:",dcdw.shape)
        #print("shape of dcb:",dcb.shape)
        #print("shape of network.weights[",i,"]:",network.weights[i].shape)
        
       #print("dcdw for layer {0} = {1}".format(i,dcdw))
       #print("dcb for layer {0} = {1}".format(i,dcb))
    #     bmax_val = np.amax(network.weights[i])
    #     bmin_val = np.amin(network.weights[i])
        network.weights[i] -= alpha * dcdw
        network.biases[i] -= alpha * dcb
    #     amax_val = np.amax(network.weights[i])
    #     amin_val = np.amin(network.weights[i])


    # if amax_val==bmax_val and amin_val == bmin_val:
    #     print("No update")

    return


# In[10]:


def train_GD(network,train_data,valid_data,mini_batch_size=100,alpha=1,epochs=50):
    
    X = train_data[0] #(784,50000)
    y = train_data[1] #(10,50000)
    #print(X.shape,y.shape)
    
    alpha = alpha/float(mini_batch_size)
    for e in range(epochs):

        Xtrans = X.transpose()
        ytrans = y.transpose()
        #print("Xtrans[0].shape",Xtrans[0].shape)
        #print("ytrans[0].shape",ytrans[0].shape)
        #print("Xtrans[0].reshape(X.shape[0],1).shape",Xtrans[0].reshape(X.shape[0],1).shape)
        
        #------------------Al right----------------#
        training_list = [ [ Xtrans[i],ytrans[i] ] for i in range(X.shape[1]) ]

        random.shuffle(training_list)
        
        for k1 in range(0,len(training_list),mini_batch_size):

            #list --> subset of training_list
            training_batch = training_list[k1:k1+mini_batch_size]

            
            Xbatch = np.array([sample[0] for sample in training_batch]).transpose()
            
            ybatch = np.array([sample[1] for sample in training_batch]).transpose()
            
            #print("Dimensions of Xbathc,ybatch ->",Xbatch.shape, ybatch.shape)
            #Step 1: Calculate Output
            weighted_sums,activations = feedforward(network,Xbatch)

            #Step 2: Calculate Error at final layer
            delta_output_layer = cost_gradient(activations[-1],ybatch)*sigmoid_derivative(weighted_sums[-1])

            #Step 3: Backpropogate Error
            delta_all_layers = backprop(network,delta_output_layer,weighted_sums)

            #Step 4: Update Weights
            update_weights(network,Xbatch,activations,delta_all_layers,alpha)

        #Step 5: Validation Testing


        print("End of Epoch",e," accuracy =",(test(network,valid_data)/valid_data[1].shape[0])*100)
    
    return


# In[11]:

def main():
    train_data,valid_data,test_data = mnist_loader.load_data_wrapper()
    network = Network([784,15,10])
    train_GD(network,train_data,valid_data,mini_batch_size=10,alpha=0.5,epochs=30)
    accuracy = (test(network,test_data)/valid_data[1].shape[0])*100
    print("On Testing Data: Accuracy =",accuracy)


if __name__=='__main__':
    main()