
# coding: utf-8

# In[1]:


#Import Statements

import numpy as np


# In[2]:


def weighted_sum(X,W,B):
    #y = wtranspose.x + biases
    print("Dimensions of X,W,B:",X.shape,W.shape,B.shape)
    Y = X.dot(W) + B
    return Y


# In[3]:


def forward_pass(X,network,Weighted_Sums,Inputs):
    W = network.weights[0]
    B = network.biases[0]
    Inputs.append(X)
    Y = weighted_sum(X,W,B)
    Weighted_Sums.append(Y)
    O = sigmoid(Y)
    for i in range(1,len(network.weights)):
        W = network.weights[i]
        B = network.biases[i]
        Inputs.append(O)
        #Calculate weighted sum, wTx+B
        Y = weighted_sum(O,W,B)
        Weighted_Sums.append(Y)
        O = sigmoid(Y)
    #Final Output = maximum value from output classes (10 in the digit recognizer case)
    #print(O)
    #O = np.argmax(O,axis=1)+1
    return O


# In[4]:


#Backpropogate the error from output layer..
def backprop(network,error_output_layer,Weighted_Sums):

    #Create a list of length --> network.weights and initialize it with None
    error = [None]*len(network.weights)
    error[-1] = error_output_layer
    
    #Goes from 2nd last layer to 2nd layer
    
    #4layers
    #error[3] --> output
    #len(nw.wets) --> 3
    #3-1 --> 2
    #for i in range(2,0) --> i ka values --> [2,1]
    print("network.weights.len = ",len(network.weights))
    
    #print("some_str",some_variable,some_other,var,"end_str")
    #print("abc"+str(some_var))
    
    #len -> 2, then i = [1]
    for i in range(len(network.weights)-2,-1,-1):
        print("***Backprop Error for layer :",i)
        
        #Backprop error
        #δl=((wl+1)Tδl+1)⊙σ′(zl),
        
        #1st iteration ---> i=2nd last, i+1=last/output
        print("Dimensions of weights[i+1],error[i+1]",network.weights[i+1].shape,error[i+1].shape)
    

        #Dimensions of weights[i+1],error[i+1] (28, 10) (5, 10)
        something = error[i+1].dot(np.transpose(network.weights[i+1])) 
        #something---> should be (5,28)
        
        
        print("Dimensions of something,weighted_sums[i]",something.shape,Weighted_Sums[i].shape)
        # Dimensions of something,weighted_sums[i] (28,) (5, 28)
        
        
        error[i] = something * sigmoid_gradient(Weighted_Sums[i]) 
        
        #error[0]--> (,)
        # (x,y) hadamard (5,28) --> (5,28)
        
    #print("Error = ",error)
    return error


# In[5]:


def update_weights(network,Inputs,error,alpha):
    
    for i in range(len(network.weights)):

        #Inputs contains an extra element X at the begining, hence i
        #print(i,error[i]) #-->0,None, error[0]--None
        
        # dw = al-1 . errorl
        dw = np.transpose(Inputs[i]).dot(error[i])

        #Inputs[i] --> (5,784)
        #error[i] --> (5,28)
        # dw --> (784,28)
        #Inputs --> list of all inputs. Inputs[0]--> X Inputs[1] --> O1
        #Incoming Input --> to a Computational Neuron with some weights to that input

        #error[i] --> for 5 training samples, you are getting change in biases
        db = np.average(error[i],axis=0)
        """
            [
                [1,2,3,4,5],
                [2,3,5,6,7]
            
            ]
        
        
        """
        network.weights[i] += alpha * dw
        
        print("Dimensions of biases,db",network.biases[i].shape,db.shape)
        #Dimensions of biases,db (28,) (5, 28)
        network.biases[i]  += alpha * db

    return


# In[6]:


def train(network,X,t,alpha=0.25,epochs=100):
    Inputs = []
    Weighted_Sums = []
    for i in range(epochs):
        #Calculate output, while storing the weighted_sums and input at each layer
        O = forward_pass(X,network,Weighted_Sums,Inputs)
        
        #Calculate error in output layer
        error_output_layer = calc_error(Weighted_Sums,O,t)
        
        display_error = np.average(O-t)
        #print(display_error)
        
        print("Epoch,",i,": Error = "+str(display_error))
        
        #Update weights
        error = backprop(network,error_output_layer,Weighted_Sums)
        
        update_weights(network,Inputs,error,alpha)
        
    return


# In[7]:


#Returns the error in output layer...
def calc_error(Weighted_Sums,O,t):
    
    #partial-derivative -->dL/d(o) --> do
    
    do = -(t/O + (1-t)/(1-O))
    
    #t --> (5,)
    #O --> (5,)
    #do --> (5,)
    
    #sigma-dash(ZsubL) --> sig_grad
    sig_grad = sigmoid_gradient(Weighted_Sums[-1])
    #--->
    print("---Shape of last Weighted Sums layer = ",Weighted_Sums[-1].shape)
    
    #numpy array consisting of errors for each neuron in the output layer
    print("+++Shape of do,sig_grad",do.shape,sig_grad.shape)
    #[1,2,3,4,5] dot [[1,0,0,0,],[],...[]]
    #print("sig_grad = ",sig_grad)
    #print("do = ",do)
    error_in_output_layer = do * sig_grad
    #(5,10) hadamard (5,10) --> (5,10)
    
    #should return np array with dims (5,10)
    print("---Shape of Error in output_layer = ",error_in_output_layer.shape)
    return error_in_output_layer


# In[8]:


def sigmoid_gradient(z):
    return np.exp(-z)/((1.0 + np.exp(-z))**2)


# In[9]:


#Activation Function
def ReLU(z):
    return np.maximum(z,0,z)


# In[10]:


def sigmoid(z):
    return 1.0/(np.exp(-z) + 1.0)


# In[11]:


class Network:
    def __init__(self,layers):
        if len(layers)<2:
            print("Cannot create Neural Network with less than 2 layers")
            return
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(1, len(layers)):
            rows = layers[i-1] #784
            cols = layers[i] #28
            layer_weight = np.random.random((rows,cols))
            layer_bias = np.random.random(layers[i])
            self.weights.append(layer_weight)
            self.biases.append(layer_bias)


# In[12]:


my_net = Network([784,28,10])
#my_net.weights


# In[13]:


X = np.random.random((5,784))


# In[14]:


t = np.array([
              [0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0]
            ])


# In[15]:


train(my_net,X,t)

