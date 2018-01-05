
# coding: utf-8

# In[1]:


#Import Statements
import mnist_loader

import numpy as np


# In[2]:


def weighted_sum(X,W,B):
    #y = wtranspose.x + biases
    #print("Dimensions of X,W,B:",X.shape,W.shape,B.shape)
    Y = X.dot(W) + B
    return Y


# In[3]:


def forward_pass(X,network,Weighted_Sums,Inputs,store=False):
    W = network.weights[0]
    B = network.biases[0]
    if store:
        Inputs.append(X)
    Y = weighted_sum(X,W,B)
    if store:
        Weighted_Sums.append(Y)
    O = sigmoid(Y)
    for i in range(1,len(network.weights)):
        W = network.weights[i]
        B = network.biases[i]
        if store:
            Inputs.append(O)
        #Calculate weighted sum, wTx+B
        Y = weighted_sum(O,W,B)
        if store:
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
    #print("network.weights.len = ",len(network.weights))
    
    #print("some_str",some_variable,some_other,var,"end_str")
    #print("abc"+str(some_var))
    
    #len -> 2, then i = 0
    for i in range(len(network.weights)-2,-1,-1):
        #print("***Backprop Error for layer :",i)
        
        #Backprop error
        #δl=((wl+1)Tδl+1)⊙σ′(zl),
        
        #1st iteration ---> i=2nd last, i+1=last/output
        #print("Dimensions of weights[i+1],error[i+1]",network.weights[i+1].shape,error[i+1].shape)
    
        #print("error[",i+1,"] =",error[i+1])
        #print("np.transpose(network.weights[",i+1,"]) =",np.transpose(network.weights[i+1]))
        #Dimensions of weights[i+1],error[i+1] (28, 10) (5, 10)
        something = error[i+1].dot(np.transpose(network.weights[i+1])) 
        #something---> should be (5,28)
        
        
        #print("Dimensions of something,weighted_sums[i]",something.shape,Weighted_Sums[i].shape)
        # Dimensions of something,weighted_sums[i] (28,) (5, 28)
        
        
        #print("something = ",something)
        #print("Weighted_Sums[",i,"] =",Weighted_Sums[i])
        error[i] = something * sigmoid_gradient(Weighted_Sums[i]) 
        #print("sigmoid_gradient(Weighted_Sums[",i,"]) =",sigmoid_gradient(Weighted_Sums[i]))
        
        #error[0]--> (,)
        # (x,y) hadamard (5,28) --> (5,28)
        
    #print("Error = ",error)
    return error


# In[5]:


def update_weights(network,Inputs,error,alpha):
    
    #print("Updating network weghts, layer :",end=" ")
    for i in range(len(network.weights)):
        #print(i+1)
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

        network.weights[i] -= alpha * dw
        # if np.array_equal(old_weights, network.weights[i]):
        # 	print("No update!")
        # else:
        # 	print("Weights are updated!")
        #print("Dimensions of biases,db",network.biases[i].shape,db.shape)
        #Dimensions of biases,db (28,) (5, 28)

        network.biases[i]  -= alpha * db

    return


# In[6]:


def train(network,X,t,mini_batch_size=10,alpha=0.5,epochs=30):
    alpha = alpha/mini_batch_size
    for i in range(epochs): 
        print("Epoch",i+1,":",end=" ")
        
        """
        
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        
        
        """
        Inputs = []
        Weighted_Sums = []
        train_set_size = 50000
        for k1 in range(0,train_set_size,mini_batch_size):
            Xbatch = X[k1:k1+mini_batch_size]
            tbatch = t[k1:k1+mini_batch_size]
            
            #Calculate output, while storing the weighted_sums and input at each layer
            O = forward_pass(Xbatch,network,Weighted_Sums,Inputs,store=True)

            #Calculate error in output layer
            error_output_layer = calc_error(Weighted_Sums,O,tbatch)

            #display_error = np.average(error_output_layer)
            #print(display_error)

            #print("Epoch",i,": Error = "+str(display_error))

            #Update weights
            error = backprop(network,error_output_layer,Weighted_Sums)
            
            update_weights(network,Inputs,error,alpha)


        test(network,validation_data[0],validation_data[1])

    return


# In[7]:


def test(network,X,t):
    # Inputs = []
    # Weighted_Sums = []

    #passing None,None, cuz no neee
    O = forward_pass(X,network,None,None)

    #print("Shape of O =",O.shape)
    #print("O=",O)    

    O = np.argmax(O,axis=1)+1
    #print("t=",t)
    #display_error = np.average(np.where(O==t, 0, 1))
    
    


    #display_error = 

    print("Error =",display_error, "%")
    


# In[8]:


#Returns the error in output layer...
def calc_error(Weighted_Sums,O,t):
    
    #partial-derivative -->dL/d(o) --> do
    
    #do = -(t/O + (1-t)/(1-O))
    
    do = O - t
    #t --> (5,)
    #O --> (5,)
    #do --> (5,)
    
    #sigma-dash(ZsubL) --> sig_grad
    sig_grad = sigmoid_gradient(Weighted_Sums[-1])
    #--->
    #print("---Shape of last Weighted Sums layer = ",Weighted_Sums[-1].shape)
    
    #numpy array consisting of errors for each neuron in the output layer
    #print("+++Shape of do,sig_grad",do.shape,sig_grad.shape)
    #[1,2,3,4,5] dot [[1,0,0,0,],[],...[]]
    #print("sig_grad = ",sig_grad)
    #print("do = ",do)
    
    error_in_output_layer = do * sig_grad
    
    #(5,10) hadamard (5,10) --> (5,10)
    
    #should return np array with dims (5,10)
    #print("---Shape of Error in output_layer = ",error_in_output_layer.shape)
    #print("Error_in_output_layer =",error_in_output_layer)
    return error_in_output_layer


# In[9]:


def sigmoid_gradient(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# In[10]:


#Activation Function
def ReLU(z):
    return np.maximum(z,0,z)


# In[11]:


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


# In[12]:


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
            layer_weight = (np.random.random((rows,cols))-0.5)*200
            layer_bias = (np.random.random(layers[i])-0.5)*200
            self.weights.append(layer_weight)
            self.biases.append(layer_bias)


# In[13]:


my_net = Network([784,28,10])
#my_net.weights


# In[14]:


# X = np.random.random((5,784))


# In[15]:


# t = np.array([
#               [0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],
#               [0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],
#               [0,1,0,0,0,0,0,0,0,0]
#             ])


# In[16]:


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
X = training_data[0]
t = training_data[1]


# In[17]:


train(my_net,X,t)


# In[18]:


print("On Test Data:",end=" ")
test(my_net,test_data[0],test_data[1])


# In[19]:



