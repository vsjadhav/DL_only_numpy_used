lr=0.01 #(initial learning rate for gradient descent based algorithms)
momentum=0.1 #(momentum to be used by momentum based algorithms)
input_dim = 784
output_dim = 10
num_hidden=3 #(number of hidden layers - this does not include the 784 dimensional input layer and the 10 dimensional output layer)
sizes=[512,256,128] #(a comma separated list for the size of each hidden layer)
activation="sigmoid" #tanh #(the choice of activation function - valid values are tanh/sigmoid)
loss="ce" #sq #(possible choices are squared error[sq] or cross entropy loss[ce])
opt="adam" #(the optimization algorithm to be used: gd, momentum, nag, adam - you will be implementing the mini-batch version of these algorithms)
batch_size= 100 #(the batch size to be used - valid values are 1 and multiples of 5)
anneal=True #(if true the algorithm should halve the learning rate if at any epoch the validation loss decreases and then restart that epoch)
save_dir= ""  #r"D:\dl_iitm\ass3_backprop\"
#(the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network)
expt_dir= ""
#(the directory in which the log files will be saved - see below for a detailed description of which log files should be generated)
train= "D:\\dl_iitm\\ass3_backprop\\dataset\\fashion-mnist_train.csv"
#(path to the Training dataset)
test= "D:\\dl_iitm\\ass3_backprop\\dataset\\fashion-mnist_test.csv"
#(path to the Test dataset)
epochs = 2






import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt



def create_network(input_dim=input_dim,output_dim=output_dim,L=num_hidden,layer_sizes=sizes,b=batch_size):
    layer_sizes=[input_dim]+layer_sizes+[output_dim]
    np.random.seed(1234)
    parameters={}
    for i in range(L+1):
        parameters[f"w{i+1}"]= 0.1*np.random.randn(layer_sizes[i], layer_sizes[i+1])
        parameters[f"b{i+1}"]= np.zeros((layer_sizes[i+1],1))

    additions = {}
    for i in range(L+1):
        additions[f"a{i+1}"]= np.zeros((layer_sizes[i+1], 1))

    layer_output = {}
    for i in range(L+1):
        layer_output[f"h{i+1}"]= np.zeros((layer_sizes[i+1], 1))

    gradients = {}
    gradients_wrt_layer_io={}
    for i in range(L+1):
        gradients[f"gw{i+1}"]= np.zeros((layer_sizes[i], layer_sizes[i+1]))
        gradients[f"gb{i+1}"]= np.zeros((layer_sizes[i+1],1))
        gradients_wrt_layer_io[f"gh{i+1}"]= np.zeros((layer_sizes[i+1], 1))
        gradients_wrt_layer_io[f"ga{i+1}"]= np.zeros((layer_sizes[i+1], 1))
    
    network_output= np.zeros((layer_sizes[L+1], 1))
    
    return parameters,additions,gradients,gradients_wrt_layer_io,layer_output,network_output,layer_sizes


def sigmoid(x):
    """sigmoid"""
    a,b = x.shape
    l=np.zeros((a,b))
    for row_a in range(a):
        for column_b in range(b):
            i= x[row_a][column_b]
            l[row_a][column_b]= (1/(1+(np.exp(-i))))
    return np.array(l)

def tanh(x):
    """tanh"""
    a,b = x.shape
    l=np.zeros((a,b))
    for row_a in range(a):
        for column_b in range(b):
            i= x[row_a][column_b]
            l[row_a][column_b]= np.tanh(i)
    return np.array(l)

def softmax(x):
    """softmax"""
    a,b = x.shape
    l=np.zeros((a,b))
    for row_a in range(a):
        for column_b in range(b):
            i= x[row_a][column_b]
            l[row_a][column_b]= np.exp(i)
    
    total= l.sum(axis=0)
    for row_a in range(a):
        for column_b in range(b):
            l[row_a][column_b]/= total[column_b]
    return np.array(l)


# df = pd.read_csv(train)
data = np.array(pd.read_csv(train))
x_train= data[:, 1:]/255 #(60000, 784)
y= data[:, 0]  #(60000,)

parameters,additions,gradients,gradients_wrt_layer_io,layer_output,network_output,layer_sizes = create_network()


def feed_forward(p=parameters,a=additions,h=layer_output,L=num_hidden,layer_sizes=layer_sizes,
                activation=sigmoid,output_activation=softmax,x= x_train[:1,:]):

    x= x.reshape(1,input_dim) #(1,784)

    for i in range(L+1):
        if i==0 :
            a[f"a{i+1}"]= (p[f"w{i+1}"].T).dot(x.T) + (p[f"b{i+1}"]) #(layer_size[i+1],784)*(784,1) + (layer_size[i+1],1)
            h[f"h{i+1}"]= activation(a[f"a{i+1}"])
        elif i==L :
            a[f"a{i+1}"]= (p[f"w{i+1}"].T).dot(h[f"h{i}"]) + (p[f"b{i+1}"]) #(layer_size[i+1],layer_size[i])*((layer_size[i],1))
            h[f"h{i+1}"]= output_activation(a[f"a{i+1}"])
        else:
            a[f"a{i+1}"]= (p[f"w{i+1}"].T).dot(h[f"h{i}"]) + (p[f"b{i+1}"]) #(layer_size[i+1],layer_size[i])*((layer_size[i],1))
            h[f"h{i+1}"]= activation(a[f"a{i+1}"])

    # print(a["a1"])
    # print(h["h1"])
    # print(a["a1"].shape)
    # print(h["h1"].shape)
    # print(a["a2"].shape)
    # print(a["a3"].shape)
    # print(a["a4"].shape)
    # print(a["a2"])
    # print(h["h2"])
    # print(40*"#")
    # print(a["a3"])
    # print(40*"#")
    # print(a["a4"])
    # print(40*"#")
    return h[f"h{L+1}"] #(10,1)
    
# network_output= feed_forward()
# print(network_output)

def one_hot(x):
    size = len(x)
    vocab= []
    for i in x:
        if i not in vocab:
            vocab.append(i)
    vocab_size = len(vocab)
    
    y = np.zeros((size,vocab_size))
    for i in range(size):
        y[i, x[i]] = 1
    return np.array(y).reshape((size,vocab_size)), vocab_size

y_train, num_classes= one_hot(y) #(60000,10)

def ce(o=network_output, desired=y_train):
    """ce"""
    d= desired.T
    o= o.T
    loss = 0
    for i in range(num_classes):
        loss += -(d[0][i]*(np.log(o[0][i])))
    return loss

def sq(o=network_output, desired=y_train):
    """sq"""
    d= desired.T
    o= o.T
    loss = 0
    for i in range(num_classes):
        loss += (((d[0][i])-(o[0][i]))**2)
    return loss

def loss_calculator(o=network_output, d=y_train, loss=ce):
    loss = loss(o,d)
    return loss

# print(f"loss: {loss_calculator()}")

def common_grad_calculator(one_hot_output,output_activation,network_output,loss):
    output_activation_name = output_activation.__doc__
    loss_name= loss.__doc__
    if (output_activation_name == "softmax") & (loss_name=="ce"):
        rtrn = (-(one_hot_output-network_output))
        # print(type(rtrn))
        # print(rtrn.shape)
        return rtrn
    else: 
        #TODO: write code for other cases
        pass
        
def derivative_of_activation(x,activation):
    activation_name = activation.__doc__
    a,b = x.shape
    derivatives=np.zeros((a,b))
    one = np.ones((a,b))
    if activation_name=="sigmoid" :
        derivatives = sigmoid(x)*(one- sigmoid(x))
        return derivatives
    else:pass
        

def backprop(p=parameters,a=additions,grad=gradients,grad_io=gradients_wrt_layer_io,h=layer_output,L=num_hidden,
            x=x_train,y=y_train,activation=sigmoid,output_activation=softmax,loss=ce,network_output=network_output):
    # desired_output= y_train[:b,:]
    
    grad_io[f"ga{L+1}"] = common_grad_calculator(y,output_activation,network_output,loss) #(outpt_dim,1)
    for i in reversed(range(1,L+1)):
        grad[f"gw{i+1}"] += h[f"h{i}"].dot(grad_io[f"ga{i+1}"].T)
        grad[f"gb{i+1}"] += grad_io[f"ga{i+1}"]

        grad_io[f"gh{i}"] = p[f"w{i+1}"].dot(grad_io[f"ga{i+1}"]) #grad wrt layer output(h)
        grad_io[f"ga{i}"] = grad_io[f"gh{i}"]*(derivative_of_activation(grad_io[f"ga{i}"],activation)) #grad wrt layer input(a)
    grad[f"gw{1}"] += x.dot(grad_io[f"ga{1}"].T)
    grad[f"gb{1}"] += grad_io[f"ga{1}"]


def gradient_descent(X=x_train,Y=y_train,p=parameters,a=additions,grad=gradients,grad_io=gradients_wrt_layer_io,
                    h=layer_output,b=batch_size,L=num_hidden,activation=sigmoid,output_activation=softmax,loss=ce,
                    lr=lr,epochs=epochs):
    err=0
    k=0
    while k<epochs:
        j=0
        for n,sample in enumerate(zip(X,Y)):
            x,y = sample
            x= x.reshape((layer_sizes[0], 1))
            y= y.reshape((layer_sizes[L+1], 1))
            network_output = feed_forward(p=p,a=a,h=h,L=L,activation=activation,output_activation=output_activation,x= x)
            err+= loss_calculator(o=network_output, d=y, loss=sq)
            backprop(p=p,a=a,grad=grad,grad_io=grad_io,h=h,L=L,x=x,y=y,activation=activation,
            output_activation=output_activation,loss=loss,network_output=network_output)
            if (n+1)%b == 0 :
                for i in range(L+1):
                    p[f"w{i+1}"]= p[f"w{i+1}"] - (lr* grad[f"gw{i+1}"]) 
                    p[f"b{i+1}"]= p[f"b{i+1}"] - (lr* grad[f"gb{i+1}"])
                    gradients[f"gw{i+1}"]= np.zeros((layer_sizes[i], layer_sizes[i+1]))
                    gradients[f"gb{i+1}"]= np.zeros((layer_sizes[i+1],1))
                print(f"batch {j+1} of epoch {k+1} complete")
                print(f"average squared batch loss: {err/b}")
                err=0
                j+=1
            else: pass
        k+=1


gradient_descent()

test_data = np.array(pd.read_csv(test))
x_test= test_data[:, 1:]/255 #(10000, 784)
y1= test_data[:, 0]  #(10000,)
y_test= one_hot(y1)

def test(X=x_test,Y=y_test,L=num_hidden):
    err = 0
    for n,sample in enumerate(zip(X,Y)):
            x,y = sample
            x= x.reshape((layer_sizes[0], 1))
            y= y.reshape((layer_sizes[L+1], 1))
            network_output = feed_forward(x= x)
            err+= loss_calculator(o=network_output, d=y, loss=sq)
    ex,dim= X.shape
    print(f"test squared err(average) is: {err/ex}")

test()