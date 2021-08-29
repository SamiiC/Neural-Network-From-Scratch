import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import gradient
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')


data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

#transpose data to make it easier to work with for each training set where the rows are training examples and columns are the 784 pixel values

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev =data_dev[1:n]
X_dev = X_dev / 255.



data_train = data[1000:m].T 
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape



def init_params():
    # biases (constant value added to each node of following layer)
    W1 = np.random.rand(10, 784) - 0.5 # layer 1 (784 pixels input -> 10 nodes)

    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5 # layer 2
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# non-linear activation function to add complexity to hidden layers and make the NN work
def ReLU(Z):
    return np.maximum(Z, 0)

#This is an activation function, converting output layer to probabilities
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
   # set inital vales to nodes in first layer: Z1 =W1 *X + b1
   Z1 = W1.dot(X) + b1
   A1 = ReLU(Z1)
   Z2 = W2.dot(A1) + b2
   A2 = softmax(Z2)  #A2 values to probabilities
   return Z1, A1, Z2, A2


#encodes training examples to make it easier for algorithm to 
#get a better prediction

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
   # find gradient of ReLU function for backpropagation
   return Z > 0


#takes previously made pred and runs it backwards through the nn
#This determines which weights and biases contributed the most to the error
#So it modifies those accordingly

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#this decides how to tune weights & biases to reduce error by the highest amount 
def grad_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2



W1, b1, W2, b2 = grad_descent(X_train, Y_train, 0.5, 400)


#graphing
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



x = int(input("Enter an integer:"))
while x !=0:
    test_prediction(x,W1,b1,W2,b2)
    x = int(input("Enter an integer:"))






#test on other dataset we didn't train on
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))

