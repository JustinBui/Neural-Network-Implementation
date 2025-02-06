#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DO NOT MODIFY THIS PART

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[2]:


# DO NOT MODIFY THIS PART

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# In[3]:


# DO NOT MODIFY THIS PART

model = nn.Sequential(nn.Linear(2, 2, bias=True), nn.Sigmoid(),
                      nn.Linear(2, 1, bias=True), nn.Sigmoid())

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1) 
losses = []

for step in range(2001): 
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()        
    losses.append(cost.item())

plt.plot(losses)
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# In[4]:


# DO NOT MODIFY THIS PART

Y_hat = model(X)
print('Predicted: ', Y_hat)
print('Actual: ', Y)


# *** Write down your answer here ***
# 
# Formula of bias for layer 2
# 
# $$\frac{\partial L}{\partial b^2} = (\hat{y_i}-y_i)$$
# 
# Formula of weights of layer 2
# $$\frac{\partial L}{\partial \vec{w}^{\,(2)}} = \begin{bmatrix}
# \frac{\partial L}{\partial w^2_{11}}\\
# \frac{\partial L}{\partial w^2_{21}}
# \end{bmatrix} =
# \begin{bmatrix}
# (\hat{y_i}-y_i)h^1_1 \\
# (\hat{y_i}-y_i)h^1_2
# \end{bmatrix}
# $$
# 
# Formula of biases of layer 1
# $$\frac{\partial L}{\partial \vec{b}^{\,(1)}} =\begin{bmatrix}
# \frac{\partial L}{\partial b^1_1}\\
# \frac{\partial L}{\partial b^1_2}
# \end{bmatrix} = 
# \begin{bmatrix}
# (\hat{y_i}-y_i)w^2_{11}h^1_1(1-h^1_1) \\
# (\hat{y_i}-y_i)w^2_{21}h^1_2(1-h^1_2)
# \end{bmatrix}
# $$
# 
# Formula of weights of layer 1
# $$
# \frac{\partial L}{\partial \vec{W}^{\,(1)}} =\begin{bmatrix}
# \frac{\partial L}{\partial w^1_{11}} & \frac{\partial L}{\partial w^1_{12}} \\
# \frac{\partial L}{\partial w^1_{21}} & \frac{\partial L}{\partial w^1_{22}}  
# \end{bmatrix} = 
# \begin{bmatrix}
# (\hat{y_i}-y_i) w^2_{11}h^1_1(1-h^1_1)x_1 & 
# (\hat{y_i}-y_i) w^2_{21}h^1_2(1-h^1_2)x_1 \\
# (\hat{y_i}-y_i) w^2_{11}h^1_1(1-h^1_1)x_2 &
# (\hat{y_i}-y_i) w^2_{21}h^1_2(1-h^1_2)x_2
# \end{bmatrix}
# $$

# # Implementing my Neural Network From Scratch

# In[5]:


import numpy as np
import matplotlib.pyplot as plt


# Loss Function: Binary Cross-Entropy 
# $$-\frac{1}{N}\sum_i^N (y_i*log(\hat{y_i}))+(1-y_i)*(log(1-\hat{y_i}))$$
# 
# Derivative of Binary Cross-Entropy (**For a single data point**): 
# $$ 
# \frac{\partial L}{\partial \hat{y_i}}=-\frac{1}{N}*\frac{y_i-\hat{y}_i}{\hat{y_i}(1-\hat{y_i})}
# $$

# In[6]:


# Helper math functions

def binaryCrossEntropyLoss(y_pred, y_exp):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # Preventing any logs by 0
    return -np.mean(y_exp * np.log(y_pred) + (1 - y_exp) * (np.log(1-y_pred)))

def derivativeBinaryCrossEntropyLoss(y_pred, y_exp):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (-1) * ((y_exp - y_pred) / (y_pred * (1-y_pred)))

def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
def derivativeSigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


# **Neural Network class implemented**

# In[7]:


class Layer:
    def __init__(self, num_input, num_output):        
        self.num_input = num_input # Number of inputs current layer has (Number of neurons in previous layer)
        self.num_output = num_output # Number of outputs current layer has (Number of neurons in current layer)
        # NOTE: The num_input of layer n must match the num_output of layer n - 1
        
        self.weights = np.random.randn(num_input, num_output)
        self.biases = np.zeros((1, num_output))
        
    def preactivate(self, features):
        self.preactivate_layer_input = features
        #print(f'FORWARD PROP: Preactivate layer input {self.preactivate_layer_input} ({self.preactivate_layer_input.shape})')
        
        return np.dot(features, self.weights) + self.biases
    
    def postactivate(self, preactivate_values):
        self.postactivate_layer_input = preactivate_values
        #print(f'FORWARD PROP: Postactivate layer input {self.postactivate_layer_input} ({self.postactivate_layer_input.shape})')
        
        return sigmoid(preactivate_values)

class SequentialFromScratch:
    def __init__(self, *args): # *args allows you to pass in as many layers as possible
        self.layers = [l for l in args]
        self.depth = len(self.layers)
        self.errors = []
        
        print('SequentialFromScratch(')
        for num, layer in enumerate(self.layers):
            print(f'  {layer.__class__.__name__} {num}: ({layer.num_input}, {layer.num_output})')
        print(')')
    
    def forwardpropagate(self, datapoint):
        for i, l in enumerate(self.layers): # Taking 1 datapoint, and propagating that all the way forward through all layers
            preactivate_values = l.preactivate(datapoint)
            postactivate_values = l.postactivate(preactivate_values)
            datapoint = postactivate_values
        
        return postactivate_values # y_pred
    
    def backpropagate(self, output_error, learning_rate):
        for i, l in enumerate(reversed(self.layers)): # Going backwards            
            # POSTACTIVATION LAYER: Get input_error=dL/dx for a given error=dL/dy.
            output_error = derivativeSigmoid(l.postactivate_layer_input) * output_error
                        
            # PREACTIVATION LAYER: Compute dL/dw, dL/db for a given output_error=dL/dy. Gets input_error=dL/dx for the next iteration.
            input_error = np.dot(output_error, l.weights.T) 
            weights_error = np.dot(l.preactivate_layer_input.T, output_error) # Used to update weights

            # UPDATE PARAMETERS
            l.weights -= learning_rate * weights_error # w = w - (r * dL/dw), where r is the learning rate
            l.biases -= learning_rate * output_error # dL/db is always 1, therefore just take the value of layer after (error)
            
            output_error = input_error
                
    def fit(self, x_train, y_train, learning_rate, epochs, logs=False):
        N = len(y_train) # Number of instances
        cost_list = [] # Used to record all cost values to graph on a chart
        
        for i in range(epochs):
            y_predictions = []
            error_per_epoch = 0
            
            for x, y in zip(x_train, y_train): # Going 1 sample at a time in our training set
                # Forward propagation
                y_pred = self.forwardpropagate(x)
                                
                # Keep track of all y_pred values
                y_predictions.append(y_pred)
                
                error_per_epoch += binaryCrossEntropyLoss(y_pred, y)
                                                
                # Back propagation
                error_derivative = derivativeBinaryCrossEntropyLoss(y_pred, y) # dL/dY
                self.backpropagate(error_derivative, learning_rate)
            
            total_cost = error_per_epoch / N
            cost_list.append(total_cost)
            
            if logs:
                print(f'[Epoch {i + 1} / {epochs}, Loss={total_cost}]')
            
        return cost_list

    def predict(self, x):
        return self.forwardpropagate(x)


# **Training my model using 6000 epochs**

# In[8]:



neural_net = SequentialFromScratch(
    Layer(2, 2),
    Layer(2, 1)
)

X = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y = np.array([[[0]], [[1]], [[1]], [[0]]])

losses = neural_net.fit(x_train=X, y_train=y, learning_rate=0.1, epochs=6000, logs=True)


# **Predicting $\hat{y}$ for a given $\vec{x}$**

# In[13]:


y_pred = neural_net.predict(X)

print('Predicted: ', end='')
for _ in y_pred:
    print(_, end=' ')

print(f'\nEXPECTED: ', end='')
for _ in y:
    print(_, end=' ')


# **Plotting a graph showing the relationship between the epoch number and the loss.**

# In[14]:


plt.plot(losses)
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

