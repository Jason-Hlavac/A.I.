import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_der(x):
    return x * (1 - x )
training_inputs = np.array([[0, 0, 1], [1,1,1], [1, 0, 1], [0, 1, 1]])
training_output = np.array([[0,1,1,0]]).T
synaptic_weights = 2 * np.random.random((3, 1)) - 1
print ("Starting synaptic weights:")
print (synaptic_weights)

for i in range(100000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_output - outputs
    adjustments = error * sigmoid_der(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)
print ("synaptic weights after training :")
print(synaptic_weights)
print("Outputs after training:")
print(outputs)
