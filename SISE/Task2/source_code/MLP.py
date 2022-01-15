from random import random
from math import exp


# sigmoidal function
def sigmoidal_function(x):
    return 1 / (1 + exp(-x))

# derivative of sigmoidal function (argument is a result of sigmoidal function!)
def derivative_of_sigmoidal_function(y):
    return y * (1 - y)

# identity function
def identity_function(x):
    return x

# derivative of identity function (argument is not important)
def derivative_of_identity_function(y):
    return 1

class ActivationFunctionNotFound(Exception):
    pass

class Layer:

    def __init__(self, no_of_inputs, no_of_outputs, activation_function, derivative_of_activation_function):
        no_of_inputs += 1  # bias
        self.no_of_inputs = no_of_inputs
        self.no_of_outputs = no_of_outputs
        self.activation_function = activation_function
        self.derivative_of_activation_function = derivative_of_activation_function
        self.neurons = [[random() for _ in range(no_of_inputs)] for _ in range(no_of_outputs)]
        self.gradients = [[0 for _ in range(no_of_inputs)] for _ in range(no_of_outputs)]

    def test(self, inputs):
        self.inputs = [1.0] + inputs  # bias
        self.activations = [self.activation_function(sum(x * w for x, w in zip(self.inputs, neuron)))
                            for neuron in self.neurons]
        return self.activations

    def learn(self, errors, learn_speed, momentum, calculate_previous=True):
        # calculate previous errors (if necessary)
        if calculate_previous:
            previous_errors = [sum(self.derivative_of_activation_function(activation) * neuron[i] * error
                                   for activation, neuron, error
                                   in zip(self.activations, self.neurons, errors))
                               for i in range(1, len(self.inputs))]
        else:
            previous_errors = []
        # calculate gradient
        gradients = []
        for activation, error, old_gradient in zip(self.activations, errors, self.gradients):
            factor = -error * self.derivative_of_activation_function(activation)
            gradients.append([factor * x + momentum * g for x, g in zip(self.inputs, old_gradient)])
        self.gradients = gradients
        # calculate new weights
        new_neurons = []
        for neuron, gradient in zip(self.neurons, self.gradients):
            new_neurons.append([w - learn_speed * g for w, g in zip(neuron, gradient)])
        self.neurons = new_neurons
        return previous_errors


class MLP:

    def __init__(self, layers_description):
        # layers description is a list of strings, where first string is
        # a number of inputs (first not processing layer) and the rest of elements
        # contains one number and one letter, where number is a number of neurons and
        # letter is either 's' for sigmoidal activation function or 'i' for identity function
        # eg. ['3', '4i', '2s']
        self.layers = []
        no_of_inputs = int(layers_description[0])
        for quantity, func_type in ((int(layer[:-1]), layer[-1]) for layer in layers_description[1:]):
            if func_type == 's':
                self.layers.append(Layer(no_of_inputs, quantity, sigmoidal_function, derivative_of_sigmoidal_function))
            elif func_type == 'i':
                self.layers.append(Layer(no_of_inputs, quantity, identity_function, derivative_of_identity_function))
            else:
                raise ActivationFunctionNotFound
            no_of_inputs = quantity

    def test(self, inputs):
        for layer in self.layers:
            inputs = layer.test(inputs)
        return inputs

    def learn(self, inputs, outputs, learn_speed, momentum=0.9):
        my_outputs = self.test(inputs)
        errors = [output - my_output for output, my_output in zip(outputs, my_outputs)]
        for layer in reversed(self.layers[1:]):
            errors = layer.learn(errors, learn_speed, momentum)
        self.layers[0].learn(errors, learn_speed, momentum, False)

    def weights_to_string(self):
        text = ""
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron:
                    text += str(weight) + ","
                text = text[:-1]
                text += "   "
            text = text[:-3]
            text += "\n"
        text = text[:-1]
        return text
