import tensorflow as tf
import numpy as np

def weights_init(shape):
    '''
    Weights initialization helper function.
    
    Input(s): shape - Type: int list, Example: [5, 5, 32, 32], This parameter is used to define dimensions of weights tensor
    
    Output: tensor of weights in shape defined with the input to this function
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def bias_init(shape, bias_value=0.05):
    '''
    Bias initialization helper function.
    
    Input(s): shape - Type: int list, Example: [32], This parameter is used to define dimensions of bias tensor.
              bias_value - Type: float number, Example: 0.01, This parameter is set to be value of bias tensor.
    
    Output: tensor of biases in shape defined with the input to this function
    '''
    return tf.Variable(tf.constant(bias_value, shape=shape))


def convd2_custom(input, filter_size, number_of_channels, number_of_filters, max_pool=False, padding='SAME', 
                activation=tf.nn.relu):
    '''
    This function is used to define a convolutional layer for a network,
    
    Input(s): input - this is input into convolutional layer (Previous layer or an image)
              filter_size - also called kernel size, kernel is moved (convolved) across an image. Example: 3
              number_of_channels - how many channels the input tensor has
              number_of_filters - this is hyperparameter, and this will set one of dimensions of the output tensor from 
                                  this layer. Note: this number will be number_of_channels for the layer after this one
              max_pool - if this is True, output tensor will be 2x smaller in size. Max pool is there to decrease spartial 
                        dimensions of our output tensor, so computation is less expensive.
              padding - the way that we pad input tensor with zeros ("SAME" or "VALID")
              activation - the non-linear function used at this layer.
              
              
    Output: Convolutional layer with input parameters.
    '''
    weights = weights_init([filter_size, filter_size, number_of_channels, number_of_filters])
    bias = bias_init([number_of_filters])
    
    layer = tf.nn.conv2d(input, filter=weights, strides=[1, 1, 1, 1], padding=padding) + bias
    
    layer = activation(layer)
    
    if max_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    return layer



def dense_custom(input, input_shape, output_shape, activation=tf.nn.relu, dropout=None):
    '''
    This function is used to define a fully connected layer for a network,
    
    Input(s): input - this is input into fully connected (Dense) layer (Previous layer or an image)
              input_size - how many neurons/features the input tensor has. Example: input.shape[1]
              output_shape - how many neurons this layer will have
              activation - the non-linear function used at this layer.    
              dropout - the regularization method used to prevent overfitting. The way it works, we randomly turn off
                        some neurons in this layer
              
    Output: fully connected layer with input parameters.
    '''
    weights = weights_init([input_shape, output_shape])
    bias = bias_init([output_shape])
    
    layer = tf.matmul(input, weights) + bias
    
    if activation != None:
        layer = activation(layer)
    
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer



def flatten(layer):
    '''
    This method is used to convert convolutional output (4 dimensional tensor) into 2 dimensional tensor.
    
    Input(s): layer - the output from last conv layer in your network (4d tensor)
    
    Output(s): reshaped - reshaped layer, 2 dimensional matrix
               elements_num - number of features for this layer
    '''
    shape = layer.get_shape()
    
    elements_num = shape[1:4].num_elements()
    
    reshaped = tf.reshape(layer, [-1, elements_num])
    return reshaped, elements_num