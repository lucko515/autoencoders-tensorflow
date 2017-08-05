# Deep Convolutional Autoencoder in TensorFlow

The Deep Autoencoders are one that are even more accuracte in the process of reconstructing (decompressing) images than the Simple or the Stacked autoencoders. The Deep Convolutional autoencoders are even used for image denosing!

**The architecture of Deep autoencoder.*


<div style="text-align:center"><img src ="deep_af_autoencoder.png" /></div>


<br><br>
NOTE: In this project I have used Convolutional layers instead of fully connected layers but the structure is the same as we have in this image.

## Dataset

MNIST dataset is used to test the Autoencoder. You can import it by using Tensorflow built-in functions.

## Install

### &nbsp;&nbsp;&nbsp; Supported Python version
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Python version used in this project: 3.5+

### &nbsp;&nbsp;&nbsp; Libraries used

> *  [Tensorflow](http://tensorflow.org) 1.2.0
> *  [Numpy](http://www.numpy.org) 1.10.4
> *  [Matplotlib](https://matplotlib.org) 1.5.1

## Code

The main code used in this project is inside **deep_conv_autoencoder.ipynb**.

There is file called **autoencoders_helper_functions.py**, in this file you will find all helper functions (layers, preprocessing) used in the main code file.

## Run

To run this project you will need some software, like Anaconda, which provides support for running .ipynb files (Jupyter Notebook).

After making sure you have that, you can run from a terminal or cmd next lines:

`ipython notebook deep_conv_autoencoder.ipynb`

or

`jupyter notebook deep_conv_autoencoder.ipynb`

Helper code:

To run the file with helper code execute next line in the terminal:

`python autoencoders_helper_functions.py`


## License

MIT License

Copyright (c) 2017 Luka Anicin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
