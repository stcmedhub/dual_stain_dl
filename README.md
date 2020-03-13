*dual_stain_dl* contains the presented network structures and installation guides

# Inception V3

Since SurePath slides have a much higher density and somewhat different appearance of cells, a separate deep learning algorithm was developed for SurePath. We strongly enriched the training set with SurePath-based tiles and used InceptionV328 as a predefined complex network structure consisting of 54 layers 7) and Keras as a higher machine-learning software abstraction layer, allowing access to the TensorFlow backend. The “Inception V3” network (Supplemental Figure 2) was originally applied to the ILSVRC (ImageNet) classification challenge, to classify objects into one of 1000 categories 8). Due to the binary classification problem of double-stain event detection, we changed the classification layers of the network to an output of two classes (>40% likelihood for a positive tile). We manually labelled a training set of 20,625 images, see Table DL3 for details. Data preprocessing was applied to the training set by rescaling intensity values to a range from 0 to 1, applying mean subtraction and normalization by dividing each dimension by its standard deviation. The networks were trained using the Stochastic Gradient Descent optimizer (SGD) with heavy data augmentation (rotation, color, shifting) but without transfer learning, resulting in a total training set size of 330,000 images. 

## Installation
The Inception V3 model can be found here: https://github.com/stcmedhub/keras-applications/blob/master/keras_applications/inception_v3.py
A description of the Inception V3 model can be found here: https://keras.io/applications/#inceptionv3
Installation steps for Keras using tensorflow can be found here: 


# CNN4 

For ThinPrep-based dual-stain cytology slides CNN4 was developed. CNN4 has 4 convolutional layers, each with a subsequent pooling layer and two fully connected layers. The final output layer was a softmax layer (>50% likelihood for a positive tile). Dropout layers were used to avoid overfitting during training. Rectified Linear Units (ReLUs) were used as activation function. CNN4 was trained using Theano as a Python software development library supporting the development of deep-learning based algorithms. Data preprocessing was applied to the training set by rescaling intensity values to a range of [0,1], applying mean subtraction and normalization by dividing each dimension by its standard deviation. We augmented the training set by flipping or rotating the images (leaving the underlying class unchanged). See the installation steps to perform training and classifiy with CNN4.

## Installation

We recommend using `venv
<https://docs.python.org/3/library/venv.html>`_ (when using Python 3)
or `virtualenv
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
to install CNN.

1. Install python 3.4.5 
2. Install Theano http://deeplearning.net/software/theano_versions/0.8.X/install.html
3. CNN4 comes with a list of know dependencies. Install the requirements with: $ pip install -r https://github.com/stcmedhub/dual_stain_dl/blob/master/requirements.txt
4. Train CNN4 using it's fit function as described in https://nbviewer.jupyter.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb

.. note:: 
  the dependencies of CNN are currently unmaintained. However, if you follow the
  installation instructions, you should still be able to get it to
  work (namely with library versions that are outdated at this point).

For more help running Lasagne with nolearn please refer to: https://github.com/dnouri/nolearn 
Further help can be found on the Lasagne repository: https://lasagne.readthedocs.io/en/latest/

## License

See the `LICENSE.txt <LICENSE.txt>`_ file for license rights and
limitations (MIT).
