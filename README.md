*dual_stain_dl* contains the presented network structures and installation guides

# Inception V3

# CNN4 

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
