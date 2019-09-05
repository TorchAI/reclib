.. RecLib documentation master file, created by
   sphinx-quickstart on Mon Aug  7 09:11:08 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

Built on PyTorch, RecLib makes it easy to design and evaluate new deep
learning models for recommender system, along with the infrastructure to
easily run them in the cloud or on your laptop.  RecLib was designed with the
following principles:

* *Hyper-modular and lightweight.* Use the parts which you like seamlessly with PyTorch.
* *Extensively tested and easy to extend.* Test coverage is above 90% and the example
  models provide a template for contributions.
* *Take object oriented design* seriously, making it easy to implement correct
  models without the pain.
* *Experiment friendly.*  Run reproducible experiments with as little as work possible.

RecLib includes reference implementations of high quality models for CTR, ad ranking and more (see https://github.com/tingkai-zhang/reclib#models).

RecLib is built and maintained by the Tingkai Zhang. The RecLib project is uniquely positioned to provide
state of the art models with high quality engineering.

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   api/reclib.commands
   api/reclib.common
   api/reclib.data
   api/reclib.models
   api/reclib.predictors
   api/reclib.modules
   api/reclib.nn
   api/reclib.service
   api/reclib.tools
   api/reclib.training
   api/reclib.pretrained




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
