===========
Parallelism
===========

Skorch supports model parallelism via `Dask
<https://dask.pydata.org>`_.  In this section we'll describe how to
use Dask to efficiently distribute a grid search or a randomized
search on hyper paramerers across multiple GPUs and potentially
multiple nodes.

Let's assume that you have two GPUs that you want to run a hyper
parameter search on.

The key here is using the CUDA environment variable
`CUDA_VISIBLE_DEVICES
<https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/>`_
to limit which devices are visible to our CUDA application.  We'll set
up Dask workers that, using this environment variable, each see one
GPU only.  On the Pytorch side, we'll have to make sure to set the
device to ``cuda`` when we initialize the :class:`.NeuralNet` class.

Let's run through the steps.  First, install Dask and some related packages::

  pip install dask distributed dask_searchcv

Next, assuming you have two GPUs on your machine, let's start up a
Dask scheduler and two Dask workers::

  dask-scheduler
  CUDA_VISIBLE_DEVICES=0 dask-worker 127.0.0.1:8786 --nthreads 1
  CUDA_VISIBLE_DEVICES=1 dask-worker 127.0.0.1:8786 --nthreads 1

Now, instead of importing ``GridSearchCV`` or ``RandomizedSearchCV``
from scikit-learn, import from ``dask_searchcv`` instead.  And set up
a client that connects to your scheduler before you run the search:

.. code:: python

    from dask_searchcv import GridSearchCV
    from dask.distributed import Client

    scheduler_address = '127.0.0.1:8786'
    client = Client(address=scheduler_address)

    X, y = load_my_data()
    model = get_that_model()      

    gs = GridSearchCV(
        model,
        param_grid={'net__lr': [0.01, 0.03]},
        scoring='accuracy',
        )
    gs.fit(X, y)
    print(gs.cv_results_)

You can also use Palladium to do the job.  An example is included in
the source in the ``examples/rnn_classifier`` folder.  Change in there
and run the following command, after having set up your Dask workers::

  PALLADIUM_CONFIG=dask-config.py,palladium-config.py pld-grid-search --impl dask_searchcv.GridSearchCV
