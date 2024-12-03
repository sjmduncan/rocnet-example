test.toml
=========

``test.toml`` is used by ``examine_training_run.py``, ``test_file.py``, and ``test_tile.py``.

.. code-block:: toml

    n_samples = 20
    use_cuda = true
    datasets = [ 
        "../rocnet.data/default",
    ]
    models = [
        "../rocnet.weights/default",
    ]
    files = [
         "../rocnet.data/raw/default/default.laz",
    ]

