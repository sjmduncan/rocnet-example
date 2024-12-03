RocNet
==============

The main points of entry are:

1. rocnet.trainer.Trainer
2. rocnet.rocnet.RocNet
3. rocnet.rocnet.RocnetFile

rocnet.dataset
---------------------

.. code-block:: python

   DEFAULT_METADATA = {
      # Either 'tileset' or 'lazset'
      "type": "tileset",
      # For stuff derived from datasets like modelnet, where each category
      # can be loaded as an independent dataset
      "recurse": False,
      # Maximum grid dim, must be a power of two, models which use smaller grid_dim
      # can still load this dataset but the data will be down-sampled
      "grid_dim": 256,
      # Voxel size for used to preserve scale
      "vox_size": 1.0,
      # What percentage of the tiles are allocated for testing
      "train_fraction": 0.85,
      # Either 'Occupancy' or 'RGB'
      "vox_attrib": "Occupancy"
   }

.. automodule:: rocnet.dataset
   :members:
   :undoc-members:

rocnet.file
------------------

.. automodule:: rocnet.file
   :members:
   :undoc-members:

rocnet.model
-------------------

.. automodule:: rocnet.model
   :members:
   :undoc-members:

rocnet.octree
--------------------

.. automodule:: rocnet.octree
   :members:
   :undoc-members:

rocnet.rocnet
--------------------

.. code-block:: python

   DEFAULT_CONFIG = {
      "feature_code_size": 80,
      # Set to 0 for occupancy only, or 3 for RGB data
      "voxel_channels": 0,
      "classifier_hidden_size": 200,
      "node_channels": 64,
      "node_channels_internal": 128,
      "grid_dim": 256,
      "leaf_dim": 32,
      # Set to false to flatten the root node representation instead of using the root encoder.
      # If set to false then feature_code_size must equal 64 * node_channels
      "has_root_encoder": True,
      "loss_params": {
         "recon_cre_gamma": 0.8333,
         "recon_cre_scale": 6,
         "label_scale": 10,
      },
   }


.. automodule:: rocnet.rocnet
   :members:
   :undoc-members:

rocnet.trainer
---------------------

.. code-block:: python

   DEFAULT_CONFIG = {
      "dataset_path": "../rocnet.data/test",
      # Set to less than zero to use all available data
      "max_samples": -1,
      "max_epochs": 300,
      "batch_size": 50,
      "verbose": False,
      "use_torchfold": False,
      "validation": {  # Validation runs on the test dataset
         "stop_threshold": 5,
         "min_stop_epochs": 30,
      },
      "learning": { # Learning rates and gamma decay
         "warmup_epochs": 1,
         "lr_warmup": 0.0001,
         "lr_init": 0.0015,
         "lr_min": 0.0001,
         "exp_gamma": 0.96,
         "exp_epoch": 10,
      },
      "snapshot": { # Save weights and training state snapshots at regular intervals
         "enabled": False,
         "interval": 50,
      },
      "profile": { # Enable pytorch profiling
         "enabled": False,
         "enable_trace": False,
      }
   }


.. automodule:: rocnet.trainer
   :members:
   :undoc-members:

rocnet.utils
-------------------

.. automodule:: rocnet.utils
   :members:
   :undoc-members:

