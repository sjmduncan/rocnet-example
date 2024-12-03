train.toml
==========

``train.toml`` is used by ``train.py``

.. code-block:: toml

   note = ""
   dataset_path = "../rocnet.data/test"
   max_samples = -1
   max_epochs = 300
   batch_size = 50
   verbose = false
   use_torchfold = false

   [validation]
   stop_threshold = 5
   min_stop_epochs = 30

   [learning]
   warmup_epochs = 1
   lr_warmup = 0.0001
   lr_init = 0.0015
   lr_min = 0.0001
   exp_gamma = 0.96
   exp_epoch = 10

   [snapshot]
   enabled = false
   interval = 50

   [profile]
   enabled = false
   enable_trace = false

   [model]
   note = ""
   feature_code_size = 80
   voxel_channels = 1
   classifier_hidden_size = 200
   node_channels = 64
   node_channels_internal = 128
   grid_dim = 256
   leaf_dim = 32
   has_root_encoder = true

   [model.loss_params]
   recon_cre_gamma = 0.8333
   recon_cre_scale = 6
   label_scale = 10