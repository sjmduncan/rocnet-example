tile.toml
=========

``tile.toml`` is used by ``tile.py`` to define parameters for creating a tiled dataset

.. code-block:: toml

    grid_dim = 256
    vox_size = 0.25
    train_fraction = 0.85
    clean = true
    save_intermediate = true
    # Transforms to apply to the .laz files before tiling, of the form [tx, ty, tz, rz] which is a 3D translation and a rotation (radians) about the vertical axis
    # Note that the vertical axis is the one with the smallest extent
    # Choose translation values values that are
    # 1. not integer multiples of the voxel size
    # 2. not integer multiples of the grid_dim
    # 3. not integer multiples the leaf_dim you expect to use in the RocNet model
    transforms = [ 
        [ 0.0, 0.0, 0.0, 0.0,],
        [ 0.2, 0.2, 0.2, 0.0,],
        [ 0.6, 0.4, 0.7, 0.0,],]

