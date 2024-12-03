Train with SLURM
================

These instructions assume you access the cluster via ``ssh`` (including using the VSCode remote connect plugin), and will use ``sbatch`` to start training jobs.
Mostly, this is the same as the Quickstart instructions, but you probably have to live with whichever versions of python and CUDA are already installed on the nodes you're going to use.

All of these instructions are run on the cluster

1. Get code ``git clone --depth 1 https://altitude.otago.ac.nz/rocnet/rocnet-example.git``
2. Replace the ``{{PLACEHOLDER}}`` values in ``cluster-utils/slurm-check-*.sh`` files match your environment
3. Run ``sbatch cluster-utils/slurm-check-cuda.sh`` to figure out which version of CUDA is running on your target SLURM partition

   - You'll get a ``slurm-{JOB-NUMBER}.out`` file which will contain the output of ``nvidia-smi`` that includes a bunch of stuff about the GPU and also the version of CUDA that is installed.
   - The job might not start immediately, if you don't get the ``.out`` file within a few seconds you can check pending jobs using ``squeue --me`` which will list running and pending jobs and their statuses

4. Modify ``setup.sh`` so that the pytorch ``--index_url`` is for the correct version of CUDA (unstructions in ``setup.sh``)
5. Run ``setup.sh`` to get all the prerequisites
6. Run ``sbatch cluster-utils/slurm-check-pytorch.sh``, and check that the ``.out`` file reports the correct python and pytorch versions, and doesn't have any errors.
7. Download the data using either ``wget https://share.sjmd.dev/rocnet/example-data.zip`` or ``curl -O https://share.sjmd.dev/rocnet/example-data.zip`` (approx 1.3GB). It contains:

   1. ``laz`` - a colletion of ``.laz`` files
   2. ``dataset`` - a dataset of tiles created from the ``.laz`` files
   3. ``weights`` - a training config file, and an example training run with a set of model weights and training progress snapshots

8. Copy the ``data`` folder inside ``example-data.zip`` to ``rocnet-example/data``
9. Copy ``cluster-utils/slurm-train.sh`` to  ``rocnet-example/data/weights`` and make sure that all of the ``{{PLACEHOLDERS}}`` are replaced with real stuff from your environment

   - Also make sure that the ``--mem`` directives is within the memory limit of the GPU that you're using
   - Consider changing the ``--time`` directive if your job is likely to take longer (or shorter)
10. From within ``rocnet-example/data/weights/`` run ``sbatch slurm-train.sh``, use ``squeue --me`` to check the status and/or wait for the ``.out`` file to appear.

   - you can use ``tail -f slurm-{JOB_NUM}.out`` to modify messages as the output of the file changes

11. Once the training run is done you need to retrieve the training run folder (probably using ``scp`` or ``rsync``), and then you can use the ``examine_*`` and ``test_*`` scripts to evaluate the result.



