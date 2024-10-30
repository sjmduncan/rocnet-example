#!/bin/bash

# Workaround for how python does special things if you're on windows
# https://github.com/pypa/virtualenv/commit/993ba1316a83b760370f5a3872b3f5ef4dd904c1
python -m venv .venv
if [ -d ".venv/Scripts" ]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

# Get compatible versions of Open3D and numpy (otherwise you get silent failures and segfaults)
# https://github.com/isl-org/Open3D/issues/6970
pip3 install numpy==1.26.4 open3d==0.18.0

# Get the correct URL for your version of CUDA here: https://pytorch.org/get-started/locally/
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

pip3 install toml easydict matplotlib laspy[lazrs,laszip]

# Install rocnet from the git repository
pip3 install git+https://altitude.otago.ac.nz/rocnet/rocnet.git
