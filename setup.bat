python -m venv .venv
call .venv/Scripts/activate.bat

@REM Open3D is incredibly brittle, this makes sure you get a compatible version of 
@REM numpy otherwise you'll get silent failures and segfaults
@REM https://github.com/isl-org/Open3D/issues/6970
pip3 install numpy==1.26.4 open3d==0.18.0

@REM Get the correct URL for your version of CUDA here:
@REM https://pytorch.org/get-started/locally/
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

pip3 install toml easydict matplotlib laspy[lazrs,laszip]

@REM Install rocnet from the git repository
pip3 install git+https://altitude.otago.ac.nz/rocnet/rocnet.git