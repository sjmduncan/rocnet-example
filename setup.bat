python -m venv .venv
call .venv/Scripts/activate.bat

pip3 install numpy open3d==0.19.0 toml easydict matplotlib laspy[lazrs,laszip] torchinfo

@REM Get the correct URL for your version of CUDA here: https://pytorch.org/get-started/locally/
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

@REM Install rocnet from the git repository
pip3 install git+https://github.com/sjmduncan/rocnet.git