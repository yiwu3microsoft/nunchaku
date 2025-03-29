param (
    [string]$PYTHON_VERSION,
    [string]$TORCH_VERSION,
    [string]$CUDA_VERSION,
    [string]$MAX_JOBS = ""
)

# Check if TORCH_VERSION is 2.5 or 2.6 and set the corresponding versions for TORCHVISION and TORCHAUDIO
if ($TORCH_VERSION -eq "2.5") {
    $TORCHVISION_VERSION = "0.20"
    $TORCHAUDIO_VERSION = "2.5"
    Write-Output "TORCH_VERSION is 2.5, setting TORCHVISION_VERSION to $TORCHVISION_VERSION and TORCHAUDIO_VERSION to $TORCHAUDIO_VERSION"
}
elseif ($TORCH_VERSION -eq "2.6") {
    $TORCHVISION_VERSION = "0.21"
    $TORCHAUDIO_VERSION = "2.6"
    Write-Output "TORCH_VERSION is 2.6, setting TORCHVISION_VERSION to $TORCHVISION_VERSION and TORCHAUDIO_VERSION to $TORCHAUDIO_VERSION"
}
else {
    Write-Output "TORCH_VERSION is not 2.5 or 2.6, no changes to versions."
}

# Conda 环境名称
$ENV_NAME = "build_env_$PYTHON_VERSION_$TORCH_VERSION"

# 创建 Conda 环境
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# 安装依赖
conda install -y ninja setuptools wheel pip
pip install --no-cache-dir torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url "https://download.pytorch.org/whl/cu$($CUDA_VERSION.Substring(0,2))/"

# 设置环境变量
$env:NUNCHAKU_INSTALL_MODE="ALL"
$env:NUNCHAKU_BUILD_WHEELS="1"
$env:MAX_JOBS=$MAX_JOBS

# 进入当前脚本所在目录并构建 wheels
Set-Location -Path "$PSScriptRoot\.."
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

python -m build --wheel --no-isolation

# 退出 Conda 环境
conda deactivate
conda remove -y -n $ENV_NAME --all
Write-Output "Build complete!"
