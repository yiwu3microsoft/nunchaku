#!/bin/bash
# Define the versions for Python, Torch, and CUDA
python_versions=("3.10" "3.11" "3.12" "3.13")
torch_versions=("2.5" "2.6")
cuda_versions=("12.4")

# Loop through all combinations of Python, Torch, and CUDA versions
for python_version in "${python_versions[@]}"; do
  for torch_version in "${torch_versions[@]}"; do
    for cuda_version in "${cuda_versions[@]}"; do
      bash scripts/build_linux_wheel.sh "$python_version" "$torch_version" "$cuda_version"
    done
  done
done