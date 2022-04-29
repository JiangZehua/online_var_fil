python -m pip install tqdm scipy pandas matplotlib

# for machines don't have GPUs
conda install pytorch torchvision -c pytorch

# for machines have GPUs
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# for 3090
# conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

python -m pip install hydra-core --upgrade