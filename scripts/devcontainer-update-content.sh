#!/usr/bin/env bash

set -ef -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=$( dirname "${SCRIPT_DIR}" )

$SCRIPT_DIR/generate-conda-envs.py --python 3.10 --ctk 11.8 --os linux --compilers --openmpi

conda env create -n legate -f environment-test-linux-py3.10-cuda11.8-compilers-openmpi.yaml

echo "conda activate legate" >> ~/.bashrc

__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda env list

conda activate legate

conda env list

echo "Make sure pynvml is installed:"
python -c "import pynvml"

# $BASE_DIR/install.py --cuda
