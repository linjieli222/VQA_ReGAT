## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

## This code is modified by Linjie Li from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa
## GNU General Public License v3.0

## Process data

python3 tools/create_dictionary.py
python3 tools/compute_softscore.py
