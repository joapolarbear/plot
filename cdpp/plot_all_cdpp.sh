#!/bin/bash

set -x

python3 cdpp_finetune.py
python3 cdpp_tir.py
python3 cmpp_finetune.py
python3 cmpp_tir.py
python3 end2end_cross_device.py
python3 end2end_cross_model.py
python3 pe.py
python3 sche_search.py
