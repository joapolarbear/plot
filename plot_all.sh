#!/bin/bash

set -x

python3 all_128_gpu.py
python3 clock_sync.py
python3 cluster_scale.py
python3 motivation.py
python3 replayer_error.py
python3 tensor_fusion_search_rst.py
python3 tensor_fusion_search_rst2.py
python3 tensor_grp2itertime.py
python3 tsfs_opfs_rst.py
python3 tsfs_opfs_rst2.py
python3 tsfs_opfs_rst3.py
python3 xla_search_rst.py
python3 xla_search_rst2.py
python3 xla_search_rst3.py