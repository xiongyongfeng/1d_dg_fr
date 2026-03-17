#!/bin/bash
rm -r soln_DG_tvdRK1_200*
../main_ns_k2  ../../examples/sod/sod.json
python3 ../../examples/sod/post_sod.py soln_DG_tvdRK1_200
