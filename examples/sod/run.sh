#!/bin/bash
rm -r soln_DG_tvdRK1_200*
../../main_ns  sod.json
python3 post_sod.py soln_DG_tvdRK1_200
