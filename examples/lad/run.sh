#!/bin/bash
rm -r soln_fr_false_0*
../../main_lad example.json | tee example.log
python3 post.py