#!/bin/bash
workdir=$1
python tools/dist_test.py $workdir/pillar*.py --work_dir $1 --checkpoint $workdir/latest.pth
