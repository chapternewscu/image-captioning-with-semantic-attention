#!/bin/bash
# will change to different directory for different datasets
cd coco-caption
python myeval.py $1
cd ../
