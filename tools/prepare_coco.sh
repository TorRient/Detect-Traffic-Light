#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset coco --set train2017 --target $DIR/../data/train.lst
python $DIR/prepare_dataset.py --dataset coco --set val2017 --target $DIR/../data/val.lst --shuffle False
