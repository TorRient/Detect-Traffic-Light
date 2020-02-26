#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset traffic_light --year 2024 --set trainval --target $DIR/../data/train.lst
#python $DIR/prepare_dataset.py --dataset traffic_light --year 2024 --set test --target $DIR/../data/val.lst --shuffle False
