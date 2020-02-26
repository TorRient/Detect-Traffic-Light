#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
python prepare_dataset.py --dataset eastsea --year 2021 --set trainval --target $DIR/../data/train.lst
python prepare_dataset.py --dataset eastsea --year 2021 --set test --target $DIR/../data/val.lst --shuffle False
