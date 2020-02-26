# Detect-Traffic-Light with YOLOv3
Using Gluoncv detect traffic light with 4 class ["traffic light", "auxiliary left light", "auxiliary right light", "train light"]

# Link pretrain model
https://drive.google.com/drive/folders/1TQ8OsNaUdQqGY51nAofguZMSCDqwaf2E?usp=sharing

# Use test
Download pretrain then run: python traffic_light.py

# How to train
You need format data follow kind of data Pascal VOC
Paste annotates into ./data/traffic_light/annotations and paste images into ./data/traffic_light/images
run: python preprocess.py
run: ./tools/prepare_pascal.sh
run train: python3 tip_yolo.py --network darknet53 --dataset voc --gpus 0 --batch-size 8 -j 16 --log-interval 100 --lr-decay-epoch 160,180 --epochs 200 --syncbn --warmup-epochs 4

# Reference
MXNET GLUONCV
https://github.com/apache/incubator-mxnet
https://github.com/dmlc/gluon-cv
# Thank for document Gluoncv
https://gluon-cv.mxnet.io/tutorials/index.html
