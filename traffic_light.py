from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import cv2
import trafficLightColor
import numpy as np
import time
import mxnet as mx
import argparse

# Argument
parser = argparse.ArgumentParser(description='Trafic Light')
parser.add_argument('--input_video', type=str, default='1.mp4',
                        help="Path input video")
parser.add_argument('--network', type=str, default='yolo3_darknet53_custom',
                        help="Model using for detect")
parser.add_argument('--weights', type=str, default='yolo3_darknet53_custom_0070_0.9182.params',
                        help="Weights of model detect")                           
parser.add_argument('--thresh',default=0.7,
                        help="threshold for detect")

args = parser.parse_args()

classes = ["traffic light", "auxiliary left light", "auxiliary right light", "train light"]
net = model_zoo.get_model(args.network, classes=classes, pretrained_base=True)
net.load_parameters(args.weights, ctx=mx.cpu(0))

cap = cv2.VideoCapture(args.input_video)
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
def split_label(bounding_boxes, scores, class_IDs):
    '''
    Each label, get score max
    '''
    bbox_traffic = -1
    scores_traffic = -1
    bbox_left = -1
    scores_left = -1
    bbox_right = -1
    scores_right = -1
    bbox_train = -1
    scores_train = -1
    for i, label in enumerate(class_IDs):
        if label == 0 and isinstance(bbox_traffic, int): # Get max
            bbox_traffic = bounding_boxes[i]
            scores_traffic = scores[i]
        elif label == 1 and isinstance(bbox_left, int): # Get max
            bbox_left = bounding_boxes[i]
            scores_left = scores[i]
        elif label == 2 and isinstance(bbox_right, int): # Get max
            bbox_right = bounding_boxes[i]
            scores_right = scores[i]
        elif label == 3 and isinstance(bbox_train, int): # Get max
            bbox_train = bounding_boxes[i]
            scores_train = scores[i]
        if not isinstance(bbox_traffic, int) and not isinstance(bbox_left, int) and not isinstance(bbox_right, int) and not isinstance(bbox_train, int):
            break
    return bbox_traffic, scores_traffic, bbox_left, scores_left, bbox_right, scores_right, bbox_train, scores_train

def detect_light(frame, thresh_traffic, thresh_left, thresh_right, thresh_train):
    '''
    Return bbox and score > thresh
    '''
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = data.transforms.presets.ssd.transform_test(frame, short=512)
    # Get params
    class_IDs, scores, bounding_boxes = net(rgb_nd)
    bounding_boxes = bounding_boxes.asnumpy().reshape(-1,4)
    scores = scores.asnumpy().reshape(-1,)
    class_IDs = class_IDs.asnumpy().reshape(-1,)

    # Get scores label correspond
    bbox_traffic, scores_traffic, bbox_left, scores_left, bbox_right, scores_right, bbox_train, scores_train = split_label(bounding_boxes, scores, class_IDs)

    bboxs = np.zeros((4,4))
    scoress = [0,0,0,0]
    # Find bbox for traffic light first
    if scores_traffic >= thresh_traffic:
        bboxs[0] = bbox_traffic
        scoress[0] = scores_traffic
    # Find bbox for auxiliary light
    if scores_left >= thresh_left:
        bboxs[1] = bbox_left
        scoress[1] = scores_left
    if scores_right >= thresh_right:
        bboxs[2] = bbox_right
        scoress[2] = scores_right
    # Find bbox for train light
    if scores_train >= thresh_train:
        bboxs[3] = bbox_train
        scoress[3] = scores_train
    
    return frame, bboxs, scoress

def overlap(bbox_1, bbox_2):
    '''
    Caculate overlap
    -----------
    Return area overlap
    '''
    x1_min = bbox_1[0] if bbox_1[0] > 0 else 0
    y1_min = bbox_1[1] if bbox_1[1] > 0 else 0
    x1_max = bbox_1[2] if bbox_1[2] > 0 else 0
    y1_max = bbox_1[3] if bbox_1[3] > 0 else 0
    x2_min = bbox_2[0] if bbox_2[0] > 0 else 0
    y2_min = bbox_2[1] if bbox_2[1] > 0 else 0
    x2_max = bbox_2[2] if bbox_2[2] > 0 else 0
    y2_max = bbox_2[3] if bbox_2[3] > 0 else 0
    #check overlap
    if x1_min <= x2_min < x1_max \
        and y1_min <= y2_min < y1_max:
        return (min(x1_max, x2_max) - x1_min)*(min(y1_max, y2_max) - y1_min)
    elif x2_min <= x1_min < x2_max \
        and y2_min <= y1_min < y2_max:
        return (min(x2_max, x1_max) - x2_min)*(min(y2_max, y1_max) - y2_min)
    else:
        return 0

def validate_bbox(list_bbox, thresh = 0.85):
    max_valid = 0
    index = 0
    for i, bbox_1 in enumerate(list_bbox):
        frency = 0
        x1_min = bbox_1[0] if bbox_1[0] > 0 else 0
        y1_min = bbox_1[1] if bbox_1[1] > 0 else 0
        x1_max = bbox_1[2] if bbox_1[2] > 0 else 0
        y1_max = bbox_1[3] if bbox_1[3] > 0 else 0
        area = (x1_max - x1_min)*(y1_max - y1_min)
        for bbox_2 in list_bbox:
            if np.array_equal(bbox_1, bbox_2):
                continue
            if overlap(bbox_1, bbox_2)/area >= thresh:
                frency +=1
        if frency > max_valid:
            max_valid = frency
            index = i
    # print(max_valid)
    if max_valid >= 2:
        return list_bbox[index]
    else:
        return np.zeros(4,)

bboxs = np.zeros((4,4))
bboxs_trf = []
bboxs_lef = []
bboxs_rig = []
bboxs_tra = []
scores = [0,0,0,0]
thresh_traffic = args.thresh
thresh_left = args.thresh
thresh_right = args.thresh
thresh_train = 0.9 #Default
# Read until video is completed
count = -1
list_time_detect_auxi = np.array([0,12]) # 8frame/1s
list_time_detect_traffic = np.array([0,3,6,9,15,21,51,93,150,957]) #8frame/1s
list_time_detect_train = np.array([198,207,222,312,441]) # Detect định kỳ
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count == 960: #120*8
        count = 0
    if count %3 != 0:
        continue
    print('count: ',count)
    #--------------------------------------------------------------------------------------
    # Detect within list seconds (VALIDATE)
    frame_t = frame
    print(list_time_detect_auxi)
    print(list_time_detect_traffic)
    print(list_time_detect_train)
    if count in list_time_detect_auxi or count in list_time_detect_traffic or count in list_time_detect_train:
        if count in list_time_detect_auxi: # Find bbox auxi
            list_time_detect_auxi = np.delete(list_time_detect_auxi,0)
            # list_time_detect_auxi.insert(0,time_detect + 1)
            frame, bboxs_tmp, scores = detect_light(frame_t, thresh_traffic, thresh_left, thresh_right, thresh_train)
            if not np.array_equal(bboxs_tmp[1], np.zeros(4,)): #left
                bboxs_lef.append(bboxs_tmp[1])
            if not np.array_equal(bboxs_tmp[2], np.zeros(4,)): #right
                bboxs_rig.append(bboxs_tmp[2])
            if not np.array_equal(bboxs_tmp[3], np.zeros(4,)): #train
                bboxs_tra.append(bboxs_tmp[3])
        if count in list_time_detect_traffic: # Find bbox traffic
            list_time_detect_traffic = np.delete(list_time_detect_traffic,0)
            frame, bboxs_tmp, scores = detect_light(frame_t, thresh_traffic, thresh_left, thresh_right, thresh_train)
            if not np.array_equal(bboxs_tmp[0], np.zeros(4,)): #traffic
                bboxs_trf.append(bboxs_tmp[0])
        if count in list_time_detect_train: # Find bbox traffic
            list_time_detect_train = np.delete(list_time_detect_train,0)
            if len(list_time_detect_train) == 0:
                list_time_detect_train = np.array([198, 207, 222, 312, 441]) # Detect định kỳ
            frame, bboxs_tmp, scores = detect_light(frame_t, thresh_traffic, thresh_left, thresh_right, thresh_train)
            if not np.array_equal(bboxs_tmp[0], np.zeros(4,)): #traffic
                bboxs_tra.append(bboxs_tmp[0])
    else:
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, frame = data.transforms.presets.ssd.transform_test(frame, short=512)
    
    # Validate bbox
    # print('tesst: ', bboxs_trf)
    print('-----------------')
    if len(bboxs_trf) == 1:
        pass
        # bboxs[0] = bboxs_trf[0]
    elif len(bboxs_trf) > 2:
        #check overlap
        bboxs[0] = validate_bbox(bboxs_trf)

    if len(bboxs_lef) == 1:
        pass
        # bboxs[1] = bboxs_lef[0]
    elif len(bboxs_lef) > 2:
        #check overlap
        bboxs[1] = validate_bbox(bboxs_lef)

    if len(bboxs_rig) == 1:
        pass
        # bboxs[2] = bboxs_rig[0]
    elif len(bboxs_rig) > 2:
        #check overlap
        bboxs[2] = validate_bbox(bboxs_rig)

    if len(bboxs_tra) == 1:
        pass
        # bboxs[3] = bboxs_tra[0]
    elif len(bboxs_tra) > 2:
        #check overlap
        bboxs[3] = validate_bbox(bboxs_tra)
    #--------------------------------------------------------------------------------------
    for i, bbox in enumerate(bboxs):
        if np.array_equal(bbox,np.zeros(4,)):
            continue

        xmin = int(bbox[0]) if bbox[0] > 0 else 0
        ymin = int(bbox[1]) if bbox[1] > 0 else 0
        xmax = int(bbox[2]) if bbox[2] > 0 else 0
        ymax = int(bbox[3]) if bbox[3] > 0 else 0
        # Display the resulting frame
        
        traffic_light = frame[ymin:ymax, xmin:xmax]
        color_tmp = trafficLightColor.estimate_label(traffic_light)
        if color_tmp == 'red':
            print('RED')
            if i == 0 and np.array_equal(bboxs[1],np.zeros(4,)) and np.array_equal(bboxs[2],np.zeros(4,))\
                and len(list_time_detect_auxi) <= 6 and len(list_time_detect_traffic) > 0: # Bắt đầu chuyển RED thì detect luôn auxi
                list_time_detect_auxi = np.insert(list_time_detect_auxi, 0, round(count+24)%960)
                list_time_detect_auxi = np.insert(list_time_detect_auxi, 0, round(count+12)%960)
                list_time_detect_auxi = np.insert(list_time_detect_auxi, 0, round(count+6)%960)
                list_time_detect_auxi = np.insert(list_time_detect_auxi, 0, round(count+3)%960)
                list_time_detect_auxi, indices = np.unique(list_time_detect_auxi, return_inverse=True) # Sort and remove duplicate
            cv2.putText(frame,"RED", (int((xmax+xmin)/2)-70,int((ymax+ymin)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,0,0), 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        elif color_tmp == "yellow":
            # print('YELLOW')
            cv2.putText(frame,"YELLOW", (int((xmax+xmin)/2)-100,int((ymax+ymin)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,255,0), 2)
        elif color_tmp == 'black':
            # print('BLACK')
            cv2.putText(frame,"BLACK", (int((xmax+xmin)/2)-90,int((ymax+ymin)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,0), 2)
        else:
            # print('GREEN')
            cv2.putText(frame,"GREEN", (int((xmax+xmin)/2)-90,int((ymax+ymin)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()