import os
import sys
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
import cv2
from random import shuffle

annotationpaths = sorted(glob('./annotations/*.xml'))
num_train_samples = int(0.9 * len(annotationpaths))
print('num_train_samples: ', num_train_samples)
shuffle(annotationpaths)
trainpaths = annotationpaths[:num_train_samples]
testpaths = annotationpaths[num_train_samples:]
annotationpaths = [trainpaths, testpaths]
suffices = ['_trainval.txt', '_test.txt']

for i, suffix in enumerate(suffices):
    for j, annotationpath in enumerate(annotationpaths[i]):
        print('annotationpath : ',annotationpath)
        tree = ET.parse(annotationpath)
        annotation = tree.getroot()
        filename = annotation.find('filename')
        path = annotation.find('path')

        # Check if image.shape is valid
        size = annotation.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)

        # Image old path
        image_raw_path = os.path.join('./images', filename.text)

        image = cv2.imread(image_raw_path)
        assert image.ndim == 3
        assert image.shape[0] > 0 and image.shape[1] > 0 and image.shape[2] >= 3 and image.shape[2] <= 4

        if width <= 0 or height <= 0 or depth != 3:
            print('Image {} (new name: {}.xml) with invalid shape. Take shape from image!'.format(filename.text, j))
            size.find('height').text = str(image.shape[0])
            size.find('width').text = str(image.shape[1])
            size.find('depth').text = str(3)

        # Modify folder tag
        folder = annotation.find('folder')
        folder.text = 'JPEGImages'

        # Modify filename
        filename.text = '%.6d' % ((num_train_samples * i) + j) + '.jpg'

        # Modify path
        path.text = os.path.abspath(os.path.join('./JPEGImages', filename.text))

        # Convert image to jpg and move to new place
        cv2.imwrite(path.text, image[:,:,:3])

        # Create text files
        objects = ['traffic light', 'auxiliary left light', 'auxiliary right light', 'train light']
        checkob = ['traffic light', 'auxiliary left light', 'auxiliary right light', 'train light']
        for object in  annotation.findall('object'):
            name = object.find('name')
            if name.text not in checkob:
                print('Annotation: {} had object with invalid class name!!!'.format(os.path.basename(annotationpath)))
                raise ValueError
            try:
                objects.remove(name.text)
            except:
                print('Multiple objects with class {} in {}'.format(name.text, image_raw_path))
                continue
            difficult = object.find('difficult')
            if int(difficult.text) == 0:
                with open(os.path.join('./ImageSets/Main', name.text + suffix), 'a') as f:
                    f.write(filename.text.split('.')[0] + ' 1\n')
            elif int(difficult.text) == 1:
                with open(os.path.join('./ImageSets/Main', name.text + suffix), 'a') as f:
                    f.write(filename.text.split('.')[0] + ' 0\n')
            else:
                raise ValueError('difficult contained strange value: {}'.format(difficult))
        for object in objects:
            with open(os.path.join('./ImageSets/Main', object + suffix), 'a') as f:
                f.write(filename.text.split('.')[0] + ' -1\n')
        with open('./ImageSets/Main/' + suffix[1:], 'a') as f:
            f.write(filename.text.split('.')[0] + '\n')
        tree.write(os.path.join('./Annotations/', filename.text.split('.')[0] + '.xml'))
