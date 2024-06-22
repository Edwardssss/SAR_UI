"""
文件格式转换，HRSID数据集转换为YOLO格式
路径：
data
    - HRSID
        - train.txt
        - test.txt
HRSID
    - annotations
        - train2017.json
        - test2017.json
    - JPEGImages
    - train
        - images
        - labels
    - test
        - images
        - labels
    generate_label.py
运行generate_label.py，将会生成data路径下的train.txt和test.txt，以及train和test文件夹，分别存放训练集和测试集的图片和标签
同时添加了cv代码查看images和bbox， DrawImages类中的draw_images方法, DrawImages(gl).draw_images(index)查看第index张图片
"""

import json
import os
import cv2
from ultralytics.utils.plotting import save_one_box

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


class GenerateLabel:
    def __init__(self, json_file, current_path):
        self.json_file = json_file
        self.current_path = current_path
        self.parent_path = os.path.dirname(current_path)
        self.data = read_json(json_file)
        # self.categories = self.data['categories']['name']
        self.dataset_path = os.path.join(self.parent_path, 'data/HRSID')
        self.yaml_path = os.path.join(self.parent_path, 'data/HRSID.yaml')

    def __len__(self):
        return len(self.data['images'])

    def load_image(self, index):
        image = self.data['images'][index]
        return image

    def load_annotation(self, index):
        annotation = self.data['annotations'][index]
        return annotation

    def generate_dataset(self):
        # mkdir data/HRSID
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        # generate train txt
        if "train" in self.json_file:
            with open(os.path.join(self.dataset_path, 'train.txt'), 'w') as f:
                for i in range(len(self.data['images'])):
                    image = self.load_image(i)
                    f.write('HRSID\\train\\images\\' + image['file_name'] + '\n')
            # mkdir train in current_path
            if not os.path.exists(os.path.join(self.current_path, 'train')):
                os.makedirs(os.path.join(self.current_path, 'train'))
                os.makedirs(os.path.join(self.current_path, 'train', 'images'))
                os.makedirs(os.path.join(self.current_path, 'train', 'labels'))
            # move images from JPEGImages to train/images
            for i in range(len(self.data['images'])):
                image = self.load_image(i)
                image_file = os.path.join(self.current_path, 'JPEGImages', image['file_name'])
                if os.path.exists(image_file):
                    new_image_file = os.path.join(self.current_path, 'train', 'images', image['file_name'])
                    os.rename(image_file, new_image_file)
                # generate labels in train/labels
                # find all annotaions image_id = image['id']
                ann = []
                for j in range(len(self.data['annotations'])):
                    if self.data['annotations'][j]['image_id'] == image['id']:
                        ann.append(self.data['annotations'][j])
                label_file = os.path.join(self.current_path, 'train', 'labels',
                                          image['file_name'].replace('.jpg', '.txt'))
                with open(label_file, 'w') as f:
                    height = image['height']
                    width = image['width']
                    for a in ann:
                        category_id = a['category_id'] - 1
                        bbox = a['bbox']
                        x_center = bbox[0] + bbox[2] / 2
                        y_center = bbox[1] + bbox[3] / 2
                        x_center /= width
                        y_center /= height
                        bbox_width = bbox[2] / width
                        bbox_height = bbox[3] / height
                        f.write(f'{category_id} {x_center} {y_center} {bbox_width} {bbox_height}\n')

        if "test" in self.json_file:
            with open(os.path.join(self.dataset_path, 'test.txt'), 'w') as f:
                for i in range(len(self.data['images'])):
                    image = self.load_image(i)
                    f.write('HRSID\\test\\images\\' + image['file_name'] + '\n')
            # mkdir test in current_path
            if not os.path.exists(os.path.join(self.current_path, 'test')):
                os.makedirs(os.path.join(self.current_path, 'test'))
                os.makedirs(os.path.join(self.current_path, 'test', 'images'))
                os.makedirs(os.path.join(self.current_path, 'test', 'labels'))
            # move images from JPEGImages to test/images
            for i in range(len(self.data['images'])):
                image = self.load_image(i)
                image_file = os.path.join(self.current_path, 'JPEGImages', image['file_name'])
                if os.path.exists(image_file):
                    new_image_file = os.path.join(self.current_path, 'test', 'images', image['file_name'])
                    os.rename(image_file, new_image_file)
                # generate labels in test/labels
                ann = []
                for j in range(len(self.data['annotations'])):
                    if self.data['annotations'][j]['image_id'] == image['id']:
                        ann.append(self.data['annotations'][j])
                label_file = os.path.join(self.current_path, 'test', 'labels',
                                          image['file_name'].replace('.jpg', '.txt'))
                with open(label_file, 'w') as f:
                    height = image['height']
                    width = image['width']
                    for a in ann:
                        category_id = a['category_id'] - 1
                        bbox = a['bbox']
                        x_center = bbox[0] + bbox[2] / 2
                        y_center = bbox[1] + bbox[3] / 2
                        x_center /= width
                        y_center /= height
                        bbox_width = bbox[2] / width
                        bbox_height = bbox[3] / height
                        f.write(f'{category_id} {x_center} {y_center} {bbox_width} {bbox_height}\n')


class DrawImages:
    def __init__(self, GenerateLabel):
        self.GenerateLabel = GenerateLabel
        self.current_path = GenerateLabel.current_path

    def draw_images(self, index):
        image = self.GenerateLabel.load_image(index)
        ann = []
        for j in range(len(self.GenerateLabel.data['annotations'])):
            if self.GenerateLabel.data['annotations'][j]['image_id'] == image['id']:
                ann.append(self.GenerateLabel.data['annotations'][j])
        image_file = os.path.join(self.current_path, 'train\\images', image['file_name'])
        image = cv2.imread(image_file)
        for a in ann:
            bbox = a['bbox']
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    current_path = os.getcwd()
    # 获取当前路径的上一级
    parent_path = os.path.dirname(current_path)
    print(parent_path)

    json_file = './annotations/train2017.json'
    json_file_test = './annotations/test2017.json'
    data = read_json(json_file)
    data_test = read_json(json_file_test)
    print(data.keys())
    print('info:', data['info'])
    print('licenses:', data['licenses'])
    print('categories:', data['categories'])
    print('type:', data['type'])
    print('images:', data['images'][0])
    print('annotations:', data['annotations'][0])
    print(type(data['annotations'][0]['bbox']))
    print('annotations:', data['annotations'][0]['category_id'])

    # caculate data length
    print(len(data['images']))
    print(len(data['annotations']))

    gltest = GenerateLabel(json_file_test, current_path)
    gltest.generate_dataset()

    gl = GenerateLabel(json_file, current_path)

    gl.generate_dataset()
    # for i in range(len(data['images'])):
    #     if len(data['annotations'][i]['bbox']) != 4:
    #         print('error')
    #         break

    image_path = os.path.join(current_path, 'JPEGImages')
    # caculate the *.jpg number
    print(len([lists for lists in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, lists))]))
