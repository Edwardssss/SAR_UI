import os
import sys
from pathlib import Path
import random

import cv2


class DrawLabel:
    def __init__(self, image, targets):
        self.image = image
        self.targets = targets

    def draw(self):
        return self.image

    def dota_draw(self):
        for target in self.targets:
            loc = target[0]
            category = target[1]
            self.draw_bbox(loc)
            self.draw_category(loc, category)
        return self.image

    def yolo_draw(self):
        for target in self.targets:
            loc = target[0]
            category = target[1]
            self.draw_bbox_yolo(loc)
            self.draw_category(loc, category)
        return self.image

    def draw_bbox(self, loc, color=(0, 0, 255), thickness=2):
        # draw bounding box
        cv2.line(self.image, (loc[0], loc[1]), (loc[2], loc[3]), color, thickness)
        cv2.line(self.image, (loc[2], loc[3]), (loc[4], loc[5]), color, thickness)
        cv2.line(self.image, (loc[4], loc[5]), (loc[6], loc[7]), color, thickness)
        cv2.line(self.image, (loc[6], loc[7]), (loc[0], loc[1]), color, thickness)
        return self.image

    def draw_bbox_yolo(self, loc, color=(0, 0, 255), thickness=2):
        x_center = (loc[0] + loc[2] + loc[4] + loc[6]) / 4
        y_center = (loc[1] + loc[3] + loc[5] + loc[7]) / 4
        w = max(max(abs(loc[2] - loc[0]), abs(loc[4] - loc[0])),
                max(abs(loc[0] - loc[6]), abs(loc[2] - loc[4])),
                max(abs(loc[2] - loc[6]), abs(loc[4] - loc[6])))
        h = max(max(abs(loc[3] - loc[1]), abs(loc[1] - loc[7])),
                max(abs(loc[3] - loc[1]), abs(loc[1] - loc[5])),
                max(abs(loc[5] - loc[7]), abs(loc[3] - loc[5])))
        x1 = int(x_center - h / 2)
        y1 = int(y_center - w / 2)
        x2 = int(x_center + h / 2)
        y2 = int(y_center + w / 2)
        cv2.line(self.image, (x1, y1), (x2, y1), color, thickness)
        cv2.line(self.image, (x2, y1), (x2, y2), color, thickness)
        cv2.line(self.image, (x2, y2), (x1, y2), color, thickness)
        cv2.line(self.image, (x1, y2), (x1, y1), color, thickness)

    def draw_category(self, loc, category, color=(0, 0, 255), thickness=2):
        # draw category small font size
        font = cv2.FONT_HERSHEY_PLAIN
        # judge the location of the category
        if loc[1] < 10:
            cv2.putText(self.image, category, (loc[0], loc[1] + 10), font, 1, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(self.image, category, (loc[2], loc[3]), font, 1, color, thickness, cv2.LINE_AA)
        return self.image


class SRSDDataset:
    def __init__(self, root):
        self.root = root
        self.labels_loc = self.root / "data_watch/labels"
        self.images_loc = self.root / "data_watch/images"
        self.images = []
        self.labels = []
        self.load_all_image_label()

    def __getitem__(self, index):
        image_loc = self.images_loc / self.images[index]
        label_loc = self.labels_loc / self.labels[index]

        image = cv2.imread(str(image_loc))
        imagesource, gsd, targets = self.read_label(label_loc)
        image = DrawLabel(image, targets)

        return image

    def __len__(self):
        return len(self.images)

    def load_all_image_label(self):
        self.images = os.listdir(self.images_loc)
        self.labels = os.listdir(self.labels_loc)


    def read_label(self, label_loc):
        # read label file
        with open(label_loc, 'r') as f:
            lines = f.readlines()

        # imagesource = 去除：前的字符串和空格
        imagesource = lines[0].split(':')[1].strip()
        gsd = lines[1].split(':')[1].strip()
        gsd = float(gsd)

        lines_len = len(lines)
        target_nums = lines_len - 2
        targets = []

        for i in range(target_nums):
            # x1,y1,x2,y2,x3,y3,x4,y4,category,difficulty
            # loc = [x1,y1,x2,y2,x3,y3,x4,y4]
            loc = lines[i + 2].split(' ')[:8]
            loc = [int(i) for i in loc]
            category = lines[i + 2].split(' ')[8]
            difficulty = lines[i + 2].split(' ')[9]
            targets.append([loc, category, difficulty])

        return imagesource, gsd, targets


class SRSDDatset_Yolo:
    def __init__(self, SRSDDataset):
        self.SRSDDataset = SRSDDataset
        self.category_dict = self.get_category_dict()

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.SRSDDataset)

    def label_to_yolo(self):
        for label_file in self.SRSDDataset.labels:
            imagesource, gsd, targets = self.SRSDDataset.read_label(self.SRSDDataset.labels_loc / label_file)
            with open(self.SRSDDataset.train_labels_loc / label_file, 'w') as f:
                for target in targets:
                    loc = target[0]
                    category = self.category_dict[target[1]]
                    x_center = (loc[0] + loc[2] + loc[4] + loc[6]) / 4 / 1024
                    y_center = (loc[1] + loc[3] + loc[5] + loc[7]) / 4 / 1024
                    w = max(max(abs(loc[2] - loc[0]), abs(loc[4] - loc[6])),
                            max(abs(loc[2] - loc[6]), abs(loc[2] - loc[4]))) / 1024
                    h = max(max(abs(loc[3] - loc[1]), abs(loc[5] - loc[7])),
                            max(abs(loc[3] - loc[7]), abs(loc[3] - loc[5]))) / 1024
                    f.write(
                        str(category) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')

    def get_category_dict(self):
        category_dict = {}
        for label_file in self.SRSDDataset.labels:
            imagesource, gsd, targets = self.SRSDDataset.read_label(self.SRSDDataset.labels_loc / label_file)
            for target in targets:
                category = target[1]
                if category not in category_dict.keys():
                    category_dict[category] = len(category_dict)
        return category_dict


if __name__ == "__main__":
    root = os.getcwd()  # YOLOv5 root directory
    if str(root) not in sys.path:
        sys.path.append(str(root))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(root, Path.cwd()))
    dataset = SRSDDatset_Yolo(SRSDDataset(ROOT))

    # 加这句就改label了
    # dataset.label_to_yolo()
    idx = random.randint(0, len(dataset) - 1)
    print(dataset.category_dict)
    image = dataset.SRSDDataset[idx].draw()
    img2 = dataset.SRSDDataset[idx].dota_draw()
    # imshow 2 img in 1 pic
    cv2.imshow('image', image)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
