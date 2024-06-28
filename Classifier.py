import cv2
import torch
from torchvision import transforms
from PIL import Image
from torch import nn

TYPE_LIST = ['货船', '渔船', '其他']  # 定义类别列表


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 16x512x512
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x650
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 32x16x650
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32x16x650
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64x16x650
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64x16x650
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x8x325
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()  # 64x8x325
        )
        self.linear = nn.Sequential(
            nn.Linear(200704, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.cnn(x)
        fc_input = x.view(x.size(0), -1)
        # print(fc_input.size())
        fc_output = self.linear(fc_input)
        return fc_output


def classify(IMG, model):
    """
    分类函数
    :param IMG: PIL读入的图片
    :param model: 分类模型
    :return: 分类结果
    """
    transform_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    )
    img = IMG.convert("RGB")  # 如果传入的图像是RGB图像,则将这一句话注释;如果传入的是灰度图,则保留
    img_ = transform_valid(img).unsqueeze(0)  # 填充思维
    outputs = model(img_)
    print(outputs)
    _, indices = torch.max(outputs, 1)
    print(indices)
    class_res = TYPE_LIST[indices.item()]
    return class_res


# 使用示例
# sar_classifier = MyNet()
# PATH = '.\\class_best.pth'
# # 如果电脑为CPU模式则使用下面一行
# sar_classifier.load_state_dict(torch.load(PATH, map_location=torch.device('cuda:0')))
# # 如果电脑有CUDA则使用下面一行
# # sar_classifier.load_state_dict(torch.load(PATH))
# IMG_PATH = '/Users/fox/Downloads/FUSAR_Ship_Processed_version3/train/Cargo/Ship_C01S07N1135.tiff'
# img = Image.open(IMG_PATH)
# class_res = classify(img, sar_classifier)
#
# print(class_res)
