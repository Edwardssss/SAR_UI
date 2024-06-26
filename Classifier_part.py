import cv2
import torch
from torchvision import transforms
from PIL import Image

TYPE_LIST = ['货船', '渔船', '客船', '油船', '拖船', '其他']  # 定义类别列表


class SAR_classifier:
    """
    定义SAR_classifier
    """

    def __init__(self, pth_path='.\\class_best.pth'):
        """
        :param pth_path:传入pth文件路径作为实例化参数
        """
        self.model = torch.load(pth_path)
        # 如果报错"RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is
        # False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device(
        # 'cpu') to map your storages to the CPU." 则使用下面一行代码: self.model = torch.load(pth_path,
        # map_location=torch.device('cpu'))

    def classify(self, img):
        """
        定义识别函数
        :param img:传入img的numpy格式
        :return: 识别结果
        """
        transform_valid = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]
        )
        # img = img.convert("RGB")  # 如果传入的图像是RGB图像,则将这一句话注释;如果传入的是灰度图,则保留
        img_ = transform_valid(img).unsqueeze(0)  # 填充思维
        res = self.model(img_.cuda())
        _, indices = torch.max(res, 1)
        pred = indices.item()
        class_of_target = TYPE_LIST[pred]
        return class_of_target


# # 使用示例
# sar_classifier = SAR_classifier()
# img = Image.open('/Users/fox/Downloads/FUSAR_Ship_Processed/train/Cargo/Ship_C01S07N0808.tiff')
# class_of_target = sar_classifier.classify(img)
# print(class_of_target)
