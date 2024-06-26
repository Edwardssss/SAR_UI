import os.path

import torch
from stable_var import FilePath, LogRecorder
from log_gen import time_cost
from PIL import Image


@time_cost
def sar_ship_detect(model="standard"):
    raw_img = FilePath.RawImgPath
    if model == "standard":
        s_model = torch.hub.load('.\\SAR_YOLO', 'custom', path=r'model/YOLO v5-s.pt', source='local',
                                 force_reload=True)
        results = s_model(raw_img, augment=False)
    elif model == "lite":
        lite_model = torch.hub.load('.\\SAR_YOLO-Lite', 'custom', path_or_model=r'model/YOLO v5-lite.pt',
                                    source='local',
                                    force_reload=True)
        results = lite_model(raw_img, augment=False)
    else:
        # 模型错误
        return None
    return results


def image_cut():
    # 裁剪图片
    raw_image = Image.open(FilePath.RawImgPath)
    cut_image_list = []
    for i in range(0, LogRecorder.obj_num):
        indices = torch.tensor([i])
        cut_tensor = torch.index_select(LogRecorder.xyxy[0], 0, indices.cuda())
        cut_image_list.append(raw_image.crop(tuple(cut_tensor.cpu().numpy().tolist()[0][0:4])))
        obj_image_path = os.path.join(FilePath.CutSavePath, f"{LogRecorder.file_name[0:-4]}_{i}_obj.jpg")
        cut_image_list[i].save(obj_image_path)
        LogRecorder.GlobalLogger.debug(f"{i + 1}th object recognized,related image has been saved in {obj_image_path}")
    LogRecorder.GlobalLogger.debug(f"All objects are processed successfully. Total number:{LogRecorder.obj_num}")
    return cut_image_list
