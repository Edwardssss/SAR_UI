import torch
from stable_var import FilePath
from log_gen import time_cost


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
