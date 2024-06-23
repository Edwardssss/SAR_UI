class WidthHeight:
    windowWidth = 350  # 获得当前窗口宽
    windowHeight = 200  # 获得当前窗口高

    blankWidth = 960
    blankHeight = 50

    # Frame参数
    imgFrameWidth = 900
    imgFrameHeight = 600

    btnFrameWidth = 900
    btnFrameHeight = 150

    TextWidth = 50
    TextHeight = 10

    def __init__(self):
        pass


class FilePath:
    RawImgPath = None
    LogExportPath = ".\\Log"
    ResultSavePath = ".\\result"

    def __init__(self):
        pass


class ProcVar:
    proc_flag = False
    FileType = [("图片", ".jpg"), ('All Files', ' *')]

    def __init__(self):
        pass


class YOLOModel:

    def __init__(self):
        pass


class TimeCost:
    cost_time = 0

    def __init__(self):
        pass


class LogRecorder:
    xyxy = 0
    xyxyn = 0
    xywh = 0
    xywhn = 0
    obj_num = 0
    file_name = ''
    t = None
    shape = None
    GlobalLogger = None

    def __init__(self):
        pass
