import time
from stable_var import TimeCost, LogRecorder
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s -%(message)s')


# 用于计算模型计算时间
def time_cost(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        TimeCost.cost_time = time.perf_counter() - t
        print(f'func {func.__name__} cost time:{TimeCost.cost_time:.8f} s')
        return result

    return fun


def log_record(proc_result):
    LogRecorder.xywh = proc_result.xywh
    LogRecorder.xywhn = proc_result.xywhn
    LogRecorder.xyxy = proc_result.xyxy
    LogRecorder.xyxyn = proc_result.xyxyn
    LogRecorder.obj_num = proc_result.xyxy[0].size(0)
    LogRecorder.file_name = proc_result.files[0]
    LogRecorder.t = proc_result.t
    LogRecorder.shape = proc_result.s
    LogRecorder.GlobalLogger.debug("Raw image file name:%s" % LogRecorder.file_name)
    LogRecorder.GlobalLogger.debug(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {LogRecorder.shape}" % LogRecorder.t)


def get_logger(logger_name, log_dir):
    log_format = '[%(asctime)s] %(message)s'  # 定义日志输出格式
    logger = logging.getLogger(logger_name)  # 创建日志对象
    logger.setLevel(logging.DEBUG)  # 启动日志级别
    # 判断是否存在重复的logger对象，防止重复打印日志
    if not logger.handlers:
        # FileHandler 负责将日志写入文件
        file_handler = logging.FileHandler(log_dir + '/' + 'result_' + logger_name + '.txt', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        # logger绑定处理对象FileHandler
        # 将日志输出保存到文件中
        logger.addHandler(file_handler)
    return logger
