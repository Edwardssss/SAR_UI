import time
from stable_var import TimeCost, LogRecorder


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
