from tkinter import *
from stable_var import WidthHeight,LogRecorder,FilePath
import BtnCallBack
from HyperLink.tkinter.tkHyperlinkManager import HyperlinkManager
import webbrowser
from functools import partial
from log_gen import get_logger

if __name__ == "__main__":
    main_window = Tk()  # 创建窗口对象的背景色

    # 创建Logger对象
    LogRecorder.GlobalLogger = get_logger("SAR_UI_log",FilePath.LogExportPath)
    LogRecorder.GlobalLogger.debug("script launch successfully")

    main_window.title("SAR图像目标识别演示程序")
    main_window.iconbitmap(".\\res\\icon\\radar.ico")
    main_window.resizable(False, True)  # 固定窗口大小

    screenWidth, screenHeight = main_window.maxsize()  # 获得屏幕宽和高

    geometryParam = '%dx%d+%d+%d' % (
        WidthHeight.windowWidth, WidthHeight.windowHeight, (screenWidth - WidthHeight.windowWidth) / 2,
        (screenHeight - WidthHeight.windowHeight) / 2)
    main_window.geometry(geometryParam)  # 设置窗口大小及偏移坐标

    # 文本控件
    Text_box = Text(main_window, width=WidthHeight.TextWidth, height=WidthHeight.TextHeight)
    Text_box.grid(column=1, row=1, columnspan=4, rowspan=1)

    # 按钮控件
    select_img_btn = Button(main_window, command=lambda: BtnCallBack.ImgPathSelectCallBack(Text_box))
    select_img_btn["text"] = "选取图片路径"
    select_img_btn.grid(column=1, row=2)

    begin_proc_btn = Button(main_window, command=lambda: BtnCallBack.ProcCallBack(Text_box, select_var))
    begin_proc_btn["text"] = "开始检测"
    begin_proc_btn.grid(column=2, row=2)

    # export_log_btn = Button(main_window, command=lambda: BtnCallBack.LogExportCallBack(Text_box))
    # export_log_btn["text"] = "导出log数据"
    # export_log_btn.grid(column=3, row=2)

    select_log_btn = Button(main_window, command=lambda: BtnCallBack.LogPathSelectCallBack(Text_box))
    select_log_btn["text"] = "选取log路径"
    select_log_btn.grid(column=4, row=2)

    save_result_btn = Button(main_window)
    save_result_btn["text"] = "导出结果路径"
    save_result_btn.grid(column=3, row=2)

    about_btn = Button(main_window, command=lambda: BtnCallBack.AboutCodeCallBack(Text_box))
    about_btn["text"] = "关于代码"
    about_btn.grid(column=4, row=3)

    # 标签控件
    model_label = Label(main_window, text="模型选择：")
    model_label.grid(column=1, row=3)

    # 单选框控件
    select_var = StringVar()
    select_var.set("standard")
    model_select_s_btn = Radiobutton(main_window, text='YOLO v5-s', value="standard", variable=select_var)
    model_select_lite_btn = Radiobutton(main_window, text='YOLO v5-lite', value="lite", variable=select_var)
    model_select_s_btn.grid(column=2, row=3)
    model_select_lite_btn.grid(column=3, row=3)

    # 创建超链接
    hyperlink = HyperlinkManager(Text_box)
    # 欢迎语
    Text_box.insert(INSERT, "初次使用前请阅读使用手册，手册在根目录中\n")
    Text_box.insert(INSERT, "训练代码",
                    hyperlink.add(partial(webbrowser.open, "https://github.com/oy159/sar_detect.git")))
    Text_box.insert(INSERT, "\n")

    LogRecorder.GlobalLogger.debug("OK let's begin detect objects in SAR images!")

    # main_window.wm_attributes('-topmost', 1)  # 窗口置顶

    main_window.mainloop()  # 进入消息循环
