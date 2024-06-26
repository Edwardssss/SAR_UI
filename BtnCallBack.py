import tkinter
import tkinter.messagebox
from stable_var import FilePath, ProcVar, TimeCost
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import filedialog
import detect
import os
import webbrowser
from log_gen import log_record
from stable_var import LogRecorder
from Classifier_part import SAR_classifier
import torch


def ProcCallBack(textbox, model_var):
    if FilePath.RawImgPath is None:
        tkinter.messagebox.showinfo(title="警告", message="未选择图片文件，请重新选择\n")
        textbox.insert(tkinter.INSERT, "未选择图片文件，请重新选择\n")
        LogRecorder.GlobalLogger.debug("Raw image file error,please confirm the existence of the image")
    else:
        ImgDisplayWindow = tkinter.Toplevel()
        ImgDisplayWindow.resizable(False, False)

        raw_img_raw = Image.open(FilePath.RawImgPath)
        raw_img_photo = ImageTk.PhotoImage(raw_img_raw)

        raw_img_img = tkinter.Label(ImgDisplayWindow, image=raw_img_photo)
        raw_img_img.grid(column=1, row=1, columnspan=1, rowspan=1)

        textbox.insert(tkinter.INSERT, "图片处理中\n")
        LogRecorder.GlobalLogger.debug("Processing,please wait in patience")
        # 此处将图片导入网络中
        proc_result = detect.sar_ship_detect(model=model_var.get())
        if proc_result is None:
            textbox.insert(tkinter.INSERT, "目标检测失败，可能是因为模型或图片文件不正确\n")
            LogRecorder.GlobalLogger.debug("Exception during processing,please check model and image file")
            return
        # 保存结果至result文件夹
        if model_var.get() == "standard":
            proc_result.save(save_dir=FilePath.ResultSavePath, exist_ok=True)
            LogRecorder.GlobalLogger.debug("Model:YOLO v5-s")
        elif model_var.get() == "lite":
            proc_result.save(save_dir=FilePath.ResultSavePath)
            LogRecorder.GlobalLogger.debug("Model:YOLO v5-lite")
        # 记录Log
        log_record(proc_result=proc_result)
        # 裁剪区域
        obj_image_list = detect.image_cut()
        textbox.insert(tkinter.INSERT, "所有目标检测完成，总计%d个目标\n" % LogRecorder.obj_num)
        # 进行分类
        classifier = SAR_classifier()
        class_result_list = []
        for i in range(0, LogRecorder.obj_num):
            class_result_list.append(classifier.classify(obj_image_list[i]))
        # 对结果进行标注
        unlabel_img = Image.open(os.path.join(FilePath.ResultSavePath, LogRecorder.file_name))
        draw_img = ImageDraw.Draw(unlabel_img)
        text_font = ImageFont.truetype(".\\res\\font\\msyh.ttf", 20)

        for j in range(0, LogRecorder.obj_num):
            indices = torch.tensor([j])
            draw_img.text(
                tuple(torch.index_select(LogRecorder.xywh[0], 0, indices.cuda()).cpu().numpy().tolist()[0][0:2]),
                f'{class_result_list[j]}', (255, 0, 0), font=text_font)

        unlabel_img.save(os.path.join(FilePath.CutSavePath, f"{LogRecorder.file_name[0:-4]}_labeled.png"))
        textbox.insert(tkinter.INSERT, "分类器检测完成，结果保存于%s\n" % os.path.join(FilePath.CutSavePath, f"{LogRecorder.file_name[0:-4]}_labeled.png"))

        textbox.insert(tkinter.INSERT, "图片已保存至%s\n" % FilePath.ResultSavePath)
        LogRecorder.GlobalLogger.debug("Result is saved in %s" % FilePath.ResultSavePath)

        (_, result_name) = os.path.split(FilePath.RawImgPath)
        result_path = os.path.join(FilePath.ResultSavePath, result_name)
        proc_img_raw = Image.open(result_path)
        proc_img_photo = ImageTk.PhotoImage(proc_img_raw)
        proc_img_img = tkinter.Label(ImgDisplayWindow, image=proc_img_photo)
        proc_img_img.grid(column=2, row=1, columnspan=1, rowspan=1)

        label_img_raw = Image.open(os.path.join(FilePath.CutSavePath, f"{LogRecorder.file_name[0:-4]}_labeled.png"))
        label_img_photo = ImageTk.PhotoImage(label_img_raw)
        label_img_img = tkinter.Label(ImgDisplayWindow, image=label_img_photo)
        label_img_img.grid(column=3, row=1, columnspan=1, rowspan=1)

        # label_img = tkinter.Label(ImgDisplayWindow, image=ImageTk.PhotoImage(
        #     Image.open(os.path.join(FilePath.CutSavePath, f"{LogRecorder.file_name[0:-4]}_labeled.png"))))
        # label_img.grid(column=3, row=1, columnspan=1, rowspan=1)

        textbox.insert(tkinter.INSERT, "图片显示完成\n")
        LogRecorder.GlobalLogger.debug("Illustration done")
        textbox.insert(tkinter.INSERT, f"此次模型处理用时{TimeCost.cost_time:.8f}s")
        LogRecorder.GlobalLogger.debug(f"Time cost:{TimeCost.cost_time:.8f}s")
        ProcVar.proc_flag = True
        LogRecorder.GlobalLogger.debug("Process done")
        ImgDisplayWindow.mainloop()


# def LogExportCallBack(textbox):
#     if ProcVar.proc_flag is False:
#         tkinter.messagebox.showinfo(title="警告", message="未进行处理，请处理后再导出结果\n")
#         textbox.insert(tkinter.INSERT, "未进行处理，请处理后再导出结果\n")
#     else:
#         # 导出Log文件
#
#         tkinter.messagebox.showinfo(title="提示", message="已导出为log文件\n")
#         textbox.insert(tkinter.INSERT, "已导出为log文件\n")


def ImgPathSelectCallBack(textbox):
    FilePath.RawImgPath = filedialog.askopenfilename(title="请选择图片文件", filetypes=ProcVar.FileType)
    textbox.insert(tkinter.INSERT, "已选择图片文件路径 %s\n" % FilePath.RawImgPath)


def LogPathSelectCallBack(textbox):
    FilePath.LogExportPath = filedialog.askdirectory(title="请选择log文件保存路径")
    textbox.insert(tkinter.INSERT, "已选择log文件导出路径 %s\n" % FilePath.LogExportPath)


def SaveResultCallBack(textbox):
    FilePath.ResultSavePath = filedialog.askdirectory(title="请选择检测结果保存路径")
    textbox.insert(tkinter.INSERT, "已选择检测结果导出路径 %s\n" % FilePath.ResultSavePath)


def AboutCodeCallBack(textbox):
    textbox.insert(tkinter.INSERT, "已打开项目训练代码仓库 https://github.com/oy159/sar_detect.git\n")
    webbrowser.open("https://github.com/oy159/sar_detect.git")
