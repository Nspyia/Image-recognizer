import os
import tkinter
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk
from translate import Translator

import get_Inception_model
from tensorflow_predictor import TensorflowPredictor

root = tkinter.Tk()  # 生成root主窗口
root.title("图像分类")  # 设置窗体标题
root.geometry("820x820")  # 设置窗体大小

if not os.path.exists('./inception_model/classify_image_graph_def.pb'):  # 如果没下载model，则下载model
    get_Inception_model.download_inception_model()  # 下载model

translator = Translator(to_lang="chinese")  # 新建Translator对象


def translator_prediction_result(pre_res):  # 翻译模块
    res = pre_res.split("\n")[0] + '\n'
    for line in pre_res.split("\n")[1:-1]:
        s = translator.translate(line.split(',')[1])
        res += line + " (机翻: " + s + ")\n"
    return res  # 返回翻译结果


img_label = Label(root, width='800', height='533')  # 这是是显示预测图片的全局变量
res_label = Label(root)  # 这是是显示预测文字的全局变量
pdt = TensorflowPredictor()  # 新建预测类(自己写的)


def selector_image():  # 选择图片按钮点击发生的事件
    img_path = filedialog.askopenfilename(initialdir='./images')  # 弹窗选择图像文件返回图像地址
    pre_res = pdt.predict_image(image_path=img_path)  # 利用地址调用预测函数返回结果字符串
    pre_res = translator_prediction_result(pre_res)  # 机器翻译结果字符串
    photo = ImageTk.PhotoImage(file=img_path)
    img_label.config(imag=photo)  # 更新图片
    img_label.pack()
    res_label.config(text=pre_res, justify=LEFT, font=("微软雅黑", 13))  # 更新文字
    res_label.pack()
    root.mainloop()  # 进入消息循环
    return


btn_sel = tkinter.Button(root, text='选择图片', command=selector_image)  # 选择图片按钮
btn_sel.pack()

root.mainloop()  # 进入消息循环（必需组件）
