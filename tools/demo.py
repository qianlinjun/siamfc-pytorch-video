# -*-coding: utf-8 -*-

from __future__ import absolute_import

import os,sys
import glob
import numpy as np

sys.path.append(r"C:\qianlinjun\公司\视频项目\code\siamfc-pytorch-master\siamfc")
from siamfc import TrackerSiamFC
import ops

import cv2
import numpy as np
global frame
point1 = None
point2 = None
global g_rect

getBB = False



def on_mouse(event, x, y, flags, param):
    global frame, point1, point2,g_rect, getBB
    img2 = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        print("1-EVENT_LBUTTONDOWN")
        point1 = (x, y)
        # cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        # cv2.imshow('image', img2)
 
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        print("2-EVENT_FLAG_LBUTTON")
        # cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        # cv2.imshow('image', img2)
        point2 = (x, y)
 
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        print("3-EVENT_LBUTTONUP")
        point2 = (x, y)
        # cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        # # cv2.imshow('image', img2)
        if point1!=point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            g_rect=[min_x,min_y,width,height]
            # cut_img = img[min_y:min_y + height, min_x:min_x + width]
            # cv2.imshow('ROI', cut_img)
            getBB = True
 
def get_image_roi(rgb_image):
    '''
    获得用户ROI区域的rect=[x,y,w,h]
    :param rgb_image:
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    global img
    img=bgr_image
    cv2.namedWindow('image')
    while True:
        cv2.setMouseCallback('image', on_mouse)
        # cv2.startWindowThread()  # 加在这个位置
        cv2.imshow('image', img)
        key=cv2.waitKey(0)
        if key==13 or key==32:#按空格和回车键退出
            break
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return g_rect
 
# def select_user_roi(image_path):
#     '''
#     由于原图的分辨率较大，这里缩小后获取ROI，返回时需要重新scale对应原图
#     :param image_path:
#     :return:
#     '''
#     orig_image = image_processing.read_image(image_path)
#     orig_shape = np.shape(orig_image)
#     resize_image = image_processing.resize_image(orig_image, resize_height=800,resize_width=None)
#     re_shape = np.shape(resize_image)
#     g_rect=get_image_roi(resize_image)
#     orgi_rect = image_processing.scale_rect(g_rect, re_shape,orig_shape)
#     roi_image=image_processing.get_rect_image(orig_image,orgi_rect)
#     image_processing.cv_show_image("RECT",roi_image)
#     image_processing.show_image_rect("image",orig_image,orgi_rect)
#     return orgi_rect
 
 
# if __name__ == '__main__':
#     # image_path="../dataset/images/IMG_0007.JPG"
#     image_path="../dataset/test_images/lena.jpg"
 
#     # rect=get_image_roi(image)
#     rect=select_user_roi(image_path)
#     print(rect)

import tkinter
from tkinter import filedialog
def main():
    global frame,point1, point2
    net_path = r'C:\qianlinjun\公司\视频项目\code\siamfc-pytorch-master\pretrained\siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    trackf = 0

    # TODO:单纯打开监控 或 本地摄像头 或 视频
    # camera_path = "rtsp:..."  # 你的监控ip 
    # camera_path = r"C:\Users\qianlinjun\Videos\wjj\1.mp4"     # 本地摄像
    # camera_path = 视频地址  如video.avi    video.mp4
    root = tkinter.Tk() # 创建一个Tkinter.Tk()实例 
    root.withdraw() # 将Tkinter.Tk()实例隐藏
    default_dir = r"C:\Users\qianlinjun\Videos\wjj"
    file_path = filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser(default_dir)))


    
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if getBB is False:
                if point1 is not None and point2 is not None:
                    cv2.rectangle(frame, point1, point2, (0, 0, 255), thickness=2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(30) == ord('q'):
                    break
            else:
                if trackf == 0:
                    tracker.init(frame, g_rect)
                    tracker.init(frame, g_rect, latestFrame=True)
                else:
                    if trackf % 20 == 0 :
                        # 假如中间有误检,那么会导致检测失败
                        # 70%相信第一帧,30相信最新的一帧
                        tracker.init(frame, resBox, latestFrame=True)
                        print("init tracker with frame:{}".format(trackf))
                    resBox = tracker.update(frame)
                    ops.show_image(frame, resBox, colors=[0,255,0], cvt_code=None)
                trackf += 1


if __name__ == '__main__':
    # seq_dir = os.path.expanduser('~/data/OTB/Crossing/')
    # img_files = sorted(glob.glob(r"C:\qianlinjun\workSetData\siamfc_seq_data\*.jpg"))
    # print(img_files)
    # anno = np.array([1280*0.432031,720*0.4625,1280*0.245312,720*0.311111]) #np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    main()
    
    