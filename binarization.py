import sys
import tkinter.filedialog as fd
import glob
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

# 単色にする(他の色を0にする)
def only_color(image, color="blue"):
    image_copy = copy.copy(image)
    colorlist = {"blue":[1,2], "green":[0,2], "red":[0,1]}
    for i in range(len(image_copy)):
        for j in range(len(image_copy[0])):
            for k in range(2):
                image_copy[i][j][colorlist[color][k]] = 0
    return image_copy

# 閾値から特定の色の範囲に二値化
def pick_color(image, color="blue", original=0):
    image_copy = copy.copy(image)
    if(color=="blue"):
        bgr = [210,150,40]
        thresh = 40
    elif(color=="red"):
        if(original==0): # original
            bgr = [-30,-30,150]
            thresh = 40
        elif(original==1): # ピンクも抽出
            bgr = [180, 180, 235]
            thresh = 40
        elif(original==2): # 狭い範囲の赤の抽出
            bgr = [180, 180, 235]
            thresh = 20
        elif(original==3):
            bgr = [200, 180, 275]
            thresh = 40
        elif(original==4):
            bgr = [190, 170, 260]
            thresh = 40
        elif(original==5):
            bgr = [190, 170, 270]
            thresh = 50

    #色の閾値
    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    #画像の2値化
    maskBGR = cv2.inRange(image_copy,minBGR,maxBGR)
    #画像のマスク（合成）
    resultBGR = cv2.bitwise_and(image_copy, image_copy, mask = maskBGR)
    return resultBGR

# hsvに変換して赤色を抽出
def hsv_color(image, color="red"):
    image_copy = copy.copy(image)
    #赤を抽出
    low_hsv_min = np.array([0, 200,50])
    low_hsv_max = np.array([1, 255, 255])

    #画像の2値化（Hueが0近辺）
    maskHSV_low = cv2.inRange(image_copy,low_hsv_min,low_hsv_max)

    high_hsv_min = np.array([178, 200,0])
    high_hsv_max = np.array([179, 255, 255])

    #画像の2値化（Hueが179近辺）
    maskHSV_high = cv2.inRange(image_copy,high_hsv_min,high_hsv_max)

    #２つの領域を統合
    hsv_mask = maskHSV_low | maskHSV_high

    #画像のマスク（合成）
    return cv2.bitwise_and(image_copy, image_copy, mask = hsv_mask)

# 特定の色だけ抽出、表示(使用していない)
def specific_color(image, mode="pick", color="red"):
    if(mode=="only"): image_r = only_color(image, color)
    elif(mode=="pick"): image_r = pick_color(image, color)
    print(f"image:{image_r}")
    print(f"row:{len(image_r)}, column:{len(image_r[0])}")
    cv2.imshow('img',image_r)
    cv2.imshow('img',cv2.resize(image_r,None,fx=0.5,fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 背景差分
def bgsubtract(image_back=np.array([]), image_front=np.array([])):
    # 画像の読み込み
    if(len(image_back)==0): image_back = cv2.imread(fd.askopenfilename())
    else: image_back = copy.copy(image_back)
    if(len(image_front)==0): image_front = cv2.imread(fd.askopenfilename())
    else: image_front = copy.copy(image_front)

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    fgmask = fgbg.apply(image_back)
    fgmask = fgbg.apply(image_front)

    # 表示
    # cv2.imshow('frame',fgmask)
    # cv2.imshow('frame',cv2.resize(fgmask,None,fx=0.5,fy=0.5))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return fgmask

# 表示
def image_visual(image=np.array([]), save=0, check=0):
    # check
    if(check):
        print(f"image_num:{len(image)}")
        for i in range(len(image)):
            print(f"image {i} is :{len(image[i])}x{len(image[i][0])}")
    # merging
    image_num = len(image)
    if(image_num==1): pass
    else:
        image_visual = np.hstack((image[0], image[1]))
        if(image_num==3): np.vstack((image_visual, image[2]))
        elif(image_num==4):
            image_4 = np.hstack((image[2], image[3]))
            image_visual = np.vstack((image_visual, image_4))
    #resize
    image_visual = cv2.resize(image_visual,None,fx=0.8,fy=0.8)
    # visualizaiton
    cv2.imshow('img',image_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if(save):
        bg_diff_path  = "/home/sskr3/ros_ws/user_camera/image_result/"+env+"/airhug_"+str(length)+"mm.png"
        cv2.imwrite(bg_diff_path,image_visual)

def main():
    for fname in glob.glob(path_dark):
        print(f"filepath:{fname}")
        # 画像を読み込む
        # file = fd.askopenfilename()
        image = cv2.imread(fname)

        # グレースケール
        # image_gray_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image_gray_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image_gray_brg = cv2.cvtColor(image[:, :, [0, 2, 1]], cv2.COLOR_BGR2GRAY)
        # image_gray_grb = cv2.cvtColor(image[:, :, [1, 2, 0]], cv2.COLOR_BGR2GRAY)

        # imreadで返された配列を出力 
        # print(f"image:{image}")
        # print(f"row:{len(image)}, column:{len(image[0])}")

        # 読み込んだ画像を表示する
        # cv2.imshow('img',image)
        # cv2.imshow('img',cv2.resize(image,None,fx=0.5,fy=0.5))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Rのみ
        image_only = only_color(image, "red")
        # image_hsv = hsv_color(image, "red")
        image_pick = pick_color(image, "red", 1)
        image_pick_2 = pick_color(image, "red", 2)
        image_pick_3 = pick_color(image, "red", 3)
        image_pick_4 = pick_color(image, "red", 5)

        # 二値化
        if(1):
            path_back = "/home/sskr3/ros_ws/user_camera/image_cap_2023-11-5-14-47/3_back.jpg"
            path_front = "/home/sskr3/ros_ws/user_camera/image_cap_2023-11-5-14-47/3_front.jpg"
        else:
            path_back = fd.askopenfilename()
            path_front = fd.askopenfilename()
        image_back = cv2.imread(path_back)
        image_front = cv2.imread(path_front)
        image_bg = bgsubtract(image_back, image_front)

        # 結合
        # image_visual(np.array([image_gray_bgr, image_gray_rgb, image_gray_brg, image_gray_grb]))
        # image_visual(np.array([image, image_pick, image_pick_3, image_pick_4]))
        diff_simple = cv2.absdiff(image_back, image_front)
        image_visual(np.array([image_back, image_front, only_color(diff_simple, "red"), cv2.cvtColor(image_bg, cv2.COLOR_GRAY2RGB)]), 1)

if __name__ == "__main__":
    # img_fs=cv2.resize(img,None,fx=0.5,fy=0.5)
    path_bright = "/home/sskr3/ros_ws/user_camera/image_bright/*.png"
    path_dark = "/home/sskr3/ros_ws/user_camera/image_dark/*.png"
    path_LED = "/home/sskr3/ros_ws/user_camera/image_LED/*.png"
    # env, length = sys.argv[1], sys.argv[2]
    main()