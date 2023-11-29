import sys
import tkinter.filedialog as fd
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# 二値化画像から画素値を取得するプログラム
def get_pixel(image_bg, mode=1):
    [height, width] = [len(image_bg), len(image_bg[0])]
    pixel = [[], []]
    for i in range(height):
        if(mode):
            mean = np.sum([image_bg[i][j]/255*j for j in range(width)])/int(np.count_nonzero(image_bg[i]!=0))
            if(int(np.count_nonzero(image_bg[i]==255))!=0):
                print(i, mean)
                pixel[0].append(i)
                pixel[1].append(mean)
        else:
            for j in range(width):
                if(image_bg[i][j] == 255):
                    pixel[0].append(i)
                    pixel[1].append(j)
    return np.array(pixel)

# 画素値を入力すると3次元カメラ座標を出力
def get_pos_vector(pixel, camera_matrix, dist_coeffs):
    # pixelは[[list_y], [list_x]]とする
    # 画素値をカメラ座標に変換
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    pos = [[], []]
    for i in range(len(pixel[0])):
        pixel_coords = np.array([[pixel[1][i]], [pixel[0][i]], [1]]) # (u,v,1)
        camera_coords = np.dot(inv_camera_matrix, pixel_coords)
        for j in range(2):
            pos[j].append(camera_coords[j][0])
        # print(camera_coords[2][0])
    pos.append([1]*len(pos[0]))
    return np.array(pos)

# ラインレーザとの交点からスケール不定性を削除し、絶対座標を取得する
def get_pos(pos_vector, laser_pos, laser_angle):
    # get_pos_vectorだとスケール不定性があるので、レーザ平面との交点を導出する
    if(laser_angle<90): point = np.array([[laser_pos, 0, 0], [0, 10, laser_pos*np.tan(np.radians(laser_angle))], [0, -10, laser_pos*np.tan(np.radians(laser_angle))]])
    elif(laser_angle==90): point = np.array([[laser_pos, 0, 0], [laser_pos, 10, 10], [laser_pos, -10, 10]])
    vector = np.array([point[1]-point[0], point[2]-point[0]])
    vector_normal = np.cross(vector[0], vector[1])
    vector_normal = np.array([vector_normal[i]/np.linalg.norm(vector_normal, ord=2) for i in range(3)])
    print(vector_normal)
    [a,b,c] = vector_normal
    d = -np.dot(vector_normal, point[0]) # ax+by+cz+d=0 のパラメータ
    print(f"param:{a,b,c,d}")
    if(c!=0): point_any = np.array([1,0,(-d-a*1)/c]) # ラインレーザはyが任意。x=1を代入して導出
    else: point_any = np.array([-d/a,0,0])
    print(f"point_any:{point_any}")
    position = []
    # 直線の方向ベクトル：pos_vector.T[i]
    # 平面の法線ベクトル：vector_normal
    for i in range(len(pos_vector[0])):
        # t0 = (-d - np.dot(vector_normal, pos_vector.T[i])) / np.dot(vector_normal, pos_vector.T[i])
        t = (-d) / np.dot(vector_normal, pos_vector.T[i])
        position.append(pos_vector.T[i] * t)
        # print("*- "*20)
        # print(np.dot(vector_normal, pos_vector.T[i]))
        # print(-d / np.dot(vector_normal, pos_vector.T[i]))
        # print(t0)
        # print(t)
        # print("*- "*10)
    return np.array(position).T

def main():
    # 二値化
    path_back = "/home/sskr3/ros_ws/user_camera/image_"+str(env)+"/airhug_"+str(length)+"mm_0v.png"
    path_front = "/home/sskr3/ros_ws/user_camera/image_"+str(env)+"/airhug_"+str(length)+"mm_5.2v.png"
    image_back = cv2.imread(path_back)
    image_front = cv2.imread(path_front)
    image_bg = bgsubtract(image_back, image_front)

    # ごま塩ノイズ除去
    image_filter = cv2.medianBlur(image_bg,3)

    # resize
    image_resize = cv2.resize(image_filter, (640, 480))
    # raw
    cv2.imshow(f'raw data {sys.argv[1]}-{sys.argv[2]}',cv2.resize(image_front, (640, 480)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # bg
    cv2.imshow(f'{sys.argv[1]}-{sys.argv[2]}',image_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    K = np.array([[724.50357,   0.     , 313.78345],
                    [0.     , 727.36014, 257.98553],
                    [0.     ,   0.     ,   1.     ]]) # カメラの内部パラメータ[pixel]
    dist = np.array([[0.001742, -0.104322, 0.003385, -0.006469, 0.000000]])

    # 画素値取得
    pixel = get_pixel(image_resize, 0) # np.array([[縦の画素値], [横の画素値]]) mode=1は行で1出力に平均化している
    position_vector = get_pos_vector(pixel, K, dist) # np.array([[x], [y], [z]])：カメラ座標系におけるベクトル
    position = get_pos(position_vector, 44, 90)

    # debug plot(画像座標系において、黒い点が検出したレーザ点群)
    plt.figure()
    plt.title("laser point in image",size = 30)
    plt.xlabel("u",size = 30)
    plt.ylabel("v",size = 30)
    plt.plot([0,640,640,0,0],[0,0,-480,-480,0])
    plt.scatter(pixel[1], -pixel[0], color="black")
    plt.show()

    # debug plot(画素の縦に対する距離の値)
    plt.figure()
    plt.title("v[pixel] , distance[mm]", size = 30)
    plt.grid()
    plt.xlabel("distance[mm]", size = 30)
    plt.ylabel("v[pixel]", size = 30)
    # plt.plot([0,640,640,0,0],[0,0,480,480,0])
    plt.scatter(position[2], -pixel[0], color="black")
    plt.xlim([0,int(int(sys.argv[2])*1.5)])
    plt.show()

    # plot
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("position", size = 20)
    ax.set_xlabel("x[mm]", size = 20)
    ax.set_ylabel("y[mm]", size = 20)
    ax.set_zlabel("z[mm]", size = 20)
    ax.set_xlim([-100, 100])
    # ax.set_ylim([-500, 500])
    # ax.set_zlim([0,int(int(sys.argv[2])*1.5)])
    ax.scatter([0], [0], [0], color="red")
    ax.scatter(position[0], position[1], position[2], color="black")
    plt.show()


    # 結合
    # image_visual(np.array([image_gray_bgr, image_gray_rgb, image_gray_brg, image_gray_grb]))
    # image_visual(np.array([image, image_pick, image_pick_3, image_pick_4]))
    # image_visual(np.array([image_back, image_front, image_pick_3, cv2.cvtColor(image_bg, cv2.COLOR_GRAY2RGB)]), 1)

if __name__ == "__main__":
    # img_fs=cv2.resize(img,None,fx=0.5,fy=0.5)
    path_bright = "/home/sskr3/ros_ws/user_camera/image_bright/*.png"
    path_dark = "/home/sskr3/ros_ws/user_camera/image_dark/*.png"
    path_LED = "/home/sskr3/ros_ws/user_camera/image_LED/*.png"
    env, length = sys.argv[1], sys.argv[2]
    main()