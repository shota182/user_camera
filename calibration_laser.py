import sys
import os
import copy
import time
import datetime
import glob
import subprocess
import paramiko
import tkinter.filedialog as fd
import cv2
import copy
import numpy as np
from scipy import optimize
from natsort import natsorted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    mode = sys.argv[1]
    if(mode=="capture"): capture()
    elif(mode=="process"): process()
    elif(mode=="test"): test()
    # レーザのキャリブレーションをするプログラム
    # レーザ照射あり、なしの画像を認識し、導出した直線の方程式からレーザ平面の方程式を導出するプログラム

    # フォルダ読み込み、ソート

    # 全画像-1でfor文
    # コーナの画像座標導出

def capture():
    # 変数
    raspi = True
    exposure = 50 # exposure
    gpio = 2 # laser cuicuit plus pin : gpio pin in raspberrypi
    checkerboard = (5,7)
    # 初期設定
    cap = cv2.VideoCapture(2) # web
    subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "exposure_auto=1"])
    subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_absolute={exposure}"])
    # subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "auto_exposure=1"])
    # subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_time_absolute={exposure}"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    image_counter = 0
    now = datetime.datetime.now()
    folder = f"image_cap_{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
    # ssh関係
    if(raspi):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('172.20.10.2', username='pi', password='raspberry')
        stdin, stdout, stderr = client.exec_command(f"raspi-gpio set {gpio} op") # GPIOピンを出力モードに設定
        
    pri = True
    while True:
        if(pri):
            print("next: 's',  quit: 'q'")
            pri=False
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if(key == ord('s')):
            # 背景撮影
            image_back = copy.copy(frame)
            ret, corners = cv2.findChessboardCorners(image_back, checkerboard, None)
            if(ret):
                print("checkerboard is detected in image_back")
            else:
                print("checkerboard is not detected in image_back")
                continue

            # gpio High
            if(raspi):
                stdin, stdout, stderr = client.exec_command(f"raspi-gpio set {gpio} dh") # High出力に設定
                print(f"gpio {gpio} High")
                time.sleep(0.5)

            # 前景撮影
            ret, frame = cap.read()
            image_front = copy.copy(frame)

            # gpio Low
            if(raspi):
                stdin, stdout, stderr = client.exec_command(f"raspi-gpio set {gpio} dl") # Low出力に設定
                print(f"gpio {gpio} Low")


            # コーナー検出
            if(not ret): continue
            ret, corners = cv2.findChessboardCorners(image_front, checkerboard, None)
            if(ret):
                print("checkerboard is detected in image_front")
            else:
                print("checkerboard is not detected in image_front")
                continue

            # 背景差分
            print("next: 's',  delete: 'd',  quit: 'q'")
            diff = cv2.bgsegm.createBackgroundSubtractorMOG()
            fgmask = diff.apply(image_back)
            fgmask = diff.apply(image_front)
            diff = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)

            # 最小二乗法で直線の方程式を導出
            points = np.argwhere(diff == 255)  # 白色（値が255）の点群の座標を取得
            X = points[:, 1].reshape(-1, 1)
            y = points[:, 0]
            A = np.hstack([X, np.ones_like(X)])
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            for x in range(diff.shape[1]):
                y = int(m * x + c)
                if 0 <= y < diff.shape[0]:
                    diff[y, x] = [0, 255, 0]  # 緑色の直線

            # image = np.hstack((np.hstack((image_back, image_front)), diff))
            diff_simple = cv2.absdiff(image_back, image_front)
            fgbg = cv2.createBackgroundSubtractorMOG2()
            fgbg_g = fgbg.apply(image_back)
            fgbg_g = fgbg.apply(image_front)
            diff_gaussian = cv2.cvtColor(fgbg_g, cv2.COLOR_GRAY2RGB)
            image = np.hstack((np.hstack((diff_simple, diff_gaussian)), diff))
            print("delete: 'd',  save: 's'")
            cv2.imshow('img',image)
            # sで保存、dで削除
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                try: os.mkdir(folder)
                except: pass
                cv2.imwrite(f"{folder}/{image_counter}_back.jpg", image_back)
                cv2.imwrite(f"{folder}/{image_counter}_front.jpg", image_front)
                # cv2.imwrite(f"{folder}/{image_counter}_diff0.jpg", diff_simple)
                # cv2.imwrite(f"{folder}/{image_counter}_diff1.jpg", diff_gaussian)
                # cv2.imwrite(f"{folder}/{image_counter}_diff2.jpg", diff)
                print(f"saved image No.{image_counter}")
                image_counter += 1
                cv2.destroyWindow("img")
            elif key == ord('d'):
                print("skipped image")
                cv2.destroyWindow("img")
                continue
        elif key == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()

def process():
    pass

def test():
    # デバッグ、プログラム作成時に使うテスト用関数

    # 背景差分、二値化した画像に最小二乗法で直線を当てはめて表示
    if(0):
        # 変数
        file_list = "/home/sskr3/ros_ws/user_camera/image_cap_2023-12-11-15-9/*.jpg"
        chessboard_size = (5, 7)
        square_size = 15.0 # [mm]
        camera_matrix = np.array([[724.50357,   0.     , 313.78345],
                                [0.     , 727.36014, 257.98553],
                                [0.     ,   0.     ,   1.     ]])
        distortion_coeff = np.array([[0.001742, -0.104322, 0.003385, -0.006469, 0.000000]])
        point_all = [] # レーザ全点群
        # 初期設定
        if(file_list == 0): file_list = fd.askdirectory()+"/*.jpg"
        file_list_natsorted = natsorted(glob.glob(file_list)) # ファイル名の数字(str)昇順
        # winsize = (chessboard_size[0]*2+1, chessboard_size[1]*2+1)
        winsize = (5,5)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0]*square_size:square_size,0:chessboard_size[1]*square_size:square_size].T.reshape(-1,2)
        # print(file_list_natsorted)

        # 各画像の処理
        for i in range(0, len(file_list_natsorted), 2):
            image_back = cv2.imread(file_list_natsorted[i])
            image_front = cv2.imread(file_list_natsorted[i+1])
            gray = cv2.cvtColor(image_front,cv2.COLOR_BGR2GRAY)

            # 背景差分
            # ナチュラル差分
            # diff_simple = cv2.absdiff(image_back, image_front)
            # diff_simple_right = copy.copy(diff_simple)
            # diff_simple_right[:, :640 // 2, :] = 0
            # MOG
            image_back_red = image_back[:, :, 2]
            image_front_red = image_front[:, :, 2]
            diff = cv2.bgsegm.createBackgroundSubtractorMOG()
            fgmask = diff.apply(image_back_red)
            fgmask = diff.apply(image_front_red)
            diff_mog = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            diff_mog_right = copy.copy(fgmask)
            diff_mog_right[:, :640 // 2] = 0
            # 点群の閾値
            pix = get_pixel(diff_mog_right, 0) # [[x], [y]]
            if(len(pix[0]) < 30): continue
            # 背景差分表示
            points = np.argwhere(diff_mog_right == 255)  # 白色（値が255）の点群の座標を取得

            # 最小二乗法で直線の方程式を導出
            X = points[:, 1].reshape(-1, 1)
            y = points[:, 0]
            A = np.hstack([X, np.ones_like(X)])
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            # 直線を描画して表示
            ret, corners = cv2.findChessboardCorners(image_front, chessboard_size, None)
            # コーナーを描画して表示
            img = cv2.drawChessboardCorners(image_front, chessboard_size, corners, 1)

            # point1 = (0, int(c))
            # point2 = (diff_mog_right.shape[1], int(m * diff_mog_right.shape[1] + c))
            # cv2.line(img, point1, point2, (0, 255, 0), 0.5)  # 緑色の実線、太さ2

            for x in range(diff_mog_right.shape[1]):
                y = int(m * x + c)
                if 0 <= y < diff_mog_right.shape[0]:
                    img[y, x] = [0, 255, 0]  # 緑色の直線
            cv2.imshow('Checkerboard Corners', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    # レーザキャリブレーションのテスト
    if(1):
        # 変数
        file_list = "/home/sskr3/ros_ws/user_camera/image_cap_2023-12-11-15-38/*.jpg"
        chessboard_size = (5, 7) # チェッカーボードのコーナーの数
        square_size = 15.0 # 1辺[mm]
        camera_matrix = np.array([[724.50357,   0.     , 313.78345],
                                [0.     , 727.36014, 257.98553],
                                [0.     ,   0.     ,   1.     ]]) # カメラ内部行列
        distortion_coeff = np.array([[0.001742, -0.104322, 0.003385, -0.006469, 0.000000]]) # レンズ歪みパラメータ
        point_all = [] # レーザ全3次元点群を格納[[x,y,z],...]
        # 初期設定
        if(file_list == 0): file_list = fd.askdirectory()+"/*.jpg"
        file_list_natsorted = natsorted(glob.glob(file_list)) # ファイル名の数字(str)昇順
        # winsize = (chessboard_size[0]*2+1, chessboard_size[1]*2+1) # cornersの探索範囲
        winsize = (5,5)
        # winsize = (11,11)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # corners収束判定基準
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0]*square_size:square_size,0:chessboard_size[1]*square_size:square_size].T.reshape(-1,2)
        # print(file_list_natsorted)

        # 各画像の処理
        for i in range(0, len(file_list_natsorted), 2):
            image_back = cv2.imread(file_list_natsorted[i])
            image_front = cv2.imread(file_list_natsorted[i+1])
            gray = cv2.cvtColor(image_front,cv2.COLOR_BGR2GRAY)

            # 背景差分
            # ナチュラル差分
            # diff_simple = cv2.absdiff(image_back, image_front)
            # diff_simple_right = copy.copy(diff_simple)
            # diff_simple_right[:, :640 // 2, :] = 0
            image_back_red = image_back[:, :, 2]
            image_front_red = image_front[:, :, 2]
            # MOG
            diff = cv2.bgsegm.createBackgroundSubtractorMOG() # Rのみ
            fgmask = diff.apply(image_back_red)
            fgmask = diff.apply(image_front_red)
            diff_mog_right = copy.copy(fgmask)
            diff_mog_right[:, :640 // 2] = 0
            # 二値化テスト
            # diff2 = cv2.bgsegm.createBackgroundSubtractorMOG() # BGR全部抜き出し
            # fgmask2 = diff2.apply(image_back)
            # fgmask2 = diff2.apply(image_front)
            diff2 = cv2.createBackgroundSubtractorMOG2() # MOG2
            fgmask2 = diff2.apply(image_back_red)
            fgmask2 = diff2.apply(image_front_red)
            diff_mog_right2 = copy.copy(fgmask2)
            diff_mog_right2[:, :640 // 2] = 0

            if(0): # 二値化のチェック
                image = np.hstack((diff_mog_right, diff_mog_right2))
                cv2.imshow('Camera', image)
                cv2.waitKey(0)

            # 背景差分から255のピクセルを抜き出し
            pix = get_pixel(diff_mog_right, 0) # [[x], [y]]
            if(len(pix[0]) < 30): continue
            normalized_coordinates = np.dot(np.linalg.inv(camera_matrix), np.array([pix[0], pix[1], [1]*len(pix[0])])) # 点群の方向ベクトル

            if(0): # normalized_coordinatesのデバッグ
                print(normalized_coordinates)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(normalized_coordinates[0], normalized_coordinates[1], normalized_coordinates[2])
                ax.scatter([0], [0], [0], color="red")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()

            print("* - "*20)
            print(file_list_natsorted[i])

            # チェッカーボード3d座標
            ret, corners = cv2.findChessboardCorners(image_front, chessboard_size, None)
            print(ret)

            corners2 = cv2.cornerSubPix(gray,corners,winsize,(-1,-1),criteria) # cornersだと粗いからサブピクセル精度で出す
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, distortion_coeff) # PnP問題
            R, Jacob = cv2.Rodrigues(np.array([rvecs.ravel()])) # チェッカーボード座標系の回転行列の計算
            objp_w = (np.dot(R, objp.T)+np.tile(tvecs, len(objp))).T # cornersがチェッカーボード座標系->カメラ座標系
            a,b,c,d = calc_plane(objp_w) # チェッカーボードの平面の方程式パラメータ
            # print(f"corners2: {corners2}")
            # print(f"rvecs: {rvecs}")
            # print(f"tvecs: {tvecs}")
            # print(f"R    : {R}")
            # print(f"objp_w : {objp_w}")
            print(f"チェッカーボード平面の方程式: \n{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            [line_a, line_b] = fit_line_2d(pix)
            print(f"画像座標系 レーザ直線の方程式: \ny = {line_a:.2f}x + {line_b:.2f}")

            # レーザ座標方向ベクトルとチェッカーボード3次元座標の交点からレーザの3次元座標点群を取得
            intersection_points = []
            for norm in normalized_coordinates.T:
                t = -d / np.dot(np.array([a, b, c]), norm)
                intersection_points.append(norm * t) # チェッカーボード平面上の点の位置ベクトル
                point_all.append(norm*t)
            intersection_points = np.array(intersection_points) # レーザの3次元座標点群

            # print(intersection_points.shape)
            # print(f"intersection, x: {intersection_points.T[0]}")
            # print(f"intersection, y: {intersection_points.T[1]}")
            # print(f"intersection, z: {intersection_points.T[2]}")

            if(0): # チェッカーボードとレーザ直線のプロット
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(objp_w.T[0], objp_w.T[1], objp_w.T[2], color="gray") # チェッカーボード平面
                ax.scatter(intersection_points.T[0], intersection_points.T[1], intersection_points.T[2], color="green") # レーザ点群
                ax.scatter([0], [0], [0], color="red")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()

        # レーザの全点群から平面の方程式を導出
        a_pl, b_pl, c_pl, d_pl = calc_plane(np.array(point_all))

        # レーザのパラメータ
        print(f"レーザ平面の方程式 \n {a_pl}x + {b_pl}y + {c_pl}z + {d_pl} = 0")
        print(f"カメラとレーザの相対距離(右が正): {-d_pl/a_pl}")
        # print(f"レーザのpitch角: {np.degrees(np.arctan2(-(a_pl)/c_pl, 1))}")

        # レーザ直線プロット
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 平面
        xx, yy = np.meshgrid(range(30, 150, 10), range(-210, 210, 20))
        zz = (-d_pl - a_pl * xx - b_pl * yy) / c_pl
        # ---------------------------------------------------
        # カラーマップに基づき、色を生成する。
        # Z < 0 のパッチはアルファチャンネルを0として色を透明にする。
        # norm = plt.Normalize(vmin=zz.min(), vmax=zz.max())
        # colors = plt.cm.jet(norm(zz))
        # colors[zz < 0] = (0, 0, 0, 0)
        # ax.plot_surface(xx, yy, zz, alpha=0.5, color="pink", facecolors=colors, rstride=1, cstride=1) # レーザ平面
        # ---------------------------------------------------
        ax.plot_surface(xx, yy, zz, alpha=0.5, color="pink") # レーザ平面
        ax.scatter(np.array(point_all).T[0], np.array(point_all).T[1], np.array(point_all).T[2], color="red", lw=0.4) # レーザ全点群
        # ax.plot([-50,-50,50,50,-50], [-50,50,50,-50,-50], [0,0,0,0,0], color="red")
        ax.plot([0, 70,70,-70,-70,70,0,-70,-70,0,70], [0,-30,30,30,-30,-30,0,-30,30,0,30], [0,30,30,30,30,30,0,30,30,0,30], color="black", lw=2) # カメラ座標系原点
        ax.scatter([0], [0], [0], color="black", lw=3)
        # ax.plot([-60, 150], [0,0], [-(a_pl*(-60)+d_pl)/c_pl, -(a_pl*(150)+d_pl)/c_pl], color="cyan")
        ax.set_xlabel('X[mm]', fontsize=15)
        ax.set_ylabel('Y[mm]', fontsize=15)
        ax.set_zlabel('Z[mm]', fontsize=15)
        ax.set_xlim([-300, 300])
        ax.set_ylim([-200, 200])
        ax.set_zlim([0, 400])
        plt.show()

        if(0): # レーザ角度確認
            plt.figure()
            plt.plot([-60, 150], [-(a_pl*(-60)+d_pl)/c_pl, -(a_pl*(150)+d_pl)/c_pl], color="cyan")
            plt.axis('equal')
            plt.show()

# 3次元座標から平面の方程式を導出
def calc_plane(points, bool_print=0): # points:[[x,y,z],...]
    if(bool_print): print(f"points:{points}")

    # データの中心を原点に移動
    centered_points = points - np.mean(points, axis=0)

    # 最小二乗法を使用して平面の法線ベクトルを計算
    A = centered_points
    _, _, V = np.linalg.svd(A)
    normal_vector = V[-1] # 法線ベクトル

    # 平面の法線ベクトルを正規化
    normal_vector /= np.linalg.norm(normal_vector)

    # 平面の方程式の係数を抽出
    a, b, c = normal_vector

    # 平面の方程式: ax + by + cz + d = 0 のdを計算
    d = -np.dot(normal_vector, np.mean(points, axis=0))
    return a,b,c,d

# 二値化画像から255の値を持つ座標を抜き出す
def get_pixel(image_bg, mode=1): # image_bg:GRAY
    [height, width] = [len(image_bg), len(image_bg[0])]
    # print(height, width)
    pixel = [[], []]
    for i in range(height):
        if(mode): # 1行ごとに平均をとっている
            mean = np.sum([image_bg[i][j]/255*j for j in range(width)])/int(np.count_nonzero(image_bg[i]!=0))
            if(int(np.count_nonzero(image_bg[i]==255))!=0):
                print(i, mean)
                pixel[0].append(i)
                pixel[1].append(mean)
        else: # 全点群
            for j in range(width):
                if(image_bg[i][j] == 255):
                    pixel[0].append(j)
                    pixel[1].append(i)
    return np.array(pixel)

# 最小二乗法で直線の方程式を求める
def fit_line_2d(points): # [[x], [y]]
    # データを行列に変換
    X = points[0]
    y = points[1]

    # データ行列にバイアス列を追加
    X = np.c_[X, np.ones(X.shape[0])]

    # 最小二乗法による回帰係数の計算
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return beta

# 3次元の直線の方程式を導出(使ってない)
def fit_line_3d(normal_vector, point):
    # 平面上の点からの垂直な直線の方程式を求める
    d = -np.dot(normal_vector, point)
    return np.append(normal_vector, d)

if __name__ == "__main__":
    main()