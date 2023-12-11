import cv2
import numpy as np
import time
import sys
import subprocess
import threading
import itertools
import matplotlib.pyplot as plt
from matplotlib import animation

pix = [] # 画素値平均の時間配列
pix_num = 25 # FPS

def red_binary_threshold(frame):
    # 赤色の抽出
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([80, 80, 255])
    # lower_red = np.array([0, 0, 100])
    # upper_red = np.array([110, 110, 255])
    red_mask = cv2.inRange(frame, lower_red, upper_red)
    
    # 二値化
    binary_frame = cv2.threshold(red_mask, 1, 255, cv2.THRESH_BINARY)[1]

    
    points = np.column_stack(np.where(binary_frame == 255))
    # points = cv2.findNonZero(binary_frame)
    
    return binary_frame, points

def dtft_plot():
    fig = plt.figure(figsize=(10, 6))
    params = {
        'fig': fig,
        'func': update,  # グラフを更新する関数
        'interval': 10,  # 更新間隔 (ミリ秒)
        'frames': itertools.count(0, 0.1),  # フレーム番号を生成するイテレータ
    }
    anime = animation.FuncAnimation(**params)
    plt.show()

def update(frame):
    x = np.arange(0, 1, 1/pix_num)
    print(pix)
    if(len(pix)==pix_num):
        plt.cla()
        # plt.ylim(0,100)
        plt.ylim(250, 400)
        # plt.ylim(250, 650)
        plt.grid()
        plt.plot(x, pix, color="red")

def main():
    global pix
    height = 480
    width = 640
    if(len(sys.argv) > 1): video_path = int(sys.argv[1])
    # else: video_path = '/home/sskr3/Videos/ウェブカム/2023-12-04-162420.webm'
    # else: video_path = '/home/sskr3/Videos/laser_binarization/use/45deg/20231129094447_output1.avi'
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture('/dev/video2')
    current_time = time.strftime("%Y%m%d%H%M%S")
    subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "exposure_auto=1"])
    subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_absolute=40"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 動画保存のための設定
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = {"MJPG":('MJPG', ".avi")}
    w_fourcc = cv2.VideoWriter_fourcc(*fourcc["MJPG"][0])
    # out = cv2.VideoWriter(f"test_vibration{fourcc['MJPG'][1]}", w_fourcc, 50.0, (640, 480), True)
    out1 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output1.avi', w_fourcc, 50.0, (640, 480), True)
    out2 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output2.avi', w_fourcc, 50.0, (640, 480), True)
    # out3 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output3.avi', w_fourcc, 50.0, (frame_width*2,frame_height), True)
    # out1 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output1.avi', w_fourcc, 50.0, (frame_width,frame_height), True)
    # out2 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output2.avi', w_fourcc, 50.0, (frame_width,frame_height), True)
    # out3 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output3.avi', w_fourcc, 50.0, (frame_width*2,frame_height), True)
    
    # plot
    thread1 = threading.Thread(target=dtft_plot)
    thread1.setDaemon(True)
    thread1.start()

    while True:
        time.sleep(0.05)
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        # 赤色の二値化
        b, points = red_binary_threshold(frame)
        binary_frame = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        # y=mx+c
        # 最小二乗法で直線の方程式を導出
        X = points[:, 1].reshape(-1, 1)
        y = points[:, 0]

        wide = 200
        point = int(binary_frame.shape[0]/2)
        point = np.arange(point-wide, point+wide)
        # x_num = np.column_stack(np.where((y > point-wide) & (y < point+wide))) # なにこれ
        x_num = np.column_stack(np.where(np.any(np.isin(y, point))))
    
        x_point = np.array([X[n] for n in x_num]) # pointで指定したheight値の点群だけ抜き出す
        if(len(np.ravel(x_point)) > 0):
            # print(x_point)
            x_mean = int(np.mean(np.ravel(x_point)))
            # print(np.mean(np.ravel(x_point)))
            pix.append(np.mean(np.ravel(x_point)))
            if(len(pix)>pix_num): pix.pop(0)
            # A = np.hstack([X, np.ones_like(X)])
            # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            for hei in range(binary_frame.shape[0]): binary_frame[hei, x_mean] = [0, 255, 0]  # 緑色の直線
            print("- * "*10)


        # A = np.hstack([X, np.ones_like(X)])
        # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # for x in range(binary_frame.shape[1]):
        #     y = int(m * x + c)
        #     if 0 <= y < binary_frame.shape[0]:
        #         binary_frame[y, x] = [0, 255, 0]  # 緑色の直線
        #         binary_frame[y, max(x-1, 0)] = [0, 255, 0]  # 緑色の直線
        #         binary_frame[y, min(x+1,width-1)] = [0, 255, 0]  # 緑色の直線
        
        # 元の画像と二値化画像を横に連結
        stacked_frame = np.hstack((frame, binary_frame))


        
        # 画面に表示
        cv2.imshow('Original vs Binary', stacked_frame)
        
        # 動画に書き込み
        # out1.write(frame)
        # out2.write(binary_frame)
        # out3.write(stacked_frame)
        
        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ラズパイレーザoff
    
    cap.release()
    out1.release()
    out2.release()
    # out3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
