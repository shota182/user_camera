import cv2
import numpy as np
import time
import sys

def red_binary_threshold(frame):
    # 赤色の抽出
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([80, 80, 255])
    red_mask = cv2.inRange(frame, lower_red, upper_red)
    
    # 二値化
    binary_frame = cv2.threshold(red_mask, 1, 255, cv2.THRESH_BINARY)[1]

    
    points = np.column_stack(np.where(binary_frame == 255))
    # points = cv2.findNonZero(binary_frame)
    
    return binary_frame, points

def main():
    if(len(sys.argv) > 1): video_path = sys.argv[1]
    else: video_path = '/home/sskr3/Videos/laser_binarization/use/45deg/20231129094447_output1.avi'
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture('/dev/video2')
    current_time = time.strftime("%Y%m%d%H%M%S")
    
    # 動画保存のための設定
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out1 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    out2 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    out3 = cv2.VideoWriter(f'/home/sskr3/Videos/laser_binarization/{current_time}_output3.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width*2,frame_height))
    
    # ラズパイレーザon
    
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

        wide = 20
        point = int(binary_frame.shape[0]/2)
        # point = np.arange(point-wide, point+wide)
        x_num = np.column_stack(np.where((y > point-wide) & (y < point+wide)))
        # x_num = np.column_stack(np.where(np.any(np.isin(y, point))))
        x_point = np.array([X[n] for n in x_num]) # pointで指定したheight値の点群だけ抜き出す
        if(np.ravel(x_point) != np.array([])):
            # print(x_point)
            x_mean = int(np.mean(np.ravel(x_point)))
            print(np.mean(np.ravel(x_point)))
            # A = np.hstack([X, np.ones_like(X)])
            # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            for hei in range(binary_frame.shape[0]):
                binary_frame[hei, x_mean] = [0, 255, 0]  # 緑色の直線
            print("- * "*10)

        # A = np.hstack([X, np.ones_like(X)])
        # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # for x in range(binary_frame.shape[1]):
        #     y = int(m * x + c)
        #     if 0 <= y < binary_frame.shape[0]:
        #         binary_frame[y, x] = [0, 255, 0]  # 緑色の直線
        
        # 元の画像と二値化画像を横に連結
        stacked_frame = np.hstack((frame, binary_frame))


        
        # 画面に表示
        cv2.imshow('Original vs Binary', stacked_frame)
        
        # 動画に書き込み
        out1.write(frame)
        out2.write(binary_frame)
        out3.write(stacked_frame)
        
        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ラズパイレーザoff
    
    cap.release()
    out1.release()
    out2.release()
    out3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
