import datetime
import subprocess
import cv2
# import numpy as np
import os
# import glob
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# nl = np.linalg

def main():
    # videocapture -> 画像を保存(qで停止)
    # カメラからのキャプチャを開始
    cap = cv2.VideoCapture(2) # webカメラ
    # 露光時間調整
    subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "auto_exposure=1"])
    subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "exposure_time_absolute=50"])

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    image_counter = 0

    # フォルダ作成関係
    now = datetime.datetime.now()
    folder = f"image_cap_{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
    os.mkdir(folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera', frame)
        # チェッカーボード画像保存
        ret, corners = cv2.findChessboardCorners(frame, (6,8), None)
        if ret:
            image_name = f"{folder}/captured_image_{image_counter}.jpg"
            cv2.imwrite(image_name, frame)
            print(f"Captured {image_name}")
            image_counter += 1
        # 'q'キーを押してループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()