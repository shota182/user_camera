import sys
import time
import datetime
import subprocess
import paramiko
import cv2
import numpy as np
import os
import glob
import tkinter.filedialog as fd
from natsort import natsorted
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
nl = np.linalg

def main():
    s = int(sys.argv[1])
    if(s == 0):
        # 適当
        print(10*np.tan(np.radians(65)))
    if(s == 1):
        # solvePnPテスト
        def draw(img, corners, imgpts):
            corner = tuple(corners[0].ravel())
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
            return img
        
        camera_matrix = np.array([[724.50357,   0.     , 313.78345],
                                  [0.     , 727.36014, 257.98553],
                                  [0.     ,   0.     ,   1.     ]])
        distortion_coeff = np.array([[0.001742, -0.104322, 0.003385, -0.006469, 0.000000]])
        chessboard_size = (8, 6)
        square_size = 24.0  # [mm]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0]*square_size:square_size,0:chessboard_size[1]*square_size:square_size].T.reshape(-1,2)

        axis = np.float32([[square_size,0,0], [0,square_size,0], [0,0,-square_size]]).reshape(-1,3)
        for fname in glob.glob('/home/sskr3/ros_ws/user_camera/airhug/data/*.png'):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # 回転行列と並進行列を計算する
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, distortion_coeff)
                R, Jacob = cv2.Rodrigues(np.array([rvecs.ravel()]))

                print("*- "*20)
                print(f"rvecs: {rvecs}")
                print(f"R : {R}")

                # チェッカーボードの3次元座標
                objp_w = (np.dot(R, objp.T)+np.tile(tvecs, len(objp))).T
                print(f"objp_w size : {objp_w.shape}")
                vector = np.array([(objp_w[1]-objp_w[0])*[10,1,1], (objp_w[8]-objp_w[0])*[1,10,1]]).T
                print(f"vector : {vector}")

                # ３次元の点を画像平面に射影
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, distortion_coeff)

                img = draw(img,corners2,imgpts)
                cv2.imshow('img',img)
                k = cv2.waitKey(0) & 0xff
                if k == 's':
                    cv2.imwrite(fname[:6]+'.png', img)
                
                # plot
                fig = plt.figure(figsize = (8, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title("checker_board", size = 20)
                ax.set_xlabel("x", size = 10)
                ax.set_ylabel("y", size = 10)
                ax.set_zlabel("z", size = 10)
                ax.set_xlim([-300, 300])
                ax.set_ylim([-500, 500] )
                ax.set_zlim([0,1000])
                plt.plot(objp_w.T[0], objp_w.T[1], objp_w.T[2], color="black")
                plt.quiver([0], [0], [0], [tvecs[0]], [tvecs[1]], [tvecs[2]], color="red")
                # plt.quiver([0, tvecs[0], tvecs[0]], [0, tvecs[1], tvecs[1]], [0, tvecs[2], tvecs[2]], [tvecs[0], square_size, 0], [tvecs[1], 0, square_size], [tvecs[2], 0, 0], color="red")
                plt.quiver([tvecs[0], tvecs[0]], [tvecs[1], tvecs[1]], [tvecs[2], tvecs[2]], [square_size*10, 0], [0, square_size*10], [0, 0], color="blue")
                plt.quiver([tvecs[0], tvecs[0]], [tvecs[1], tvecs[1]], [tvecs[2], tvecs[2]], [vector[0]], [vector[1]], [vector[2]], color="green")
                plt.show()

        cv2.destroyAllWindows()
    if(s==2):
        a = np.array([[0,1,2],[3,4,5],[6,7,8]])
        print(a)
        print(a.ravel().reshape((-1,3)))
    if(s==3):
        cap = cv2.VideoCapture(2) # webカメラ
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if(s==4):
        # exposure_time_absolute設定して映像表示 -> 保存エンコーダテスト
        ex = 100
        cap = cv2.VideoCapture(2)
        subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "auto_exposure=1"])
        subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_time_absolute={ex}"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fourcc = {"MP4Vmp4":('mp4v', ".mp4"),
                #   "MP4Vmov":('mp4v', ".mov"),
                #   "VP90":('VP90', ".webm"),
                #   "MP4Vavi":('mp4v', ".avi"),
                #   "DIVX":('DIVX', ".avi"),
                #   "XVID":('xvid', ".avi"),
                #   "WMV1":('WMV1', ".wmv"),
                #   "WMV2":('WMV2', ".wmv"),
                  "MJPG":('MJPG', ".avi"), # -------------正解-------------
                #   "H263":('H263', ".avi"),
                  "H264":('H264', ".mp4"),
                  "X264":('X264', ".mp4"),
                #   "YUY2":('YUY2', ".avi"),
                #   "I420":('I420', ".avi"),
                #   "PIM1":('PIM1', ".avi"),
                #   "NV12":('NV12', ".avi"),
                  "RGBA":('RGBA', ".avi")}
        for enc in fourcc:
            print(enc)
            w_fourcc = cv2.VideoWriter_fourcc(*fourcc[enc][0])
            out = cv2.VideoWriter(f'/home/sskr3/ros_ws/user_camera/test_encode/{enc}{fourcc[enc][1]}', w_fourcc, 20.0, (640, 480), True)
            start = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Camera Frame", frame)
                out.write(frame)
                if time.time()-start > 2:
                    break
            out.release()
            cv2.destroyAllWindows()
            time.sleep(1)
        cap.release()
    if(s==5):
        # *でフォルダ内のファイル全体を抜き出したときにどんな順番で得られるか知りたい
        # フォルダ内のファイルを取得してソートする
        path_folder = "/home/sskr3/ros_ws/user_camera/image_cap_2023-11-1-12-57"
        ex = "jpg"
        file = path_folder+"/*."+ex
        file_list = glob.glob(file)
        file_list_sorted = sorted(glob.glob(file)) # ファイル名昇順
        file_list_natsorted = natsorted(glob.glob(file)) # ファイル名の数字(str)昇順
        # ソートされたファイルリストの表示
        for file in file_list_natsorted:
            print(file)
        pass
    if(s==6):
        # exposure_time_absoluteを変更する
        cap = cv2.VideoCapture(2)
        subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "auto_exposure=1"])
        ex = [50,30]
        i = 0
        subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_time_absolute={ex[i]}"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera Frame", frame)
            i += 1
            subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_time_absolute={ex[int(i%2)]}"])
            # time.sleep(0.2)
            # 'q'キーを押してループを終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if(s==7):
        # ubuntuからラズパイにssh接続&コマンド指示
        # SSHクライアントを作成
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # ラズパイに接続
        client.connect('172.20.10.2', username='pi', password='raspberry')
        # コマンドをラズパイに送信
        stdin, stdout, stderr = client.exec_command("raspi-gpio set 2 op") # GPIO 21ピンを出力モードに設定
        stdin, stdout, stderr = client.exec_command("raspi-gpio set 2 dh") # High出力に設定
        # 出力を表示
        print("out1")
        print(stdout.read().decode('utf-8'))
        time.sleep(2)

        # コマンドをラズパイに送信
        stdin, stdout, stderr = client.exec_command("raspi-gpio set 2 dl") # Low出力に設定
        # 出力を表示
        print("out2")
        print(stdout.read().decode('utf-8'))

        # 接続を閉じる
        client.close()
    if(s==8):
        # 露光時間調整、無調整どっちが早いか調べる
        cap = cv2.VideoCapture(2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "auto_exposure=1"])
        jlist = []
        for j in [10,50]:
            time_list = []
            print(j)
            subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_time_absolute={j}"])
            # 無調整
            while True:
                time_0 = time.time()
                ret, frame = cap.read()
                time_1 = time.time()
                if not ret: break
                print(time_1-time_0)
                time_list.append(time_1-time_0)
                if(len(time_list)==10):break
            jlist.append(np.sum(time_list)/10)
            print(f"mean: {np.sum(time_list)/10}")
            print("*- "*20)
        print(jlist[1]-jlist[0])
        print((jlist[1]-jlist[0])/40)
            
            # cv2.imshow("Camera Frame", frame)
        # 調整
        time_adjust = time.time()
    if(s==9):
        # time.timeのテスト
        a = time.time()
        time.sleep(1)
        b = time.time()
        print(b-a)
    if(s==10):
        # while文テスト
        cap = cv2.VideoCapture(2)
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Camera Frame", frame)
            # 'q'キーを押してループを終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    if(s==11):
        #continueテスト
        i=0
        while True:
            if(i%2 == 0):
                print(0)
                i+=1
                continue
                print("0000")
            if(i%2 == 1):
                print(1)
                i+=1
                print("1111")
    if(s==12):
        # filedialogテスト
        # file
        test = fd.askopenfilename()
        print(test)
        # folder
        test = fd.askdirectory()
        print(test)
    if(s==13):
        # 動画中にsで保存、qで終了
        cap = cv2.VideoCapture(2)
        while True:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)

            # キーボード入力を取得
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # 's'キーを押すと画像を保存
                cv2.imwrite('folder_test/captured_frame.jpg', frame)
                print("フレームを保存しました")

            if key == ord('0'):  # '0'キーを押すとループを終了
                break
        # キャプチャを解放してウィンドウを閉じる
        cap.release()
        cv2.destroyAllWindows()
    if(s==14):
        # exposure_time_absolute設定して映像表示 -> 保存エンコーダテスト
        ex = 80
        cap = cv2.VideoCapture(2)
        subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", "auto_exposure=1"])
        subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "-c", f"exposure_time_absolute={ex}"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fourcc = {"MJPG":('MJPG', ".avi")}
        w_fourcc = cv2.VideoWriter_fourcc(*fourcc["MJPG"][0])
        out = cv2.VideoWriter(f"test_vibration{fourcc['MJPG'][1]}", w_fourcc, 50.0, (640, 480), True)
        key = cv2.waitKey(0) & 0xFF
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera Frame", frame)
            out.write(frame)
            if key == ord("q"):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()