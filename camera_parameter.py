import sys
import cv2

def main(n_camera):
    # 接続しているカメラのパラメータをOpenCVでチェックするプログラム
    # 第一引数には調べたいカメラの数字を入れる。内蔵はvideo0なので0、webはvideo2なので2
    cap = cv2.VideoCapture(n_camera)

    camera_parameter = [cv2.CAP_PROP_FRAME_WIDTH,
                        cv2.CAP_PROP_FRAME_HEIGHT,
                        cv2.CAP_PROP_FPS,
                        cv2.CAP_PROP_FOURCC,
                        cv2.CAP_PROP_FORMAT,
                        cv2.CAP_PROP_MODE,
                        cv2.CAP_PROP_CONVERT_RGB,
                        cv2.CAP_PROP_BRIGHTNESS,
                        cv2.CAP_PROP_CONTRAST,
                        cv2.CAP_PROP_SATURATION,
                        cv2.CAP_PROP_HUE,
                        cv2.CAP_PROP_GAIN,
                        cv2.CAP_PROP_EXPOSURE,
                        cv2.CAP_PROP_WB_TEMPERATURE,
                        cv2.CAP_PROP_GAMMA,
                        cv2.CAP_PROP_FOCUS,
                        cv2.CAP_PROP_PAN,
                        cv2.CAP_PROP_TILT,
                        cv2.CAP_PROP_ROLL]

    camera_parameter_str = ["cv2.CAP_PROP_FRAME_WIDTH",
                            "cv2.CAP_PROP_FRAME_HEIGHT",
                            "cv2.CAP_PROP_FPS",
                            "cv2.CAP_PROP_FOURCC",
                            "cv2.CAP_PROP_FORMAT",
                            "cv2.CAP_PROP_MODE",
                            "cv2.CAP_PROP_CONVERT_RGB",
                            "cv2.CAP_PROP_BRIGHTNESS",
                            "cv2.CAP_PROP_CONTRAST",
                            "cv2.CAP_PROP_SATURATION",
                            "cv2.CAP_PROP_HUE",
                            "cv2.CAP_PROP_GAIN",
                            "cv2.CAP_PROP_EXPOSURE",
                            "cv2.CAP_PROP_WB_TEMPERATURE",
                            "cv2.CAP_PROP_GAMMA",
                            "cv2.CAP_PROP_FOCUS",
                            "cv2.CAP_PROP_PAN",
                            "cv2.CAP_PROP_TILT",
                            "cv2.CAP_PROP_ROLL"]


    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    print("-  "*10)
    for x in range(len(camera_parameter)):
        print(f"{camera_parameter_str[x]} : {cap.get(camera_parameter[x])}")
    # for i in range(14):
    #     print(cap.get(i))
    print("-  "*10)

if __name__ == "__main__":
    main(int(sys.argv[1]))