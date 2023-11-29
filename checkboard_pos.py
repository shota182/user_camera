import numpy as np
import cv2
import glob
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# なんか違う

# カメラキャリブレーションは済
# チェッカーボードのコーナー数、物理的間隔から変換行列を作成する
# cornersが画素値
    # おそらくcornersはチェッカーボードの物理的間隔をもとにして3次元座標を導出している
# 最小二乗法で平面の方程式を算出する

def fit_plane(point_cloud):
    """
    入力
        point_cloud : xyzのリスト　numpy.array型
    出力
        plane_v : 法線ベクトルの向き(単位ベクトル)
        com : 重心　近似平面が通る点
    """

    com = np.sum(point_cloud, axis=0) / len(point_cloud)
    # 重心を計算
    q = point_cloud - com
    # 重心を原点に移動し、同様に全点群を平行移動する  pythonのブロードキャスト機能使用
    Q = np.dot(q.T, q)
    # 3x3行列を計算する 行列計算で和の形になるため総和になる
    la, vectors = np.linalg.eig(Q)
    # 固有値、固有ベクトルを計算　固有ベクトルは縦のベクトルが横に並ぶ形式
    plane_v = vectors.T[np.argmin(la)]
    # 固有値が最小となるベクトルの成分を抽出

    return plane_v, com

def calc_plane_ori(points):
    # sumの計算
    [x, y, z] = [np.sum(points[0]), np.sum(points[1]), np.sum(points[2])]
    [x2, y2] = [np.sum([i**2 for i in points[0]]), np.sum([i**2 for i in points[1]])]
    [xy, xz, yz] = [np.sum([points[0][i]*points[1][i] for i in range(len(points[0]))]), np.sum([points[0][i]*points[2][i] for i in range(len(points[0]))]), np.sum([points[1][i]*points[2][i] for i in range(len(points[0]))])]
    matA = [[len(points[0]), x, y],
            [x, x2, xy],
            [y, xy, y2]]
    b = [z, xz, yz]
    return LU(matA, b)

def calc_plane(points, bool_print=0):
    if(bool_print): print(f"points:{points}")

    # データの中心を原点に移動
    centered_points = points - np.mean(points, axis=0)

    # 最小二乗法を使用して平面の法線ベクトルを計算
    A = centered_points
    _, _, V = np.linalg.svd(A)
    normal_vector = V[-1]

    # 平面の法線ベクトルを正規化
    normal_vector /= np.linalg.norm(normal_vector)

    # 平面の方程式の係数を抽出
    a, b, c = normal_vector

    # 平面の方程式: ax + by + cz + d = 0 のdを計算
    d = -np.dot(normal_vector, np.mean(points, axis=0))
    return a,b,c,d

# チェスボードのサイズ（内部コーナーの数）
chessboard_size = (7, 10)

# チェスボードのコーナーの物理的な間隔（例: 24 mm）
square_size = 24.0  # ミリメートル


# 物理的なコーナーの座標（X、Y、Z）
object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

camera_matrix = np.array([[724.50357,   0.     , 313.78345],
                          [0.     , 727.36014, 257.98553],
                          [0.     ,   0.     ,   1.     ]])

distortion_coeff = np.array([[0.001742, -0.104322, 0.003385, -0.006469, 0.000000]])

# チェスボードのコーナーを検出するための画像を読み込む
for fname in glob.glob('/home/sskr3/Pictures/*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # image = cv2.imread('sample_image.jpg')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # チェスボードのコーナーを検出
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    # print("+ - "*20)
    # print(corners)
    # print("- - "*10)
    # print(corners[0])
    # print(corners[0][0])

    if(1):
        cv2.imshow("original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        corner = np.array([corners[i][0] for i in range(len(corners))]).T
        plt.figure()
        plt.plot(corner[0], corner[1])
        plt.scatter(corner[0], corner[1])
        plt.show()

    if ret:
        # コーナーの3D座標を計算
        retval, rvecs, tvecs = cv2.solvePnP(object_points, corners, camera_matrix, distortion_coeff)

        # 回転ベクトルから回転行列への変換
        rotation_matrix, _ = cv2.Rodrigues(rvecs)

        # 3D座標を計算
        translation_matrix = np.array(tvecs)
        transformation_matrix = np.column_stack((rotation_matrix, translation_matrix))
        points = np.array([])
        point = np.array([0,0,0])
        for i in np.array([[0,0,0,1],[0,0,1,1],[0,0,2,1],[0,0,3,1],[0,0,4,1],
                           [0,1,0,1],[0,1,1,1],[0,1,2,1],[0,1,3,1],[0,1,4,1],
                           [0,2,0,1],[0,2,1,1],[0,2,2,1],[0,2,3,1],[0,2,4,1],
                           [0,3,0,1],[0,3,1,1],[0,3,2,1],[0,3,3,1],[0,3,4,1],
                           [0,4,0,1],[0,4,1,1],[0,4,2,1],[0,4,3,1],[0,4,4,1]]):
            transformation_matrix = np.vstack((transformation_matrix, i))

            # ワールド座標系での3D座標
            world_coordinates = np.dot(transformation_matrix, i)

            # 3D座標データ（X、Y、Z）
            points = np.append(points, world_coordinates[:3])
            # 3D座標を表示
            print(f"3D Coordinates ({i[:3]}):{world_coordinates[:3]}")
            print(f"distance: {np.sum(np.array([(world_coordinates[i]-point[i])**2 for i in range(3)]))**(1/2)}")
        
        points = points.reshape((-1, 3))
        # 平面の方程式を導出
        a,b,c,d = calc_plane(points)
        # v, com = fit_plane(points)
        # print(f"v:{v}")
        # print(f"com:{com}")

        # 平面の方程式を表示
        print(f"平面の方程式: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        
        # 画像の画素値と平面の方程式から画素値に対する3次元座標の取得
        # カメラ画像の画素座標
        [image_x, image_y] = corners[3][0]

        # 平面の方程式パラメータ
        [A, B, C, D] = [a, b, c, d]

        # 画素座標からノーマライズデバイス座標への変換
        normalized_coordinates = np.dot(np.linalg.inv(camera_matrix), np.array([image_x, image_y, 1]))
        print(f"normalize:{normalized_coordinates}")

        # 3D交点を計算
        t = -(D + np.dot(np.array([A, B, C]), np.array([normalized_coordinates[0], normalized_coordinates[1], 1]))) / (A**2 + B**2 + C**2)
        intersection_point = np.array([A, B, C]) * t

        # 交点を表示
        print(f"画素値{image_x,image_y}の3D座標 (X, Y, Z):{intersection_point}")
        print("+ - "*20)

        if(0):

            # 方向ベクトル
            direction_vector = normalized_coordinates  # 例として適切な値を設定

            # 平面の法線ベクトル
            normal_vector = np.array([A, B, C])

            # 直線の始点座標
            line_origin = np.array([0,0,0])

            # 平面と直線の交点を求める
            t = -(np.dot(normal_vector, line_origin) + D) / np.dot(normal_vector, direction_vector)
            intersection_point = line_origin + t * direction_vector

            # 交点を表示
            print("交点の3D座標 (X, Y, Z):", intersection_point)



        plot = 1
        if(plot):
            # Figureを追加
            fig = plt.figure(figsize = (8, 8))

            # 3DAxesを追加
            ax = fig.add_subplot(111, projection='3d')

            # Axesのタイトルを設定
            ax.set_title("checker_board", size = 20)

            # 軸ラベルを設定
            ax.set_xlabel("x", size = 10)
            ax.set_ylabel("y", size = 10)
            ax.set_zlabel("z", size = 10)

            # 軸目盛を設定
            # ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            # ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])


            ax.plot(points.T[0], points.T[1], points.T[2], color = "red") # 平面のコーナー
            ax.plot([points[0][0], points[0][0]+A], [points[0][1], points[0][1]+B], [points[0][2], points[0][2]+C]) # 平面の法線
            # ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color = "green")
            plt.show()
