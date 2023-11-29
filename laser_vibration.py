import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_plane(x, y, z, a, b, c, d):
    # Create a grid of points
    xp = np.linspace(-10, 10, 100)
    yp = np.linspace(-10, 10, 100)
    xp, yp = np.meshgrid(xp, yp)

    # Calculate the corresponding z values for each point on the grid
    zp = -(a * xp + b * yp + d) / c

    # Plot the 3D plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xp, yp, zp, alpha=0.5, color="white")
    ax.plot_surface(x, y, z, alpha=0.5, color="red")

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    

    plt.show()

def calc(roll=0, pitch=0, yaw=0, tx=0, ty=0, tz=0):
  # param
  [al, bl, cl, dl] = [4, 0, 1, -180] # レーザ平面の方程式(カメラ座標系)
#   [aw, bw, cw, dw] = [0, 0, -1, 200] # 壁面の方程式(ワールド座標系)
  [aw, bw, cw, dw] = [-2.145, 0, -1, 200] # 壁面の方程式(ワールド座標系)

  # initial
  K = np.array([[724, 0, 320],
                [0, 727, 240],
                [0, 0, 1]]) # カメラ内部行列

  [roll, pitch, yaw] = [np.radians(roll), np.radians(pitch), np.radians(yaw)]
  Rx = np.array([[1,0,0],
                [0,np.cos(yaw),-np.sin(yaw)],
                [0,np.sin(yaw),np.cos(yaw)]])
  Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                [0,1,0],
                [-np.sin(pitch),0,np.cos(pitch)]])
  Rz = np.array([[np.cos(roll),-np.sin(roll),0],
                [np.sin(roll),np.cos(roll),0],
                [0,0,1]])
  R = np.dot(np.dot(Rx, Ry), Rz)
#   R = np.dot(np.dot(Rz, Ry), Rx)
  print(f"R * {R}")
  # calc
  # 平面の方程式ワールド座標
  xs, ys, zs = R[0]@np.array([[cl],[0],[-al]]), R[1]@np.array([[cl],[0],[-al]]), R[2]@np.array([[cl],[0],[-al]])
  xt, yt, zt = R[0]@np.array([[al*bl],[-(al**2+cl**2)],[bl*cl]]), R[1]@np.array([[al*bl],[-(al**2+cl**2)],[bl*cl]]), R[2]@np.array([[al*bl],[-(al**2+cl**2)],[bl*cl]])
  
  s = np.linspace(-20, 20, 5)
  t = np.linspace(-10, 10, 5)
  s, t = np.meshgrid(s, t)
  x = tx + s*xs + t*xt
  y = ty + s*ys + t*yt
  z = tz-dl/cl + s*zs + t*zt
#   plot_plane(x, y, z, al, bl, cl, dl)
  plot_plane(x, y, z, aw, bw, cw, dw)
  
  
  # 直線の方程式のパラメータ
  k = - (aw*xt+bw*yt+cw*zt) / (aw*xs+bw*ys+cw*zs)
  l = - (aw*tx+bw*ty+cw*(tz-dl/cl)+dw) / (aw*xs+bw*ys+cw*zs)
  [u], [v], [w] = k*xs+xt, k*ys+yt, k*zs+zt
  [p], [q], [r] = tx+l*xs, ty+l*ys, tz-dl/cl+l*zs
  t = np.arange(100)
  cross = np.array([t * u + p, t * v + q, t * w + r])  # 平面の交点
  # print(f"cross: {cross}")
  # image
  g = np.dot(K, np.array([[u],[v],[w]]))
  h = np.dot(K, np.array([[p], [q], [r]]))
  print(f"g: {g.T}")
  print(f"h: {h.T}")

  i = np.array([])
  j = np.array([])
  for t in np.arange(-10, 10):
    ij = t*g + h
    i = np.append(i, ij[0]/ij[2])
    j = np.append(j, ij[1]/ij[2])
  return i, j

[roll, pitch, yaw] = [0, -10, 0] # カメラ座標系の姿勢[deg]
# i, j = calc(roll, pitch, yaw)
# i_10, j_10 = calc(0, -10, 0)
i_1, j_1 = calc(tx=-1)
i0, j0 = calc()
i1, j1 = calc(tx=1)
i2, j2 = calc(tx=2)
# i5, j5 = calc(0, 5, 0)
# i10, j10 = calc(0, 10, 0)
print(f"i : {i0[0]}, {i1[0]}, {i_1[0]}, {i2[0]}")
plt.figure()
plt.plot(i0, j0, color="red")
plt.plot(i1, j1, color="blue")
plt.plot(i_1, j_1, color="green")
plt.plot(i2, j2, color="black")

# plt.xlim(0,640)
plt.xlim(280,320)
plt.ylim(0,480)
plt.show()
