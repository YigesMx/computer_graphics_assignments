import numpy as np

def sphere(n_lat: int, n_lon: int, r: float = 1): # 球体对象生成器
    n_lat += 1 # n_lat: 纬度切分数（即有多少横条带）
    n_lon += 1 # n_lon: 经度切分数（即有多少竖条带）
    r = 1 # r: 球体半径

    vert_list = [] # 存放顶点三维坐标的列表
    vert_texcoord_list = [] # 存放每个顶点在纹理平面中的坐标的列表
    vert_norm_list = [] # 存放每个顶点的法向量的列表，对于球，每个顶点的法向量是该顶点坐标的单位向量
    vert_tangent_list = [] # 存放每个顶点的切向量的列表，切向量是指与该顶点的法向量垂直，沿经线向下的向量
    for i in range(n_lat): # 从上到下
        for j in range(n_lon): # 从左到右
            # 右手系
            theta = np.pi * i / (n_lat - 1)
            phi = 2 * np.pi * j / (n_lon - 1)
            x = np.sin(theta) * np.cos(phi)
            y = np.cos(theta)
            z = np.sin(theta) * np.sin(phi)
            vert_list.append([x*r, y*r, z*r, 1.0])
            vert_norm_list.append([x, y, z])
            vert_texcoord_list.append([1 - j / (n_lon - 1), i / (n_lat - 1)])

            theta_tangent = theta + np.pi / 2
            tx = np.sin(theta_tangent) * np.cos(phi)
            ty = np.cos(theta_tangent)
            tz = np.sin(theta_tangent) * np.sin(phi)
            vert_tangent_list.append([-tx, -ty, -tz])

    
    # 将每个经纬线围成的矩形拆分成两个三角形，并存入索引数组
    surf_list = [] # 存放每个三角形的顶点索引的列表
    for i in range(n_lat - 1):
        surf_list.append([])
        for j in range(n_lon - 1):
            surf_list[i].append([i * n_lon + j, i * n_lon + j + 1, (i + 1) * n_lon + j])
            surf_list[i].append([i * n_lon + j + 1, (i + 1) * n_lon + j + 1, (i + 1) * n_lon + j])
    
    # 计算每个面的法向量
    surf_norm_list = [] # 存放每个三角形的法向量的列表
    for i in range(n_lat - 1):
        surf_norm_list.append([])
        for j in range(2*(n_lon - 1)):
            BC = np.array(vert_list[surf_list[i][j][2]][:3]) - np.array(vert_list[surf_list[i][j][1]][:3])
            BA = np.array(vert_list[surf_list[i][j][0]][:3]) - np.array(vert_list[surf_list[i][j][1]][:3])
            norm = np.cross(BC, BA)
            surf_norm_list[i].append(norm)
    
    # 将每个三角形的顶点坐标、纹理坐标、法向量存入列表
    points, texcoords, surf_norms, vert_norms, vert_tangents= [], [], [], [], []
    for i in range(n_lat - 1):
        for j in range(2*(n_lon - 1)):
            for k in range(3):
                points.append(vert_list[surf_list[i][j][k]])
                texcoords.append(vert_texcoord_list[surf_list[i][j][k]])
                surf_norms.append(surf_norm_list[i][j])
                vert_norms.append(vert_norm_list[surf_list[i][j][k]])
                vert_tangents.append(vert_tangent_list[surf_list[i][j][k]])
    
    return np.array(points, np.float32), np.array(texcoords, np.float32), np.array(surf_norms, np.float32), np.array(vert_norms, np.float32), np.array(vert_tangents, np.float32)

def plane(h: float = 2, w: float = 2): # 平面对象生成器
    # center_x, center_y: 平面中心点坐标
    # normal: 平面法向量 y
    # tangent: 平面切向量 z
    # h, w: 平面的长和宽
    center = np.array([0,0,0], np.float32)
    normal = np.array([0,1,0], np.float32)
    tangent= np.array([0,0,1], np.float32)
    binormal = np.cross(normal, tangent) # 平面副法向量 x

    vert_list = [] # 存放顶点三维坐标的列表

    vert_list.append(np.array([*(center + h/2*binormal + w/2*tangent), 1], np.float32) )
    vert_list.append(np.array([*(center + h/2*binormal - w/2*tangent), 1], np.float32) )
    vert_list.append(np.array([*(center - h/2*binormal - w/2*tangent), 1], np.float32) )
    vert_list.append(np.array([*(center - h/2*binormal + w/2*tangent), 1], np.float32) )

    vert_texcoord_list = [] # 存放每个顶点在纹理平面中的坐标的列表
    vert_texcoord_list.append([1, 1])
    vert_texcoord_list.append([1, 0])
    vert_texcoord_list.append([0, 0])
    vert_texcoord_list.append([0, 1])

    vert_norm_list = [normal for _ in range(4)] # 存放每个顶点的法向量的列表
    vert_tangent_list = [tangent for _ in range(4)] # 存放每个顶点的切向量的列表

    surf_list = [[0, 1, 2], [0, 2, 3]] # 存放每个三角形的顶点索引的列表

    surf_norm_list = [normal for _ in range(2)] # 存放每个三角形的法向量的列表
    
    # 将每个三角形的顶点坐标、纹理坐标、法向量存入列表
    points, texcoords, surf_norms, vert_norms, vert_tangents= [], [], [], [], []
    for i in range(2):
        for j in range(3):
            points.append(vert_list[surf_list[i][j]])
            texcoords.append(vert_texcoord_list[surf_list[i][j]])
            surf_norms.append(surf_norm_list[i])
            vert_norms.append(vert_norm_list[surf_list[i][j]])
            vert_tangents.append(vert_tangent_list[surf_list[i][j]])
    
    return np.array(points, np.float32), np.array(texcoords, np.float32), np.array(surf_norms, np.float32), np.array(vert_norms, np.float32), np.array(vert_tangents, np.float32)

def sphere_cube(): # 立方体环境对象生成器
    # 生成一个 2x2x2 的正方体
    # 每个小正方形被分为两个三角形
    
    vert_list = [] # 存放顶点三维坐标的列表
    vert_list.append(np.array([1, 1, 1, 1], np.float32))
    vert_list.append(np.array([1, 1, -1, 1], np.float32))
    vert_list.append(np.array([1, -1, 1, 1], np.float32))
    vert_list.append(np.array([1, -1, -1, 1], np.float32))
    vert_list.append(np.array([-1, 1, 1, 1], np.float32))
    vert_list.append(np.array([-1, 1, -1, 1], np.float32))
    vert_list.append(np.array([-1, -1, 1, 1], np.float32))
    vert_list.append(np.array([-1, -1, -1, 1], np.float32))

    surf_list = []
    surf_list.append([0, 1, 2])
    surf_list.append([1, 2, 3])
    surf_list.append([0, 1, 4])
    surf_list.append([1, 4, 5])
    surf_list.append([0, 2, 4])
    surf_list.append([2, 4, 6])
    surf_list.append([1, 3, 5])
    surf_list.append([3, 5, 7])
    surf_list.append([2, 3, 6])
    surf_list.append([3, 6, 7])
    surf_list.append([4, 5, 6])
    surf_list.append([5, 6, 7])

    points = []
    for i in range(12):
        for j in range(3):
            points.append(vert_list[surf_list[i][j]])

    return np.array(points, np.float32)