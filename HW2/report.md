# Computer Graphics HW2 Report

## 1. Introduction

本次作业主要需要完成：物体建模与纹理贴贴图映射、环境贴图映射、光照模型计算、深度/法线贴图映射、简单阴影实现，五个部分

下面我将先对我的代码结构和运行方式进行简单的介绍，然后再对每个部分的实现进行详细的介绍。

## 2. Code Structure

本次作业的代码结构如下（注意，这里为了方便复用，创建了许多通用的类和方法，方便代码的理解，详细的注释已经在代码中给出。）

- global params：这部分主要是一些全局变量与参数
    - 窗口参数（渲染大小）
    - 动画参数（帧率、动画速度）
    - 对象参数（球体曲面细分数、贴图文件名）
    - 光源参数（光源位置、光源颜色/强度、阴影参数）
    - 视点参数（视点参数）
    - 着色参数（光照模型参数、材质参数、法线参数等）
    - 可见性参数
    - 注意，这部分的全局参数一部分可以在运行时进行更改，从而实现渲染效果的快速改变，在注释中已表明
        - 通过 `wasdrf` 可以前后左右上下移动视点，通过 qe 可以左右旋转视角，通过 `zc` 可以上下旋转视角，通过 `x` 可以快速回到初始视角，或切换两个初始点（其中，一个为侧面看球体的视角，一个为在原点的视角-方便观察环境贴图）
        - 通过数字键可以切换光照模型
            - `0 - None`
            - `1 - Flat`
            - `2 - Gouraud`
            - `3 - Phong`
        - 通过 `n` 键可以切换法线开关（无/法线贴图/高度贴图）
        - 通过 `t` 键可以切换材质开关（有/无）
        - 通过 `y` 键可以切换阴影开关（有/无）
        - 通过 `v` 键可以切换物体/环境的可见性
- shader codes：这部分主要是一些着色器代码
    - environment shader - 环境贴图着色器
    - shadow shader - 阴影着色器
    - color shader - 物体着色器
- object generators：这部分主要是一些物体生成器
    - sphere generator - 球体生成器
    - plane generator - 平面生成器
    - cube generator - 立方体生成器
- texture generators：这部分主要是纹理生成器
    - checkerboard generator - 棋盘格生成器
- classes：这部分主要是一些类
    - Tex - 纹理类：通过图片创建纹理，可以指定存储的纹理单元，提供绑定纹理到指定着色器的方法
    - Obj - 物体类：通过生成的模型信息创建物体的VAO，这里为了方便实现每个VAO独享一个VBO
    - Instance - 实例类：通过物体和纹理创建实例，可以实现VAO和Tex的共用，对每个实例可以使用不同的 model 实现各个实例的不同位移、旋转、缩放、以及动画等
- initialize：这部分主要是一些初始化
    - get_objs - 获取物体
        - 对通用材质进行定义
        - 对通用物体模型进行定义
        - 实例化物体，并提供此实例的材质、model矩阵计算方法（可以实现不同实例的特性处理，包括动画）
    - get_shadow_utils - 获取阴影工具
        - 生成用于存储的深度缓冲区
        - 生成阴影贴图（深度纹理附件）
    - initialize_shader - 通过着色器代码生成着色器
- render utils
    - bind_global_uniforms - 绑定全局参数
    - render_obj - 渲染物体
    - render_shadow - 渲染阴影
    - render_env - 渲染环境
    - clear_buffers - 清空缓冲区
    - get_renderer - 获取单次渲染的函数
        - render - 渲染阴影、渲染物体、渲染环境
- animation & interaction
    - animate - 渲染循环
    - keyboard - 键盘回调函数：用于修改全局参数
- main
    - main - 主函数

运行后可以通过上面介绍的按键进行交互，从而实现不同的渲染效果（所有五个问题都在此代码中进行了实现）。

## 3. Implementation

### Q1. Object Modeling & Texture Mapping

> 将球面进行三角化（用多个三角形表示球面），并将图片earthmap.jpg 作为纹理图贴到球面上进行绘制。

球体生成的主要思想就是：

- 生成所有顶点及顶点信息（所有需要生成的信息通过球坐标系即可容易得到）
    - 顶点坐标
        - $x = r \sin \theta \cos \phi$
        - $y = r \cos \theta$
        - $z = r \sin \theta \sin \phi$
    - 纹理坐标（Q1 进行贴图映射需要）
        - $u = 1 - \frac{\phi}{2\pi}$
        - $v = \frac{\theta}{\pi}$
    - 顶点法线（Q3 进行光照渲染需要）
        - $\vec{n} = (x, y, z)$
    - 顶点切线（Q4 进行法线贴图映射需要）
        - $\theta_t = \theta + \frac{\pi}{2}$
        - $\vec{t} = (-\sin \theta_t \sin \phi, \cos \theta_t, -\sin \theta_t \cos \phi)$
- 生成所有三角面及面信息
    - 面顶点
        - 根据 OpenGL 的默认顺序，**逆时针**选取相邻三个点为一个三角面
    - 面法线（Q3 进行光照渲染需要）
        - 根据三角形的顶点，叉乘计算三角形法线
- 通过上面的信息生成待渲染的三角形顶点序列
    - 将面信息与点信息对应
    - 生成三角形顶点序列

球体模型生成器的实现如下：

```python
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
```

然后理论上经过简单的Shader就可以得到物体和材质映射的渲染效果：

Vertex Shader:
```glsl
#version 410

layout(location = 0) in vec4 pos;
layout(location = 1) in vec2 texcoord_ori;

uniform mat4 MVP;
uniform mat4 M;

out vec2 texcoord;
void main() {
    texcoord = texcoord_ori;
    gl_Position = MVP * pos;
}
```

Fragment Shader:
```glsl
#version 410

in vec2 texcoord; //纹理坐标

uniform sampler2D tex; //纹理

void main() {
    gl_FragColor = texture(tex, texcoord);
}
```

效果：

![Q1](./src/1.gif)

### Q2. Environment Mapping

> 将图片earthmap.jpg 当做环境贴图，贴在一个立方体上。

正方体的生成只需要手动生成八个顶点、十二个三角面即可

并且环境贴图不需要法向量等信息，纹理坐标可以直接在 shader 中进行计算，所以非常简单

```python
def sphere_cube(): 
    
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
```

然后通过在 shader 中，通过球坐标系，根据立方体上的点计算球面上的纹理坐标即可

$$
\begin{aligned}
\text{texcoord}.x &= - \arctan(z,x) / 2\pi \\
\text{texcoord}.y &= \arctan(y, \text{length}(x,z)) / \pi + 0.5
\end{aligned}
$$

**Vertex Shader:**
```glsl
#version 410
layout(location = 0) in vec4 pos;

uniform mat4 MVP;
uniform mat4 M;

out vec3 frag_pos;

void main(){
    frag_pos = vec3(M * pos);
    gl_Position = MVP * pos;
}
```

**Fragment Shader:**
```glsl
#version 410
in vec3 frag_pos;

uniform sampler2D texture;

void main(){
    float lon = atan(frag_pos.z, frag_pos.x);
    float lat = atan(-frag_pos.y, length(frag_pos.xz));
    vec2 texcoord = vec2(-lon, lat) / vec2(2 * 3.1415926535, 3.1415926535) + vec2(0, 0.5);

    gl_FragColor = texture(texture, texcoord);
}
```

效果：

![Q2](./src/2.gif)

### Q3. Lighting Model

> 实现光照计算（Blinn-Phong）模型。几何可以用 1 中构造的球，方便计算法向量。实现三种不同采样频率（三角形、顶点、像素）。

这里主要实现 Blinn-Phong 光照模型，即由环境光、漫反射、镜面反射组成的光照模型。

$$
\begin{aligned}
L&=L_a+L_d+L_s \\
&=k_a I_a + k_d \left(\frac{I}{r^2}\right) \max(0,\mathbf{n} \cdot \mathbf{l}) + k_s \left(\frac{I}{r^2}\right) \max(0,\mathbf{n} \cdot \mathbf{h})^p
\end{aligned}
$$

其中，$L$ 为最终的光照强度，$L_a$ 为环境光照强度，$L_d$ 为漫反射光照强度，$L_s$ 为镜面反射光照强度，$k_a$ 为环境光系数，$k_d$ 为漫反射系数，$k_s$ 为镜面反射系数，$I_a$ 为环境光强度，$I$ 为光源强度，$r$ 为光源距离，$\mathbf{n}$ 为法向量，$\mathbf{l}$ 为光源方向，$\mathbf{h}$ 为半程向量，$p$ 为镜面反射强度。

同时，主要实现三种采样频率：
- Flat：对每个三角面进行一次采样，然后对三角面上的所有点进行着色
- Gouraud：对每个顶点进行一次采样，然后对每个顶点进行着色，最后通过插值得到三角面上的所有点的颜色
- Phong：对每个像素进行一次采样，将顶点得到的法线等信息插值后，对每个像素进行着色

这三种频率可以在程序运行时通过 0123 进行切换

在 Q1 中，我们已经在模型中生成了所需要的法线信息，下面直接介绍如何通过 shader 实现这三种采样频率的光照模型

#### Flat
在Vertex Shader中，我们直接通过面法向量来计算光照，然后将计算得到的光照以flat的方式传递给 Fragment Shader 进行光照模型的着色即可

**Vertex Shader:**
```glsl
#version 410

layout(location = 0) in vec4 pos;
layout(location = 1) in vec2 texcoord_ori;
layout(location = 2) in vec3 surf_norm;

uniform mat4 MVP;
uniform mat4 M;

uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 view_pos;

out vec2 texcoord;
flat out vec3 ambient;
flat out vec3 flat_diffuse;
flat out vec3 flat_specular;

void main() {
    texcoord = texcoord_ori;
    gl_Position = MVP * pos;
    frag_pos = vec3(M * pos);

    vec3 light_vec = light_pos - frag_pos;
    vec3 light_dir = normalize(light_vec);
    float r2 = dot(light_vec, light_vec);
    vec3 surf_norm = normalize(mat3(M) * surf_norm);
    flat_diffuse = light_color * max(dot(surf_norm, light_dir), 0.0) / r2;
    
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(surf_norm, halfway_dir), 0.0), specular_power);
    flat_specular = specular_strength * spec * light_color / r2;
}
```

**Fragment Shader:**
```glsl
#version 410

in vec2 texcoord;
flat in vec3 ambient;
flat in vec3 flat_diffuse;
flat in vec3 flat_specular;

uniform sampler2D tex;

void main() {
    vec3 tex = texture(tex, texcoord).rgb;
    vec3 color = tex * (ambient + flat_diffuse + flat_specular);
    gl_FragColor = vec4(color, 1.0);
}
```

当然也可以再引入 ka kd ks 等参数，通过 uniform 传递给 shader，实现对三种光照参数的控制

这里为了简单，没有引入这些参数，下面也一样。

#### Gouraud

在Vertex Shader中，我们直接通过顶点法向量来计算光照，然后将计算得到的光照自动插值传递给 Fragment Shader 进行光照模型的着色即可

```glsl
#version 410

layout(location = 0) in vec4 pos;
layout(location = 1) in vec2 texcoord_ori;
layout(location = 2) in vec3 vert_norm;

uniform mat4 MVP;
uniform mat4 M;

uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 view_pos;

out vec2 texcoord;
flat out vec3 ambient;
out vec3 diffuse;
out vec3 specular;

void main() {
    texcoord = texcoord_ori;
    gl_Position = MVP * pos;
    frag_pos = vec3(M * pos);

    ambient = light_color * ambient_strength;

    vec3 light_vec = light_pos - frag_pos;
    vec3 light_dir = normalize(light_vec);
    float r2 = dot(light_vec, light_vec);
    vec3 vert_norm = normalize(mat3(M) * vert_norm);
    diffuse = light_color * max(dot(vert_norm, light_dir), 0.0) / r2;
    
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(vert_norm, halfway_dir), 0.0), specular_power);
    specular = specular_strength * spec * light_color / r2;
}
```

**Fragment Shader:**
```glsl
#version 410

in vec2 texcoord;
flat in vec3 ambient;
in vec3 diffuse;
in vec3 specular;

uniform sampler2D tex;

void main() {
    vec3 tex = texture(tex, texcoord).rgb;
    vec3 color = tex * (ambient + diffuse + specular);
    gl_FragColor = vec4(color, 1.0);
}
```

#### Phong

在Vertex Shader中，我们计算出该点的 TBN 矩阵，并将光源、视点、法向量等信息转换到切线空间中再传递给 Fragment Shader

这里使用切线空间，是为了方便后面使用法线和高度贴图做准备（因为从贴图获取的法线本身就是在原切线空间上修改后的法线，这样方便使用），如果只考虑实现光照模型，可以使用和 Flat 以及 Gouraud 一样的方式

在Fragment Shader中，通过OpenGL自动插值得到每个片元的切线空间中的光源、视点、法向量等信息，再以这些插值后的信息计算当前片元的光照模型即可

**Vertex Shader:**
```glsl
#version 410

layout(location = 0) in vec4 pos;
layout(location = 1) in vec2 texcoord_ori;
layout(location = 2) in vec3 vert_norm;
layout(location = 3) in vec3 vert_tangent;

uniform mat4 MVP;
uniform mat4 M;

uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 view_pos;

out vec2 texcoord;
flat out vec3 ambient;

out vec3 tangent_frag_pos;
out vec3 tangent_light_pos;
out vec3 tangent_view_pos;

void main() {
    texcoord = texcoord_ori;
    gl_Position = MVP * pos;
    frag_pos = vec3(M * pos);

    ambient = light_color * ambient_strength;

    vec3 frag_norm = normalize(mat3(M) * vert_norm);

    //TNB
    vec3 T = normalize(mat3(M) * vert_tangent);
    vec3 B = normalize(cross(frag_norm, T));
    mat3 TBN = transpose(mat3(T, B, frag_norm));

    tangent_frag_pos = TBN * frag_pos;
    tangent_light_pos = TBN * light_pos;
    tangent_view_pos = TBN * view_pos;
}
```

**Fragment Shader:**
```glsl
#version 410

in vec2 texcoord;
flat in vec3 ambient;

in vec3 tangent_frag_pos;
in vec3 tangent_light_pos;
in vec3 tangent_view_pos;

uniform sampler2D tex;

void main() {
    vec3 tex = texture(tex, texcoord).rgb;
    
    vec3 light_vec = tangent_light_pos - tangent_frag_pos;
    vec3 light_dir = normalize(light_vec);

    vec3 frag_norm = vec3(0,0,1); // 切线空间中的法向量

    float r2 = dot(light_vec, light_vec);
    vec3 diffuse = light_color * max(dot(frag_norm, light_dir), 0.0) / r2;

    vec3 view_dir = normalize(tangent_view_pos - tangent_frag_pos);
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(frag_norm, halfway_dir), 0.0), specular_power);
    vec3 specular = specular_strength * spec * light_color / r2;

    color = tex * (ambient + diffuse + specular);
}
```

#### 光照模型效果

在最终的代码中，我将三个光照模型的效果整合进了一个shader中，可以通过按键 0123 进行切换：

![Q3](./src/3.gif)

### Q4. Normal Mapping

> Bump Mapping 的实现，利用纹理实现 Q1 中球面的凹凸效果。

实现了两种模式，一种是通过法线贴图实现，一种是通过高度贴图实现。其中法线贴图可以通过对高度贴图求导得到，我也编写了转换程序，见 `bump2normal.py`。

其实只需要在 Phong 光照模型的基础上，在shader中将法向量替换为从贴图中获取的法向量即可，既可以直接从法线贴图中获取，也可以从bumpmap的导数或者说微分中获取：

** In Fragment Shader:**
```glsl
//...
void main(){
    // ...

    vec3 frag_norm = vec3(0,0,1); //片段法向量，切线空间
    
    // 将法向量替换为从贴图中获取的法向量即可：
    if (normal_switch == 0) {
        // 无法线修改贴图
    } else if (normal_switch == 1) {
        // 法线贴图
        vec3 normal_map_rgb = texture(normal_map, texcoord).rgb;
        vec3 normal_map_xyz = normalize((normal_map_rgb - vec3(0.5,0.5,0.5))*2);
        frag_norm = normal_map_xyz;
    } else if (normal_switch == 2) {
        // 凹凸贴图
        float bump_map = texture(bump_map, texcoord).r;
        //对高度贴图求导
        float dBdu = dFdx(bump_map);
        float dBdv = -dFdy(bump_map);
        //计算切线空间下的法向量
        vec3 N = normalize(vec3(-dBdu * bump_strength, -dBdv * bump_strength, 1.0));
        frag_norm = N;
    }

    //...
}
```

效果：（可以通过 n 键切换法线计算方式或关闭法线，图中包含关/normalmap/bumpmap三种模式）

![Q4](./src/4.gif)

### Q5. Shadow

> 在球面的下方，放一个大的平面。然后实现阴影的效果。

这是整个作业最麻烦的部分，因为它需要对整个渲染过程进行扩充，然而在实现上面的功能后代码已经非常冗长了，我也是在完成这个部分之前重构了代码，整合前面的 shader，并对代码进行了重构封装，通过面向对象的方式，方便加入多个物体，使用多个shader。这一次封装也让我更加熟悉了OpenGL的shader流程，对图形管线有了更深的理解。

总的来说，实现阴影需要以下几个步骤：

- 创建光源视角下的深度纹理
- 使用深度纹理为深度缓冲区从光源视角，使用简单的 shader 进行一次渲染，得到有深度缓冲区信息的深度纹理（1st pass）
- 将深度纹理作为一个材质传入正常的 shader
- 在正常的 shader 中，同时计算出片元在灯光视角下的深度，与深度纹理中的深度值进行比较，如果当前片元的深度值大于深度纹理中的深度值，则说明当前片元被遮挡，即在阴影中，否则不在阴影中（2nd pass）

这部分涉及的代码众多，可以直接在代码中查看，这里只介绍一下主要的思路。

此外，直接完成上述步骤会出现大量摩尔纹的情况，这是因为深度贴图直接映射到表面会有锯齿误差，灯光角度越大问题越明显，所以可以通过在比较深度时加入一个随灯光角度变化的极小的误差值来解决这一问题。

效果：

![Q5](./src/5.gif)

## Summary

最后，可以看一看整合后的效果：

![Scene](src/6.gif)