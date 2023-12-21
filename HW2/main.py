import numpy as np
from numpy.typing import NDArray

from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.GL import shaders

import glm

from enum import Enum

########################## global params ############################

# 窗口参数
window_width = 1280
window_height = 960

# 动画参数
fps = 30
tpf = int(1000.0 / fps)
rot_speed = 0.5
pause = False
time_count = 0

# 对象参数
sphere_n_surf = 16
earth_texture_name = 'earth_texture.jpg'
earth_bump_map_name = 'earth_bump.jpg'
earth_normal_map_name = 'earth_normal.jpg'

# 光源参数
light_pos = np.array([3, 3, 1], np.float32) #光源位置
light_target = np.array([0, 0, 0], np.float32) #光源方向
light_color = np.array([15, 15, 15], np.float32) #光源亮度
shadow_width = 2048
shadow_height = 2048

# 视点参数
move_step = 0.2 # 0 < s < 1
rotate_step = 5 # 0 < r < 10
# 运行时可调 wasdrf, qe, zc, x
view_pos = np.array([0, 0, 4], np.float32)
target = np.array([0, 0, 0], np.float32)
up = np.array([0, 1, 0], np.float32)
up_degree = 0
reset_type = 1

# 着色参数
class ShadingFrequencyType(Enum):
    NONE = 0
    FLAT = 1
    GOURAUD = 2
    PHONG = 3

class NormalType(Enum):
    DEFAULT = 0
    NORMAL_MAP = 1
    BUMP_MAP = 2

class TextureType(Enum):
    NONE = 0
    LOAD = 1

class ShadowType(Enum):
    NONE = 0
    TWO_PASS = 1

shading_freq = ShadingFrequencyType.PHONG # 运行时可调 0,1,2,3
normal_switch = NormalType.NORMAL_MAP # 运行时可调 n
texture_switch = TextureType.LOAD # 运行时可调 t
shadow_switch = ShadowType.TWO_PASS # 运行时可调 y

ambient_strength = 0.02
specular_strength = 0.8
specular_power = 64.0

bump_strength = 3.5

# 可见性参数
class VisibilityType(Enum):
    OBJ = 0
    BOTH = 1
    ENV = 2

visibility_switch = VisibilityType.BOTH # 运行时可调 v


########################## shaders ############################

shader_dir = './shaders/'

# 环境贴图着色器
VERTEX_SHADER_ENV = open(shader_dir + 'env.vert', 'r', encoding='utf-8').read()
FRAGMENT_SHADER_ENV = open(shader_dir + 'env.frag', 'r', encoding='utf-8').read()

# 阴影和普通物体渲染着色器，采用两次渲染，第一次渲染到深度缓冲区用于生成阴影，第二次渲染到颜色缓冲区并考虑阴影
# 第一次渲染的着色器 
VERTEX_SHADER_SHADOW = open(shader_dir + 'shadow.vert', 'r', encoding='utf-8').read()
FRAGMENT_SHADER_SHADOW = open(shader_dir + 'shadow.frag', 'r', encoding='utf-8').read()

# 第二次渲染的着色器
VERTEX_SHADER_COLOR = open(shader_dir + 'color.vert', 'r', encoding='utf-8').read()
FRAGMENT_SHADER_COLOR = open(shader_dir + 'color.frag', 'r', encoding='utf-8').read()

########################## object generators ############################

from objects import sphere, plane, sphere_cube

########################## texture generators ############################

from textures import checkborad_pattern

########################## classes ############################

from classes import Tex, Obj, Instance

########################## initialize ############################

def get_objs():
    earth_texture = Tex("texture_map", np.array(Image.open(earth_texture_name)), GL_RGB, 0)
    earth_normal_map = Tex("normal_map", np.array(Image.open(earth_normal_map_name)), GL_RGB, 1)
    earth_bump_map = Tex("bump_map", np.array(Image.open(earth_bump_map_name)), GL_RED, 2)

    # sphere objs
    sphere_obj = Obj(*sphere(sphere_n_surf, sphere_n_surf)) # 生成球体顶点数据

    sphere_ins1 = Instance(sphere_obj, # 生成球体实例1
                     texture = earth_texture,
                     normal_map = earth_normal_map,
                     bump_map = earth_bump_map,
                     model_calc = lambda t: glm.rotate(glm.mat4(1.0), glm.radians(rot_speed * t), glm.vec3(0.0, 1.0, 0.0))
                     # 针对球体实例1的模型变换（自转）
                 )
    
    def sphere2_model(t): # 针对球体实例2的模型变换（公转+平移+自转+缩放）
        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(-45), glm.vec3(0.0, 1.0, 0.0))
        model = glm.rotate(model, glm.radians(20), glm.vec3(1.0, 0.0, 0.0))
        model = glm.rotate(model, glm.radians(rot_speed * t), glm.vec3(0.0, 1.0, 0.0))
        model = glm.translate(model, glm.vec3(0.0, 0.0, 1.3))
        model = glm.rotate(model, glm.radians(rot_speed * t), glm.vec3(0.0, 1.0, 0.0))
        model = glm.scale(model, glm.vec3(0.2, 0.2, 0.2))
        return model
    
    sphere_ins2 = Instance(sphere_obj, # 生成球体实例2
                        texture = earth_texture, # 可以替换为不同的纹理，例如月亮（）
                        normal_map = earth_normal_map,
                        bump_map = earth_bump_map,
                        model_calc = sphere2_model
                    )

    # plane
    texture_img = checkborad_pattern() # 生成棋盘格纹理
    normal_map_img = np.zeros((256, 256, 3), 'uint8') # 生成无凹凸法线贴图
    for row in normal_map_img:
        for item in row:
            item[0] = 127
            item[1] = 127
            item[2] = 255
    bump_map_img = np.zeros((256, 256, 3), 'uint8') # 生成平坦的凹凸贴图
    
    plane_obj = Obj(*plane()) # 生成平面顶点数据

    def plane_model(t): # 针对平面的模型变换（平移+缩放）
        model = glm.mat4(1.0)
        model = glm.translate(model, glm.vec3(0.0, -1.1, 0.0))
        model = glm.scale(model, glm.vec3(5.0, 5.0, 5.0))
        return model

    plane_ins = Instance(plane_obj, # 生成平面实例
                    texture = Tex("texture_map", texture_img, GL_RGB, 3),
                    normal_map = Tex("normal_map", normal_map_img, GL_RGB, 4),
                    bump_map = Tex("bump_map", bump_map_img, GL_RED, 5),
                    model_calc = plane_model
                )
    
    #env
    env_obj = Obj(sphere_cube()) # 生成环境贴图顶点数据
    env_ins = Instance(env_obj, # 生成环境贴图实例
                    texture = earth_texture,
                    model_calc = lambda t: glm.scale(glm.mat4(1.0), glm.vec3(5.0, 5.0, 5.0)) # 缩放
                )
    
    return [sphere_ins1, sphere_ins2, plane_ins], env_ins

def get_shadow_utils():

    # 创建一个用于渲染灯视角下深度图像的帧缓冲区
    depth_map_fbo = glGenFramebuffers(1)
    
    # 创建一个深度纹理附件
    global shadow_width
    global shadow_height
    shadow_tex = Tex("shadow_map", None, GL_DEPTH_COMPONENT, 6, shadow_width, shadow_height)
    
    return depth_map_fbo, shadow_tex

def initialize_shader(vertex_shader_code, fragment_shader_code):

    vertexshader = shaders.compileShader(vertex_shader_code, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)

    shader_program = shaders.compileProgram(vertexshader, fragmentshader)
    return shader_program

########################## render utils ############################

def bind_global_uniforms(shader_program):
    global light_pos
    global light_color

    global view_pos

    global shading_freq
    global ambient_strength
    global specular_strength
    global specular_power

    global normal_switch
    global texture_switch
    global shadow_switch

    global bump_strength
    
    # 光源参数
    glUniform3f(glGetUniformLocation(shader_program,"light_pos"),
                light_pos[0], light_pos[1], light_pos[2])
    glUniform3f(glGetUniformLocation(shader_program,"light_color"),
                light_color[0], light_color[1], light_color[2])
    
    # 视点参数
    view_pos_loc = glGetUniformLocation(shader_program, "view_pos")
    glUniform3f(view_pos_loc, view_pos[0], view_pos[1], view_pos[2])

    # 着色参数
    shading_freq_loc = glGetUniformLocation(shader_program, "shading_freq")
    glUniform1i(shading_freq_loc, shading_freq.value)

    normal_switch_loc = glGetUniformLocation(shader_program, "normal_switch")
    glUniform1f(normal_switch_loc, normal_switch.value)

    texture_switch_loc = glGetUniformLocation(shader_program, "texture_switch")
    glUniform1f(texture_switch_loc, texture_switch.value)

    shadow_switch_loc = glGetUniformLocation(shader_program, "shadow_switch")
    glUniform1i(shadow_switch_loc, shadow_switch.value)

    ambient_strength_loc = glGetUniformLocation(shader_program, "ambient_strength")
    glUniform1f(ambient_strength_loc, ambient_strength)

    specular_strength_loc = glGetUniformLocation(shader_program, "specular_strength")
    glUniform1f(specular_strength_loc, specular_strength)

    specular_power_loc = glGetUniformLocation(shader_program, "specular_power")
    glUniform1f(specular_power_loc, specular_power)
    
    bump_strength_loc = glGetUniformLocation(shader_program, "bump_strength")
    glUniform1f(bump_strength_loc, bump_strength)


def render_obj(shader_program, obj, view, proj, light_view, light_proj, shadow_map, time_count):
    glUseProgram(shader_program)

    # 传递全局参数
    bind_global_uniforms(shader_program)
    
    # 传递MVP矩阵
    obj.bind_mvps(shader_program, view, proj, time_count)
    obj.bind_light_mvps(shader_program, light_view, light_proj, time_count)
    
    # 传递纹理
    shadow_map.bind_uniform(shader_program)
    obj.bind_texture_uniforms(shader_program)
    
    # 绑定顶点数组对象
    glBindVertexArray(obj.VAO)
    # 绘制
    glDrawArrays(GL_TRIANGLES, 0, obj.num_vertex)
    # 解绑
    glUseProgram(0)

def render_shadow(shader_program, obj, light_view, light_proj, time_count):
    glUseProgram(shader_program)
    
    # 传递MVP矩阵
    obj.bind_light_mvps(shader_program, light_view, light_proj, time_count)

    # 绑定顶点数组对象
    glBindVertexArray(obj.VAO)
    # 绘制
    glDrawArrays(GL_TRIANGLES, 0, obj.num_vertex)
    # 解绑
    glUseProgram(0)

def render_env(shader_program, obj, view, proj, time_count):
    glUseProgram(shader_program)
    
    # 传递MVP矩阵
    obj.bind_mvps(shader_program, view, proj, time_count)

    # 传递纹理
    obj.bind_texture_uniforms(shader_program)
    
    # 绑定顶点数组对象
    glBindVertexArray(obj.VAO)
    # 绘制
    glDrawArrays(GL_TRIANGLES, 0, obj.num_vertex)
    # 解绑
    glUseProgram(0)


def clear_buffer():
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def get_renderer(env_shader_program, shadow_shader_program, color_shader_program, env, objs, depth_map_fbo, shadow_tex):
    def render():
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)
        global time_count
        
        # 第一次渲染，生成深度图（灯视角）

        # 将深度纹理附件附加到帧缓冲区
        glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_tex.texture, 0)
        glDrawBuffer(GL_NONE) # 不渲染颜色数据
        glReadBuffer(GL_NONE)

        global light_pos
        global light_target
        global shadow_width
        global shadow_height
        glViewport(0, 0, shadow_width, shadow_height)

        clear_buffer() # 清空颜色缓冲区和深度缓冲区

        light_proj = glm.perspective(glm.radians(90.0),float(shadow_width)/float(shadow_height),0.1,50.0)
        light_view = glm.lookAt(glm.vec3(*light_pos), glm.vec3(*light_target), glm.vec3(0.0, 1.0, 0.0))

        if visibility_switch.value <=1:
            for obj in objs:
                render_shadow(shadow_shader_program, obj, light_view, light_proj, time_count)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # 第二次渲染，正常渲染

        clear_buffer() # 清空颜色缓冲区和深度缓冲区
        global view_pos
        global target
        global up
        global window_height
        global window_width
        glViewport(0, 0, window_width, window_height)
        
        proj = glm.perspective(glm.radians(60.0),float(window_width)/float(window_height),0.1,50.0)
        view = glm.lookAt(glm.vec3(*view_pos), glm.vec3(*target),glm.vec3(*up))

        if visibility_switch.value <=1:
            for obj in objs:
                render_obj(color_shader_program, obj, view, proj, light_view, light_proj, shadow_tex, time_count)
        
        # 环境贴图渲染
        glDisable(GL_CULL_FACE)

        if visibility_switch.value >=1:
            render_env(env_shader_program, env, view, proj, time_count)

        glEnable(GL_CULL_FACE)
        glutSwapBuffers() # 交换前后缓冲区
    
    return render

########################## animation & interaction ############################

def animate(value):
    global time_count
    global pause

    glutPostRedisplay()

    glutTimerFunc(tpf, animate, 0)
    
    if not pause:
        time_count = time_count+1.0

def keyboard(key, x, y):
    global shading_freq
    global normal_switch
    global texture_switch
    global shadow_switch

    if key == b'0':
        shading_freq = ShadingFrequencyType.NONE
    elif key == b'1':
        shading_freq = ShadingFrequencyType.FLAT
    elif key == b'2':
        shading_freq = ShadingFrequencyType.GOURAUD
    elif key == b'3':
        shading_freq = ShadingFrequencyType.PHONG

    elif key == b'n':
        normal_switch = NormalType((normal_switch.value + 1) % 3)
    elif key == b't':
        texture_switch = TextureType((texture_switch.value + 1) % 2)
    elif key == b'y':
        shadow_switch = ShadowType((shadow_switch.value + 1) % 2)
    
    global pause
    if key == b' ':
        pause = not pause
    
    global view_pos
    global target
    global up
    global up_degree
    direction = target-view_pos
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)
    front = np.cross(up, right)
    # wasdrf控制视点沿前后左右上下移动（同时平移view_pos和target），0.1步长
    if key == b'w':
        view_pos += move_step * front
        target += move_step * front
    elif key == b's':
        view_pos -= move_step * front
        target -= move_step * front
    elif key == b'a':
        view_pos -= move_step * right
        target -= move_step * right
    elif key == b'd':
        view_pos += move_step * right
        target += move_step * right
    elif key == b'r':
        view_pos += move_step * up
        target += move_step * up
    elif key == b'f':
        view_pos -= move_step * up
        target -= move_step * up
    # qe控制target绕视点的up轴旋转，5度步长
    elif key == b'q':
        target = view_pos + glm.rotate(direction, glm.radians(rotate_step), up)
    elif key == b'e':
        target = view_pos + glm.rotate(direction, glm.radians(-rotate_step), up)
    # zc控制target绕视点的 right 轴旋转，5度步长，不能超过175度
    elif key == b'z':
        if up_degree<=80:
            up_degree+=rotate_step
            target = view_pos + glm.rotate(direction, glm.radians(rotate_step), right)
    elif key == b'c':
        if up_degree>=-80:
            up_degree-=rotate_step
            target = view_pos + glm.rotate(direction, glm.radians(-rotate_step), right)
    # 重置
    global reset_type
    if key == b'x':
        if reset_type == 0:
            view_pos = np.array([0.0, 0.0, 4.0], np.float32)
            target = np.array([0.0, 0.0, 0.0], np.float32)
            up = np.array([0.0, 1.0, 0.0], np.float32)
            up_degree = 0
            reset_type ^= 1
        else:
            view_pos = np.array([0.0, 0.0, 0.0], np.float32)
            target = np.array([0.0, 0.0, -4.0], np.float32)
            up = np.array([0.0, 1.0, 0.0], np.float32)
            up_degree = 0
            reset_type ^= 1
    
    global visibility_switch
    if key == b'v':
        visibility_switch = VisibilityType((visibility_switch.value + 1) % 3)
    
    print(shading_freq)
    print(normal_switch)
    print(texture_switch)
    print(shadow_switch)
    print()

########################## main ############################

def main():
    global window_width
    global window_height
   
    glutInit([])
    glutSetOption(GLUT_MULTISAMPLE, 16)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE )
    glutInitWindowSize(window_width, window_height)
    glutCreateWindow("Earth")

    env_shader_program = initialize_shader(VERTEX_SHADER_ENV, FRAGMENT_SHADER_ENV)
    shadow_shader_program = initialize_shader(VERTEX_SHADER_SHADOW, FRAGMENT_SHADER_SHADOW)
    color_shader_program = initialize_shader(VERTEX_SHADER_COLOR, FRAGMENT_SHADER_COLOR)

    objs, env= get_objs()
    depth_map_fbo, shadow_tex = get_shadow_utils()

    glutDisplayFunc(get_renderer(env_shader_program, shadow_shader_program ,color_shader_program, env, objs, depth_map_fbo, shadow_tex))
    glutKeyboardFunc(keyboard)
    glutTimerFunc(tpf, animate, 0)

    glutMainLoop()

if __name__ == '__main__':
    main()