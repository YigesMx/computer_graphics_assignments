from typing import List, Callable

import numpy as np
from numpy.typing import NDArray

from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import glm

class Tex(object):
    def __init__(self, uniform_name: str, img, type, bind, w = 1024, h = 1024):
        self.bind = bind
        self.type = type
        self.uniform_name = uniform_name

        self.texture = glGenTextures(1)

        if img is not None:
            w = img.shape[1]
            h = img.shape[0]
            texture_img = img
        else:
            texture_img = None
            
        # glActiveTexture(self.gl_bind)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, self.type, w, h, 0, self.type, GL_UNSIGNED_BYTE, texture_img)
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )
    
    @property
    def gl_bind(self):
        return GL_TEXTURE0 + self.bind

    def bind_uniform(self, shader_program):
        glActiveTexture(self.gl_bind)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(shader_program, self.uniform_name), self.bind)

class Obj(object):
    def __init__(self, points: NDArray[np.float32] = np.array([],np.float32), texcoords: NDArray[np.float32] = np.array([],np.float32), surf_norms: NDArray[np.float32] = np.array([],np.float32), vert_norms: NDArray[np.float32] = np.array([],np.float32), vert_tangents: NDArray[np.float32] = np.array([],np.float32)):
        self.points = points
        self.texcoords = texcoords
        self.surf_norms = surf_norms
        self.vert_norms = vert_norms
        self.vert_tangents = vert_tangents

        self.num_vertex = len(points)

        self.VAO = glGenVertexArrays(1) # Vertex Array Object
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1) # Vertex Buffer Object
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.points.nbytes + self.texcoords.nbytes + self.surf_norms.nbytes + self.vert_norms.nbytes + self.vert_tangents.nbytes, None, GL_STATIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.points.nbytes, points)
        glBufferSubData(GL_ARRAY_BUFFER, self.points.nbytes, self.texcoords.nbytes, texcoords)
        glBufferSubData(GL_ARRAY_BUFFER, self.points.nbytes + self.texcoords.nbytes, self.surf_norms.nbytes, surf_norms)
        glBufferSubData(GL_ARRAY_BUFFER, self.points.nbytes + self.texcoords.nbytes + self.surf_norms.nbytes, self.vert_norms.nbytes, vert_norms)
        glBufferSubData(GL_ARRAY_BUFFER, self.points.nbytes + self.texcoords.nbytes + self.surf_norms.nbytes + self.vert_norms.nbytes, self.vert_tangents.nbytes, vert_tangents)

        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, None) # 每个顶点属性数据大小为4，类型为float，不进行归一化，步长为16（即每个顶点属性数据之间相隔16个字节），偏移量为0
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(self.points.nbytes))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(self.points.nbytes + self.texcoords.nbytes))
        glEnableVertexAttribArray(2)

        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(self.points.nbytes + self.texcoords.nbytes + self.surf_norms.nbytes))
        glEnableVertexAttribArray(3)

        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(self.points.nbytes + self.texcoords.nbytes + self.surf_norms.nbytes + self.vert_norms.nbytes))
        glEnableVertexAttribArray(4)

class Instance(object):
    def __init__(self, Obj, texture = None, normal_map = None, bump_map = None, model_calc: Callable[[float], glm.mat4] = lambda t: glm.mat4(1.0)):
        self.Obj = Obj
        self.VAO = self.Obj.VAO
        self.num_vertex = self.Obj.num_vertex

        self.texture = texture
        self.normal_map = normal_map
        self.bump_map = bump_map

        self.model = model_calc
    
    def bind_mvps(self, shader_program, view, proj, time_count):
        model = self.model(time_count)
        mvp = proj * view * model

        mvp_loc = glGetUniformLocation(shader_program,"MVP")
        m_loc = glGetUniformLocation(shader_program, "M")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp))
        glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(model))
    
    def bind_light_mvps(self, shader_program, light_view, light_proj, time_count):
        model = self.model(time_count)
        mvp = light_proj * light_view * model

        mvp_loc = glGetUniformLocation(shader_program,"light_MVP")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp))
    
    def bind_texture_uniforms(self, shader_program):
        if self.texture is not None:
            self.texture.bind_uniform(shader_program)
        if self.normal_map is not None:
            self.normal_map.bind_uniform(shader_program)
        if self.bump_map is not None:
            self.bump_map.bind_uniform(shader_program)
