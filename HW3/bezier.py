import os

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
from scipy.special import comb

def color_getter():
    color_list = {
        'bg': np.array((47, 53, 66)), # FOREST BLUES
        'line': np.array((56, 173, 169)), # WATERFALL
        'dot': np.array((130, 204, 221)), # SPARY
        1: np.array((156, 136, 255)),
        2: np.array((246, 185, 59)),
        3: np.array((248, 194, 145)),
        4: np.array((235, 47, 6)),
    }

    # normalize
    for key in color_list:
        color_list[key] = color_list[key] / 255.0

    def get_color(clr):

        # if clr is digit
        if isinstance(clr, int):
            return color_list[clr%4+1]
        # if clr is str
        else:
            return color_list[clr]
    
    return get_color

get_color = color_getter()

class Window(object):
    """An abstract GLUT window with enhanced mouse controls."""

    def __init__(self, title="Untitled Window", width=800, height=800, ortho=False):
        self.ortho = ortho
        self.width = width
        self.height = height
        self.rotation = [0, 0]  # Rotation angles for x and y axes
        self.translation = [0, 0, 5]  # Translation distances for x, y, and z axes
        self.ortho_zoom = 1.0  # Zoom factor
        self.min_zoom = 0.01  # Minimum zoom factor

        glutInit()
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(title)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glClearColor(*get_color('bg'), 0)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutDisplayFunc(self.display)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.mouse_motion)
        glShadeModel(GL_FLAT)

        # 抗锯齿
        glLineWidth(0.5)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def keyboard(self, key, mouseX, mouseY):
        '''Call the code mapped to the pressed key.'''
        
        if key == b'q':
            glutDestroyWindow(glutGetWindow())
            os._exit(0)
            
        glutPostRedisplay()

    def mouse(self, button, state, x, y):
        '''Handle mouse clicking.'''
        self.mouse_button = button
        self.mouse_state = state
        self.last_mouse_pos = (x, y)
        glutPostRedisplay()

    def mouse_motion(self, x, y):
        '''Handle mouse motion for rotation and translation.'''
        dx, dy = x - self.last_mouse_pos[0], y - self.last_mouse_pos[1]
        if self.mouse_button == GLUT_LEFT_BUTTON:
            self.rotation[0] += dy
            self.rotation[1] += dx
        elif self.mouse_button == GLUT_MIDDLE_BUTTON:
            self.translation[0] -= dx * 0.01
            self.translation[1] += dy * 0.01
        elif self.mouse_button == GLUT_RIGHT_BUTTON:
            self.translation[2] += dy * 0.01
            self.ortho_zoom += dy * 0.01
            if self.ortho_zoom < self.min_zoom:
                self.ortho_zoom = self.min_zoom
        self.last_mouse_pos = (x, y)
        glutPostRedisplay()

    def reshape(self, width, height):
        '''Recalculate the clipping window when the GLUT window is resized.'''
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = float(self.width) / float(self.height)
        if self.ortho:
            glOrtho(-2.0 * self.ortho_zoom , 2.0 * self.ortho_zoom , -2.0 * aspect * self.ortho_zoom, 2.0 * aspect * self.ortho_zoom, -20.0, 20.0)
        else:
            gluPerspective(60.0 , aspect, 0.1, 20)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def display(self):
        '''Children implement this to define their rendering behavior.'''
        raise NotImplementedError

    def run(self):
        """Start up the main loop."""
        glutMainLoop()
        
        

class BezierCurve(Window):
    """Use evaluators to draw a Bezier curve."""

    def __init__(self, curves = None, ortho=False):
        """Constructor"""
        super(BezierCurve, self).__init__("Bezier Curve", ortho=ortho)
        
        self.curves = [
            ((-3 / 5.0, -4 / 5.0, 0), (-2 / 5.0, 4 / 5.0, 0), (2 / 5.0, -4 / 5.0, 0), (3 / 5.0, 4 / 5.0, 0))
        ] if curves is None else curves

        glClearColor(*get_color('bg'), 0)
        glShadeModel(GL_FLAT)
    
    """
    递归求解 n 个控制点的 bezier 曲线
    """

    def evalCoord1f(self, controlPoints, t):
        if len(controlPoints) == 1:
            return controlPoints[0]
        else:
            return (1 - t) * self.evalCoord1f(controlPoints[:-1], t) + t * self.evalCoord1f(controlPoints[1:], t)
    
    """
    bersntein 多项式 求解 bezier 曲线
    """

    # def bernstein(self, n, i, t):
    #     return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    # def evalCoord1f(self, controlPoints, t):
    #     point = [0, 0, 0]
    #     n = len(controlPoints)
    #     for i in range(len(controlPoints)):
    #         point[0] += self.bernstein(n - 1, i, t) * controlPoints[i][0]
    #         point[1] += self.bernstein(n - 1, i, t) * controlPoints[i][1]
    #         point[2] += self.bernstein(n - 1, i, t) * controlPoints[i][2]
    #     return point

    def display(self):
        """Display the control points as dots."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        aspect = float(self.width) / float(self.height)
        if self.ortho:
            glOrtho(-2.0 * self.ortho_zoom , 2.0 * self.ortho_zoom , -2.0 * aspect * self.ortho_zoom, 2.0 * aspect * self.ortho_zoom, -20.0, 20.0)

        # Set the camera position and orientation
        gluLookAt(self.translation[0], self.translation[1], self.translation[2],
                self.translation[0], self.translation[1], 0,
                0, 1, 0)

        # Apply the rotation
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)

        for idx ,curve in enumerate(self.curves):
            glColor3f(*get_color('line'))
            controlPoints = np.array(curve)

            # 下面一段请在1.2作业中注释
            # glMap1f(GL_MAP1_VERTEX_3, 0, 1, controlPoints)
            # glEnable(GL_MAP1_VERTEX_3)
            # glBegin(GL_LINE_STRIP)
            # for i in range(31):
            #     glEvalCoord1f(float(i) / 30)
            # glEnd()            

            # 1.2
            glBegin(GL_LINE_STRIP)
            for i in range(31):
                t = float(i) / 30
                x, y, z = self.evalCoord1f(controlPoints, t)
                glVertex3f(x, y, z)
            glEnd()

            glPointSize(5)
            glBegin(GL_POINTS)
            glColor3f(*get_color(idx))
            for point in controlPoints:
                glVertex3fv(point)
            glEnd()

        glFlush()


class BezierSurface(Window):
    """Use evaluators to draw a Bezier Surface."""

    def __init__(self, surfaces = None, ortho=False):
        """Constructor"""
        super(BezierSurface, self).__init__("Bezier Surface", ortho=ortho)
        self.surfaces = [
            (
                [
                    [-1.5, -1.5, 2.0],
                    [-0.5, -1.5, 2.0],
                    [0.5, -1.5, -1.0],
                    [1.5, -1.5, 2.0]
                ],
                [
                    [-1.5, -0.5, 1.0],
                    [-0.5, 1.5, 2.0],
                    [0.5, 0.5, 1.0],
                    [1.5, -0.5, -1.0]
                ],
                [
                    [-1.5, 0.5, 2.0],
                    [-0.5, 0.5, 1.0],
                    [0.5, 0.5, 3.0],
                    [1.5, -1.5, 1.5]
                ],
                [
                    [-1.5, 1.5, -2.0],
                    [-0.5, 1.5, -2.0],
                    [0.5, 0.5, 1.0],
                    [1.5, 1.5, -1.0]
                ]
            )
        ] if surfaces is None else surfaces

        glClearColor(*get_color('bg'), 0)

        # 下面一段请在作业2.2时注释掉
        # ambient = [0.2, 0.3, 0.4, 1.0]
        # position = [0.0, 5.0, 5.0, 1.0]
        # mat_diffuse = [0.1, 0.3, 0.4, 0.1]
        # mat_specular = [1.0, 1.0, 1.0, 0.1]
        # mat_shininess = [1.0]
        # glEnable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        # glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        # glLightfv(GL_LIGHT0, GL_POSITION, position)
        # glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
        # glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        # glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)
        # glShadeModel(GL_SMOOTH)
    
    """
    递归求解 bezier 曲面
    """

    def evalCoord1f(self, controlPoints, t):
        if len(controlPoints) == 1:
            return controlPoints[0]
        else:
            return (1 - t) * self.evalCoord1f(controlPoints[:-1], t) + t * self.evalCoord1f(controlPoints[1:], t)

    def evalCoord2f(self, controlPoints, u, v):
        # First, compute intermediate points for each row
        intermediatePoints = []
        for row in controlPoints:
            point = self.evalCoord1f(row, u)
            intermediatePoints.append(point)

        # Now, compute the final point using the intermediate points along v
        return self.evalCoord1f(intermediatePoints, v)
    
    """
    用 bernstein 多项式求解 bezier 曲面
    """

    # def bernstein(self, n, i, t):
    #     return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    # def evalCoord2f(self, controlPoints, u, v):
    #     # First, compute intermediate points for each row
    #     intermediatePoints = []
    #     n = len(controlPoints[0])
    #     m = len(controlPoints)
    #     for row in controlPoints:
    #         point = [0, 0, 0]
    #         for i in range(n):
    #             point[0] += self.bernstein(n - 1, i, u) * row[i][0]
    #             point[1] += self.bernstein(n - 1, i, u) * row[i][1]
    #             point[2] += self.bernstein(n - 1, i, u) * row[i][2]
    #         intermediatePoints.append(point)

    #     # Now, compute the final point using the intermediate points along v
    #     point = [0, 0, 0]
    #     for i in range(m):
    #         point[0] += self.bernstein(m - 1, i, v) * intermediatePoints[i][0]
    #         point[1] += self.bernstein(m - 1, i, v) * intermediatePoints[i][1]
    #         point[2] += self.bernstein(m - 1, i, v) * intermediatePoints[i][2]
    #     return point

    def display(self):
        """Display the control points as dots."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        aspect = float(self.width) / float(self.height)
        if self.ortho:
            glOrtho(-2.0 * self.ortho_zoom , 2.0 * self.ortho_zoom , -2.0 * aspect * self.ortho_zoom, 2.0 * aspect * self.ortho_zoom, -20.0, 20.0)


        # Set the camera position and orientation
        gluLookAt(self.translation[0], self.translation[1], self.translation[2],
                self.translation[0], self.translation[1], 0,
                0, 1, 0)

        # Apply the rotation
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)

        for surface in self.surfaces:
            controlPoints = np.array(surface)
        
            glColor3f(*get_color('dot'))

            # 下面部分请在作业2.2中注释掉
            # glMap2f(GL_MAP2_VERTEX_3, 0, 1, 0, 1, controlPoints)
            # glEnable(GL_MAP2_VERTEX_3)
            # glEnable(GL_DEPTH_TEST)

            # glMapGrid2f(20, 0.0, 1.0, 20, 0.0, 1.0) # 生成等间距评估（求值）网格
            # glPointSize(1)
            # glEvalMesh2(GL_POINT, 0, 20, 0, 20) # 与下方自行实现进行对照
            # glEvalMesh2(GL_LINE, 0, 20, 0, 20) # 绘制网格
            # glEvalMesh2(GL_FILL, 0, 20, 0, 20) # 绘制整个曲面，包括对法向量等的插值

            # 2.2
            # glEvalMesh2 中对点插值的部分
            # 实际上 glEvalMesh2 还会对法向量等进行插值,并进行三角化，画出整个面，并进行光线渲染
            glPointSize(1)
            glBegin(GL_POINTS)
            for i in range(21):
                for j in range(21):
                    # glEvalCoord2f(i/20.0, j/20.0) # 需要使用 glMap2f 进行映射
                    x, y, z = self.evalCoord2f(controlPoints, i/20.0, j/20.0)
                    glVertex3f(x, y, z)
            glEnd()
            
            # 画出贝塞尔曲面的控制点
            glPointSize(5)
            glBegin(GL_POINTS)
            for i in range(4):
                glColor3f(*get_color('dot'))
                for j in range(4):
                    glVertex3fv(controlPoints[i][j])
            glEnd()

            # 画出贝塞尔曲线
            for i,row in enumerate(controlPoints):
                glColor3f(*get_color(i))
                glBegin(GL_LINE_STRIP)
                for i in range(31):
                    t = float(i) / 30
                    x, y, z = self.evalCoord1f(row, t)
                    glVertex3f(x, y, z)
                glEnd()
            
            transpose = np.transpose(controlPoints, (1, 0, 2))
            for i,row in enumerate(transpose):
                glColor3f(*get_color(i))
                glBegin(GL_LINE_STRIP)
                for i in range(31):
                    t = float(i) / 30
                    x, y, z = self.evalCoord1f(row, t)
                    glVertex3f(x, y, z)
                glEnd()

        glutSwapBuffers()

def main():
    
    # 1.1
    # 见 md

    # 1.2
    # BezierCurve(ortho=False).run()

    # 1.3
    # circle_sol = 0.551784 # 计算详见 md
    # BezierCurve([
    #     ((1, 0, 0), (1, circle_sol, 0), (circle_sol, 1, 0), (0, 1, 0)), # 四段中心对称的贝塞尔曲线
    #     ((0, 1, 0), (-circle_sol, 1, 0), (-1, circle_sol, 0), (-1, 0, 0)),
    #     ((-1, 0, 0), (-1, -circle_sol, 0), (-circle_sol, -1, 0), (0, -1, 0)),
    #     ((0, -1, 0), (circle_sol, -1, 0), (1, -circle_sol, 0), (1, 0, 0)),
    # ], ortho=False).run()

    # 抗锯齿
    # 见 md

    # 2.1
    # 见 md

    # 2.2
    BezierSurface(ortho=True).run()
 

if __name__ == '__main__':
    main()