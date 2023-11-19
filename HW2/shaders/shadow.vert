#version 410
    layout(location = 0) in vec4 pos;

    uniform mat4 light_MVP; //模型视图投影矩阵

    void main() {
        gl_Position = light_MVP * pos;
    }