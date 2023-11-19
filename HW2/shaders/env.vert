#version 410
    layout(location = 0) in vec4 pos;

    uniform mat4 MVP;
    uniform mat4 M;

    out vec3 frag_pos;

    void main(){
        frag_pos = vec3(M * pos);
        gl_Position = MVP * pos;
    }