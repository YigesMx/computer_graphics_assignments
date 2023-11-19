#version 410
    in vec3 frag_pos;

    uniform sampler2D texture;

    void main(){
        float lon = atan(frag_pos.z, frag_pos.x);
        float lat = atan(-frag_pos.y, length(frag_pos.xz));
        vec2 texcoord = vec2(-lon, lat) / vec2(2 * 3.1415926535, 3.1415926535) + vec2(0, 0.5);

        gl_FragColor = texture(texture, texcoord);
    }