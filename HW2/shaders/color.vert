#version 410
 
    layout(location = 0) in vec4 pos; //原始顶点坐标
    layout(location = 1) in vec2 texcoord_ori; //原始纹理坐标
    layout(location = 2) in vec3 surf_norm; //原始面法向量
    layout(location = 3) in vec3 vert_norm; //原始顶点法向量
    layout(location = 4) in vec3 vert_tangent; //原始顶点切向量

    uniform mat4 MVP; //模型视图投影矩阵
    uniform mat4 M;   //法线变换矩阵
    uniform mat4 light_MVP; //光源视图投影矩阵，用于阴影

    uniform vec3 light_pos; //光源在世界坐标下的位置
    uniform vec3 light_color; //光源的颜色

    uniform vec3 view_pos; //视点在世界坐标下的位置

    uniform int shading_freq; //着色频率

    uniform float ambient_strength; //环境光强度
    uniform float specular_strength; //高光强度
    uniform float specular_power; //高光系数

    out vec2 texcoord; //纹理坐标
    out vec3 tangent_frag_pos; //片段位置在切线空间下的坐标

    flat out vec3 ambient; //环境光

    flat out vec3 flat_diffuse; //漫反射（平面着色）
    flat out vec3 flat_specular; //高光（平面着色）

    out vec3 diffuse; //漫反射（Gouraud着色）
    out vec3 specular; //高光（Gouraud着色）

    out vec3 tangent_light_pos; //光源在切线空间下的位置
    out vec3 tangent_view_pos; //视点在切线空间下的位置

    out vec4 light_frag_pos; //光源视角片段位置，用于阴影

    void main() {
        texcoord = texcoord_ori;
        gl_Position = MVP * pos;
        light_frag_pos = vec4(light_MVP * pos);

        vec3 frag_pos = vec3(M * pos);

        //通用环境光
        ambient = ambient_strength * light_color; //环境光

        if (shading_freq == 0) {

            // 无光照着色

        } else if (shading_freq == 1 || shading_freq == 2) { // Flat&Gouraud
            
            // 通用量
            vec3 light_vec = light_pos - frag_pos; //片段指向光源的向量
            vec3 light_dir = normalize(light_vec); //光源方向
            float r2 = dot(light_vec, light_vec); //距离平方
            vec3 view_dir = normalize(view_pos - frag_pos); //视线方向
            vec3 halfway_dir = normalize(light_dir + view_dir); //半向量

            if (shading_freq == 1) { // Flat

                vec3 surf_norm = normalize(mat3(M) * surf_norm); //面法向量
                
                //漫反射
                flat_diffuse = light_color * max(dot(surf_norm, light_dir), 0.0) / r2; //漫反射

                //高光
                float spec = pow(max(dot(surf_norm, halfway_dir), 0.0), specular_power); //高光系数
                flat_specular = specular_strength * spec * light_color / r2; //高光

            } else if (shading_freq == 2) { // Gouraud
            
                vec3 vert_norm = normalize(mat3(M) * vert_norm); //顶点法向量
                
                //漫反射
                diffuse = light_color * max(dot(vert_norm, light_dir), 0.0) / r2; //漫反射

                //高光
                float spec = pow(max(dot(vert_norm, halfway_dir), 0.0), specular_power); //高光系数
                specular = specular_strength * spec * light_color / r2; //高光

            }
        }
        else if (shading_freq == 3) {
        
            vec3 frag_norm = normalize(mat3(M) * vert_norm); //顶点法向量

            //TNB
            vec3 T = normalize(mat3(M) * vert_tangent);
            vec3 B = normalize(cross(frag_norm, T));
            mat3 TBN = transpose(mat3(T, B, frag_norm));

            tangent_frag_pos = TBN * frag_pos;
            tangent_light_pos = TBN * light_pos;
            tangent_view_pos = TBN * view_pos;
        
        }
    }