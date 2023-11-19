#version 410
    in vec2 texcoord;  //片段纹理坐标（插值）
    in vec3 tangent_frag_pos; //片段位置在切线空间下的坐标

    flat in vec3 ambient; //环境光

    flat in vec3 flat_diffuse; //漫反射（平面着色）
    flat in vec3 flat_specular; //高光（平面着色）

    in vec3 diffuse; //漫反射（Gouraud着色）
    in vec3 specular; //高光（Gouraud着色）

    in vec3 tangent_light_pos; //光源在切线空间下的位置
    in vec3 tangent_view_pos; //视点在切线空间下的位置
    
    in vec4 light_frag_pos; //光源视角片段位置，用于阴影
    
    uniform vec3 light_color; //光源的颜色
    
    uniform sampler2D texture_map; //纹理贴图
    uniform sampler2D normal_map; //法线贴图
    uniform sampler2D bump_map; //高度贴图

    uniform int shading_freq; //着色频率
    uniform float normal_switch; //法线修改贴图开关
    uniform float texture_switch; //纹理贴图开关
    uniform int shadow_switch; //阴影开关

    uniform float specular_strength; //高光强度
    uniform float specular_power; //高光系数

    uniform float bump_strength; //凹凸贴图强度

    uniform sampler2D shadow_map; //阴影贴图

    float calc_shadow(vec4 light_frag_pos/*, vec3 light_dir*/){
        // 除以齐次坐标w，得到光源视角下的片段位置
        vec3 light_frag_pos_proj = light_frag_pos.xyz / light_frag_pos.w;
        // 变换到[0,1]区间
        light_frag_pos_proj = light_frag_pos_proj * 0.5 + 0.5;

        // 通过 shadow_map 从光源视角下获取最近的深度值（使用[0,1]范围fragPosLight作为坐标）
        float closestDepth = texture(shadow_map, light_frag_pos_proj.xy).r; 

        // 从光源视角下获取当前片段的深度值（使用[0,1]范围fragPosLight作为坐标）
        float currentDepth = light_frag_pos_proj.z;

        // 比较两个深度值，如果当前片段在阴影中，返回1.0，否则返回0.0
        // 可以考虑与光夹角越小越使用大偏置，但这里为了三种着色频率通用，不进行light_dir的传参，当仅使用 phong 时可以开启
        // float bias = max(0.005 * (1.0 - dot(vec3(0,0,1), light_dir)), 0.0005);
        float bias = 0.002;
        float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

        return shadow;
    }

    void main() {
        // 材质
        vec3 tex;
        if (texture_switch == 0) {
            tex = vec3(1,1,1);
        } else if (texture_switch == 1) {
            tex = texture(texture_map, texcoord).rgb;
        }

        // 阴影
        float shadow = shadow_switch == 0 ? 0 : calc_shadow(light_frag_pos);

        // 着色
        vec3 brightness;
        if (shading_freq == 0) { // 无着色

            brightness = vec3(1,1,1);

        } else if (shading_freq == 1) { // Flat

            brightness = ambient + (1.0-shadow)*(flat_diffuse + flat_specular); 

        } else if (shading_freq == 2) { // Gouraud

            brightness = ambient + (1.0-shadow)*(diffuse + specular);

        } else if (shading_freq == 3) { // Phong

            //片段法向量，切线空间
            vec3 frag_norm = vec3(0,0,1);
            
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

            // 通用量
            vec3 light_vec = tangent_light_pos - tangent_frag_pos;
            vec3 light_dir = normalize(light_vec);
            float r2 = dot(light_vec, light_vec);
            vec3 view_dir = normalize(tangent_view_pos - tangent_frag_pos);
            vec3 halfway_dir = normalize(light_dir + view_dir);
        
            //漫反射
            vec3 diffuse = light_color * max(dot(frag_norm, light_dir), 0.0) / r2;

            //高光
            float spec = pow(max(dot(frag_norm, halfway_dir), 0.0), specular_power);
            vec3 specular = specular_strength * spec * light_color / r2;

            brightness = ambient + (1.0-shadow)*(diffuse + specular);
        } 
        gl_FragColor = vec4(tex*brightness, 1.0);
    }