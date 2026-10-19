#version 450

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 color;
    vec4 direction;
    vec4 params;
};

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;

layout(set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    ivec4 light_counts;
} scene;

layout(set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular;
} model_uniforms;

layout(location = 0) out vec4 outColor;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 lightColor, vec3 albedo, vec3 specColor, float shininess, float attenuation) {
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), shininess);
    vec3 diffuse = diff * lightColor * albedo;
    vec3 specular = spec * lightColor * specColor;
    return (diffuse + specular) * attenuation;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.camera_position.xyz - fragWorldPos);
    vec3 albedo = model_uniforms.albedo.rgb;
    vec3 specColor = model_uniforms.specular.rgb;
    float shininess = 32.0;

    vec3 color = scene.ambient_light.rgb * albedo;

    vec3 Ld = normalize(-scene.directional_light_direction.xyz);
    color += blinnPhong(N, V, Ld, scene.directional_light_color.rgb, albedo, specColor, shininess, 1.0);

    for (int i = 0; i < scene.light_counts.x; ++i) {
        vec3 toLight = point_lights[i].position.xyz - fragWorldPos;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 0.0001);
        float attenuation = 1.0 / (dist * dist);
        color += blinnPhong(N, V, L, point_lights[i].color.rgb, albedo, specColor, shininess, attenuation);
    }

    for (int i = 0; i < scene.light_counts.y; ++i) {
        vec3 toLight = spot_lights[i].position.xyz - fragWorldPos;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 0.0001);
        vec3 spotDir = normalize(spot_lights[i].direction.xyz);

        float theta = dot(-L, spotDir);
        float innerCos = spot_lights[i].params.x;
        float outerCos = spot_lights[i].params.y;
        float spotFactor = clamp((theta - outerCos) / max(innerCos - outerCos, 0.0001), 0.0, 1.0);

        float attenuation = (1.0 / (dist * dist)) * spotFactor;
        if (spotFactor > 0.0) {
            color += blinnPhong(N, V, L, spot_lights[i].color.rgb, albedo, specColor, shininess, attenuation);
        }
    }

    if (any(isnan(color)) || any(isinf(color))) {
        color = vec3(1.0, 0.0, 1.0);
    }

    outColor = vec4(color, 1.0);
}