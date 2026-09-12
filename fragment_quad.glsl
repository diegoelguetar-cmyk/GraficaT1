#version 330

uniform sampler2D accum_tex;
uniform float exposure;

in vec2 frag_texcoord;

out vec4 out_color;

void main() {
    vec4 color = texture(accum_tex, frag_texcoord);

    vec3 mapped = vec3(1.0) - exp(-color.rgb * exposure);

    out_color = vec4(mapped, 1.0);
}