
varying vec2 vUvs;
uniform vec2 resolution;
uniform float time;

// Converts a color from linear light gamma to sRGB gamma
vec3 fromLinear(vec3 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB * vec3(12.92);

    return vec3(mix(higher, lower, cutoff));
}

// Converts a color from sRGB gamma to linear light gamma
vec3 toLinear(vec3 sRGB)
{
    bvec3 cutoff = lessThan(sRGB, vec3(0.04045));
    vec3 higher = pow((sRGB + vec3(0.055))/vec3(1.055), vec3(2.4));
    vec3 lower = sRGB / vec3(12.92);

    return vec3(mix(higher, lower, cutoff));
}

float inverseLerp(float v, float minValue, float maxValue) {
    return (v - minValue) / (maxValue - minValue);
}

float remap(float v, float inMin, float inMax, float outMin, float outMax) {
    float t = inverseLerp(v, inMin, inMax);
    return mix(outMin, outMax, t);
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

// Copyright (C) 2011 by Ashima Arts (Simplex noise)
// Copyright (C) 2011-2016 by Stefan Gustavson (Classic noise and others)
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// https://github.com/ashima/webgl-noise/tree/master/src
vec3 mod289(vec3 x)
{
    return x - floor(x / 289.0) * 289.0;
}

vec4 mod289(vec4 x)
{
    return x - floor(x / 289.0) * 289.0;
}

vec4 permute(vec4 x)
{
    return mod289((x * 34.0 + 1.0) * x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159 - r * 0.85373472095314;
}

vec4 snoise(vec3 v)
{
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    vec3 i  = floor(v + dot(v, vec3(C.y)));
    vec3 x0 = v   - i + dot(i, vec3(C.x));

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.x;
    vec3 x2 = x0 - i2 + C.y;
    vec3 x3 = x0 - 0.5;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec4 p =
      permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
                            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    vec4 j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    vec4 x_ = floor(j / 7.0);
    vec4 y_ = floor(j - 7.0 * x_); 

    vec4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    vec4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 g0 = vec3(a0.xy, h.x);
    vec3 g1 = vec3(a0.zw, h.y);
    vec3 g2 = vec3(a1.xy, h.z);
    vec3 g3 = vec3(a1.zw, h.w);

    // Normalize gradients
    vec4 norm = taylorInvSqrt(vec4(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 *= norm.x;
    g1 *= norm.y;
    g2 *= norm.z;
    g3 *= norm.w;

    // Compute noise and gradient at P
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    vec4 m2 = m * m;
    vec4 m3 = m2 * m;
    vec4 m4 = m2 * m2;
    vec3 grad =
      -6.0 * m3.x * x0 * dot(x0, g0) + m4.x * g0 +
      -6.0 * m3.y * x1 * dot(x1, g1) + m4.y * g1 +
      -6.0 * m3.z * x2 * dot(x2, g2) + m4.z * g2 +
      -6.0 * m3.w * x3 * dot(x3, g3) + m4.w * g3;
    vec4 px = vec4(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return 42.0 * vec4(grad, dot(m4, px));
}

// The MIT License
// Copyright © 2013 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org/
//
// https://www.shadertoy.com/view/Xsl3Dl
vec3 hash3( vec3 p ) // replace this by something better
{
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
            dot(p,vec3(269.5,183.3,246.1)),
            dot(p,vec3(113.5,271.9,124.6)));

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec3 p )
{
  vec3 i = floor( p );
  vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

  return mix( mix( mix( dot( hash3( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                        dot( hash3( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                   mix( dot( hash3( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                        dot( hash3( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
              mix( mix( dot( hash3( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                        dot( hash3( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                   mix( dot( hash3( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                        dot( hash3( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

float fbm(vec3 p, int octaves, float persistence, float lacunarity, float exponentiation) {
  float amplitude = 0.5;
  float frequency = 1.0;
  float total = 0.0;
  float normalization = 0.0;

  for (int i = 0; i < octaves; ++i) {
    float noiseValue = snoise(p * frequency).w;
    total += noiseValue * amplitude;
    normalization += amplitude;
    amplitude *= persistence;
    frequency *= lacunarity;
  }

  total /= normalization;
  total = total * 0.5 + 0.5;
  total = pow(total, exponentiation);

  return total;
}

vec3 GenerateGridStars(
    vec2 pixelCoords, float starRadius, float cellWidth, 
    float seed, bool twinkle) {

    vec2 cellCoords = (fract(pixelCoords / cellWidth) - 0.5) * cellWidth;
    vec2 cellID = floor(pixelCoords / cellWidth) + seed / 100.0;
    vec3 cellHashValue = hash3(vec3(cellID, 0.0));

    float starBrightness = saturate(cellHashValue.z);
    vec2 starPosition = vec2(0.0);
    starPosition += cellHashValue.xy * (cellWidth * 0.5 - starRadius * 4.0);
    float distToStar = length(cellCoords - starPosition);
    float glow = exp(-2.0 * distToStar / starRadius);

    return vec3(glow * starBrightness);
}

vec3 GenerateStars(vec2 pixelCoords) {
    vec3 stars = vec3(0.0);

    float size = 4.0;
    float cellWidth = 500.0;
    for (float i = 0.0; i <= 2.0; ++i) {
        stars += GenerateGridStars(pixelCoords, size, cellWidth, i, true);
        size *= 0.5;
        cellWidth *= 0.35;
    }

    for (float i = 3.0; i <= 5.0; ++i) {
        stars += GenerateGridStars(pixelCoords, size, cellWidth, i, false);
        size *= 0.5;
        cellWidth *= 0.35;
    }

    return stars;
}

float sdfCircle(vec2 p, float r) {
    return length(p) - r;
}

float map(vec3 pos) {
    return fbm(pos, 6, 0.5, 2.0, 4.0);
}

vec3 calcNormal(vec3 pos, vec3 n) {
    vec2 e = vec2(0.0001, 0.0);
    return normalize(
        n + -500.0 * vec3(
            map(pos + e.xyy) - map(pos - e.xyy),
            map(pos + e.yxy) - map(pos - e.yxy),
            map(pos + e.yyx) - map(pos - e.yyx)
        )
    );
}

mat3 rotateMat(float radians, int axis) {
    float s = sin(radians);
    float c = cos(radians);
    switch (axis) {
        case 0:
            return mat3(
                1.0, 0.0, 0.0,
                0.0, c, s,
                0.0, -s, c
            );
        case 1:
            return mat3(
                c, 0.0, s,
                0.0, 1.0, 0.0,
                -s, 0.0, c
            );
        case 2:
            return mat3(
                c, s, 0.0,
                -s, c, 0.0,
                0.0, 0.0, 1.0
            );
        default:
            break;
    }

    return mat3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
}

mat3 rotateX(float radians) {
    return rotateMat(radians, 0);
}

mat3 rotateY(float radians) {
    return rotateMat(radians, 1);
}

mat3 rotateZ(float radians) {
    return rotateMat(radians, 2);
}

// vec3 decimalRGB(vec3 rgb) {
//     float x = remap(rgb.x, 0.0, 255.0, 0.0, 1.0);
//     float y = remap(rgb.y, 0.0, 255.0, 0.0, 1.0);
//     float z = remap(rgb.z, 0.0, 255.0, 0.0, 1.0);  
//     return vec3(x, y, z);
// }

vec3 DARK_GOLD  = vec3(201.0, 144.0, 65.0 ) / 255.0;
vec3 DARK_DIRT  = vec3(170.0, 75.0, 0.0) / 255.0;
vec3 DETAILS    = vec3(216.0, 202.0, 157.0) / 255.0;
vec3 LIGHT_DIRT = vec3(227.0, 220.0, 203.0) / 255.0;
vec3 ICE        = vec3(235.0, 243.0, 246.0) / 255.0;

vec3 DrawPlanet(vec2 pixelCoords, vec3 colour) {
    float d = sdfCircle(pixelCoords, 400.0);

    vec3 planetColour = vec3(0.0);

    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5));

    if (d <= 0.0) {
        float x = pixelCoords.x / 400.0;
        float y = pixelCoords.y / 400.0;

        float z = sqrt(1.0 - x * x - y * y);

        vec3 viewNormal = vec3(x, y, z);

        // Define planet rotation, position relative to rotation, and
        // view direction wrt rotation.
        mat3 planetRotation = rotateY(-time * 0.1);
        vec3 wsPosition = planetRotation * viewNormal;
        vec3 wsNormal = planetRotation * normalize(wsPosition);
        vec3 wsViewDir = planetRotation * vec3(0.0, 0.0, 1.0);

        // Noise parameters for building terrain
        vec3 noiseCoord = wsPosition * 8.0;
        float noiseSample1 = fbm(noiseCoord, 8, 0.5, 2.0, 4.0);
        float noiseSample2 = fbm(noiseCoord, 4, 16.0, 4.0, 8.0);

        // Colouring
        planetColour = toLinear(DARK_GOLD);
        vec3 lighterColour = mix(
            toLinear(LIGHT_DIRT),
            planetColour,
            smoothstep(0.4, 0.93, abs(viewNormal.y))
        );

        planetColour = mix(planetColour, lighterColour, vUvs.y);

        if (abs(viewNormal.y) > 0.0 && abs(viewNormal.y) < 1.0){
            float smoothingNoise = noiseSample1 / 10.0;
            float smoothingFactorOut = -0.2 + abs(viewNormal.y) + smoothingNoise;
            float smoothingFactorIn = 0.52 - abs(viewNormal.y) + smoothingNoise;

            float smoothFactor = (abs(viewNormal.y) < 0.37) ? smoothingFactorIn : smoothingFactorOut;
            planetColour = mix(
                toLinear(DARK_DIRT), 
                planetColour, 
                smoothstep(0.2, 0.5, smoothFactor)
            );   

            float detailing = noise(viewNormal * noiseSample2);
            planetColour = mix(
                planetColour,
                toLinear(DETAILS),
                smoothstep(0.1, 0.15, detailing)
            );
        }

        // Lighting
        vec2 specParams = mix(
            vec2(0.5, 32.0),
            vec2(0.01, 2.0),
            smoothstep(0.05, 0.06, noiseSample1));
        vec3 wsLightDir = planetRotation * lightDir;
        vec3 wsSurfaceNormal = calcNormal(noiseCoord, wsNormal);

        float wrap = 0.05;
        float dp = max(
            0.0, (dot(wsLightDir, wsSurfaceNormal) + wrap) / (1.0 + wrap));
        
        vec3 lightColour = mix(
            vec3(0.25, 0.0, 0.0),
            vec3(0.75),
            smoothstep(0.05, 0.5, dp));
        vec3 ambient = vec3(0.005);
        vec3 diffuse = lightColour * dp;

        vec3 r = normalize(reflect(-wsLightDir, wsSurfaceNormal));
        float phongValue = max(0.0, dot(wsViewDir, r));
        phongValue = pow(phongValue, specParams.y);

        vec3 specular = vec3(phongValue) * specParams.x * diffuse;

        vec3 planetShading = planetColour * (diffuse + ambient) + specular;
        planetColour = planetShading;

        // Fresnel effect
        float fresnel = smoothstep(1.0, 0.1, viewNormal.z);
        fresnel = pow(fresnel, 8.0) * dp;
        planetColour = mix(planetColour, vec3(0.33, 0.12, 0.05), fresnel);
    }

    colour = mix(colour, planetColour, smoothstep(0.0, -1.0, d));

    if (d < 40.0 && d >= -1.0) {
        // Produce dark glow around planet
        // Repeat with slightly larger circle
        float x = pixelCoords.x / 440.0;
        float y = pixelCoords.y / 440.0;
        float z = sqrt(1.0 - x * x - y * y);
        vec3 normal = vec3(x, y, z);

        float lighting = dot(normal, lightDir);
        lighting = smoothstep(0.15, 1.0, lighting);

        vec3 glowColour = vec3(0.05, 0.05, 0.1) *
            exp(-0.01 * d * d) * lighting * 0.75;
        colour += glowColour;
    }

    return colour;
}

void main() {
    vec2 pixelCoords = (vUvs - 0.5) * resolution;

    vec3 colour = vec3(0.0);
    colour = GenerateStars(pixelCoords);
    colour = DrawPlanet(pixelCoords, colour);

    gl_FragColor = vec4(fromLinear(colour), 1.0);
}
