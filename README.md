# Rotating-Planet
Shader model of a rotating planet using three.js and GLSL. The model includes a basic lighting system and procedurally generated terrain using [Fractal Brownian Motion](https://thebookofshaders.com/13/).

This project is based on SimonDev's course [GLSL Shaders from Scratch](https://simondev.teachable.com/courses/).

For reference, the pixel coordinates have been shifted so that `(0.0, 0.0)` lies at the center of the screen, rather than at the bottom left.



## Noise
The FBM algorithm uses a Simplex Noise function `snoise()`. It outputs a floating point value and takes as inputs: 
- `vec3 p`: position value;
- `int octaves`: number of superimposed waves;
- `float persistance`: amplitude increment factor per iteration;
- `float lacunarity`: frequency increment factor per iteration;
- `float exponentiation`: exponentiation amount for the final total value.

The algorithm is as follows:
```
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
```

The model also includes a hashing algorithm for discrete distributions (such as the star grid background):
```
vec3 hash3(vec3 p) {
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
            dot(p,vec3(269.5,183.3,246.1)),
            dot(p,vec3(113.5,271.9,124.6)));

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}
```
and [Domain Warping](https://iquilezles.org/articles/warp/), based on the FBM algorithm, for color-blending:
```
float domainWarpingFBM(vec3 coords) {
  vec3 offset = vec3(
    fbm(coords, 4, 0.5, 2.0, 4.0),
    fbm(coords + vec3(43.235, 23.112, 0.0), 4, 0.5, 2.0, 4.0), 0.0);
  float noiseSample = fbm(coords + offset, 1, 0.5, 2.0, 4.0);

  vec3 offset2 = vec3(
    fbm(coords + 4.0 * offset + vec3(5.325, 1.421, 3.235), 4, 0.5, 2.0, 4.0),
    fbm(coords + 4.0 * offset + vec3(4.32, 0.532, 6.324), 4, 0.5, 2.0, 4.0), 0.0);
  noiseSample = fbm(coords + 4.0 * offset2, 1, 0.5, 2.0, 4.0);

  return noiseSample;
}
```


## Stars
The stars in the background are generated using the `GenerateStars()` function. The `hash3()` function is used to determine the random position for each star in `GenerateGridStars()`. 




## Planet
The planet generation is fully done in the function `DrawPlanet()`. The following is a breakdown of each part of the creation of the planet model:

### Circular Region
The circular region for the planet is determined using a simple circle Signed Distance Function (SDF), whose result is used to blend the planet color parameter and the star background using `smoothstep()`.

### 3D model
The 3D effect is obtained by mapping the circle of the planet onto a hemisphere of a unit 3-sphere. This is an inverse stereographic projection, taking the $x$ and $y$ coordinates on the screen as the first two coordinates and $z = \sqrt{1 - (x^2 + y^2)}$ as the third, making the vector `vec3(x, y, z)` a unit normal vector to the hemisphere. This normal vector is stored in `viewNormal`

### Rotation
The planet rotation effect is obtained by applying a rotation matrix around the y-axis (`rotateY()`) on the normal vectors on the model's surface, taking a multiple of the `time` uniform as input. The relative vectors on the 3-sphere are obtained by applying the `planetRotation` matrix to `viewNormal`. 

### Colors
The colors of the model are stored in `planetColour`. The darkened poles are obtained by blending the lighter color  ![#E3DCCB](https://placehold.co/15x15/e3dccb/e3dccb.png)#E3DCCB with the darker [#C99041](https://placehold.co/15x15/c99041/c99041.png)#C99041, with respect to the vertical distance from the origin. The darker strips accross the model are a mix of [#aa4b00]()#aa4b00 and [#d8ca9d]()#d8ca9d using `domainWarpingFBM()` and a `smoothingFactor` defined using `fbm()`, and subsequently mixing this with `planetColour` to obtain a smoother, gas-looking transition to the base planet color.

### Lighting
The lighting system simulates a direct light source (`diffuse` lighting), as well as ambient light and a small reflective effect using the [Phong Reflective Model](https://en.wikipedia.org/wiki/Phong_reflection_model). The light source rotates with the planet to showcase the lit and shadowed regions of the planet. 

The terrain texture is also created by the lighting section, using the `calcNormal()` to add noise to `wsNormal`.

The model simulates the Fresnel Effect by adding a light reflectance as you approach the "edges" of the sphere.

The glow around the planet, meant to give the impression of an atmosphere, is created by almost the exact same code as the planet, but with a slightly larger circle. 



## Moon
A moon model is currently being worked on, so that it spins around its own axis and orbits the planet.
