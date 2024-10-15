# 3D Representation

## [2024-05] Tactile-Augmented Radiance Fields

#3d-scenes
#tactile
[[Tactile-Augmented Radiance Fields]]
- First dataset to capture quasi-dense, scene-level, and spatially-aligned visual-tactile data
- Simultaneously collect touch signal on the mounted **vision-based tactile sensor** sensor, then find relative 6-DOF $(R, t)$ pose between camera and touch sensor with a braille board
- Use a diffusion model to estimate the touch signal (represented as an image from a vision-based touch sensor) for other locations within the scene
