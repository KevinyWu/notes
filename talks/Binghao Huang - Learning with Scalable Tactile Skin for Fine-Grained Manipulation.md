# Binghao Huang - Learning with Scalable Tactile Skin for Fine-Grained Manipulation

[[talks/README#[2025-11-24] Binghao Huang - Learning with Scalable Tactile Skin for Fine-Grained Manipulation|README]]

## FlexiTac

- Hardware design
	- Need to consider: fabrication time, cable management, reproducibility
	- No baseline drift (after overnight test)
- Optical sensors vs FlexiTac
	- Optical sensors (GelSim, GelSight) can provide high precision
	- FlexiTac scales well for everyday robots
- No shear

## Scaling Up Tactile Data

- 3D-ViTac
	- Using 4D point cloud representation
		- 3D points for visual observation (in robot base frame)
			- 512 points
			- 4th dimension is 0 for visual point cloud
		- 4D points for tactile observation (3D points in robot base frame + 1D continuous tactile reading)
			- 256 points per finger
- Tactile on UMI Gripper
	- Using RGB + Masked tactile
	- Encode + cross-attend RGB and tactile
- [VT-Refine](https://binghao-huang.github.io/vt_refine/)
	- Tactile simulation
		- Using point cloud representation
		- Normal force is easier to simulate
	- Visuo-tactile vs. vision only experiment
		- 1536 tactile points, 1024 vision points
		- The vision-only does not use the tactile points at all (input dim 1024 points)
		- Tactile points (even without 4th tactile dimension) may still help the policy because they are computed with forward kinematics
