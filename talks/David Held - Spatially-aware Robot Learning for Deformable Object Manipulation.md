# David Held - Spatially-aware Robot Learning for Deformable Object Manipulation

[[talks/README#[2024-06-14] David Held - Spatially-aware Robot Learning for Deformable Object Manipulation|README]]

[Recording](https://youtu.be/OAVlWupYjxM?feature=shared)

## Mesh-based Dynamics with Occlusion Reasoning for Cloth Manipulation (MEDOR)

- Question: how to represent state of an object to be useful for planning?
	- 6-DOF pose for rigid object
	- What about cloth?
		- Need to reason about occlusions (folds)
		- How to reconstruct a crumpled cloth from arbitrarily crumpled configurations?
- **Self-supervised test-time finetuning**
	- Align prediction with observation: unidirectional Chamfer distance, used to measure the similarity between two point sets
	- Mapping consistency loss: map observation to canonical space, use shape completion to obtain a mesh, then map canonical back to observation
		- If you track a point, it should be mapped back to where it started
	- Both of these losses are self-supervised and **can be used at test time**
	- Test-time finetuning shows significant improvement over initial prediction
- Evaluation on cloth flattening: maximize coverage area of the cloth
	- Learn a dynamics model to plan over robot actions
	- Why does occlusion reasoning help (as opposed to just using the visible mesh)
		- Full mesh gives a more accurate reward signal for model-based RL
	- Remaining gaps in performance: using ground truth mesh leads to 23% improvement

## Self-supervised Cloth Reconstruction via Action-conditioned Cloth Tracking

- Prior works train reconstruction model in simulation
- Lack of real-world mesh data: how to collect mesh-data in self-supervised manner?
- Train on unlabeled real-world videos ![[cloth_reconstruction.png]]
	- Finetuning is same as MEDOR
- Collecting data: manipulate cloth using tweezers and track endpoints of tweezers to track actions
- **Action-conditioned cloth tracking** ![[action_conditioned_cloth_tracking.png]]
	- Dynamics model that takes in mesh and action and predicts next mesh
		- For T times: use Chamfer distance to compare prediction and real world
	- However, errors will accumulate over time between real-world and prediction
		- Test-time optimization to correct for error accumulation
			- Unidirectional Chamfer loss
			- Neighborhood Consistency loss (neighboring nodes on the cloth should move together)
- Real-world finetuning improves on MEDOR
