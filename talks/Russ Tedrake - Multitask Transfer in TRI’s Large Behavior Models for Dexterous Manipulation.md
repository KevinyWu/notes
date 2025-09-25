# Russ Tedrake - Multitask Transfer in TRI's Large Behavior Models for Dexterous Manipulation

[[talks/README#[2025-04-25] Russ Tedrake - Multitask Transfer in TRI's Large Behavior Models for Dexterous Manipulation|README]]

[Recording](https://www.youtube.com/watch?v=TN1M6vg4CsQ)

## Evidence that Pretraining is Better than Single-Task

- LBM pretraining reduces the amount of task-specific data required
- Given the same amount of task-specific data, finetuned specialists derived from pretrained LBMs outperform single-task models when aggregating over tasks
- Pretrained LBMs demonstrate increased robustness in scenarios diverging from their training conditions

## Architecture

- LBM architectures
	- **"Multitask Diffusion Policy": scaled up DP + language conditioning**
		- Takes in RGB, language, and robot state
		- CLIP-ViT for RGB
		- CLIP for language encoding
		- DiT head
		- Some architecture changes help
			- CLIP >> ResNet
			- DiT >> UNet
			- [Relative end-effector pose (UMI PD2)](https://arxiv.org/pdf/2402.10329): all actions relative to EE pose at $t_0$
			- Many other architecture changes end up mostly in noise
			- **Data normalization is much more important than diffusion vs. flow matching**
				- Normalize action at each timestep independently in the dataset (instead of using the same normalizer for all actions)
	- VLA models: VLM pretraining for the base model
	- "Video-language-action" models
		- Video-prediction pretraining for the base model

## Evaluations

- Simulation evaluations
	- Repeatable
	- More rollouts for more statistics
	- Make the simulator really good to have a strong correspondence with real
- Need to evaluate on hard tasks with low success rates to distinguish
- Need lots of rollouts to get statistical significance
- Seen tasks
	- Similar performance
- Unseen tasks (testing dexterity, not generalization)
	- LBM pretraining + new demos is better than just training on demos
	- Pretraining allows you to do **much better** with **less** demonstrations
- Seen tasks with distribution shift (testing robustness)
	- New objects, sizes, initial conditions, lighting
	- LBM does can get similar performance as single task with only 15% of the data

## Other Takeaways

- Don't trust val loss for checkpoint selection
- Data quality matters, show failure + recovery
- Pretraining results in more recovery behaviors (based on qualitative analysis)
	- LBM doesn't always complete the task, but it always looks like manipulation
- Role of pretraining: get 5-10% success rate on any task
