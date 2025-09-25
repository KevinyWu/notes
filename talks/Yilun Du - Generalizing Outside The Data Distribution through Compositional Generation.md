# Yilun Du - Generalizing Outside The Data Distribution through Compositional Generation

[[talks/README#[2024-08-27] Yilun Du - Generalizing Outside The Data Distribution through Compositional Generation|README]]

[Recording](https://youtu.be/7jvWDF6ZXPc?feature=shared)

## Learning Probabilistic Generative World Models

- Observation -> neural network -> output action
	- When environment changes, the action is no longer valid
	- **Solution: encode preferences over each action**
		- Construct an energy landscape over actions using a probabilistic generative model
		- Low energy = high likelihood = good action
		- Scalar energy represents the negative utility of each action
		- Find action to minimize energy!
		- Can compose energy functions that sum together
- Compositionality
	- Generative AI works very well for language but bad for other settings
		- For language, training distribution is similar to real world distribution
			- Language is naturally compositional!
		- For other modalities, the real world distribution is much larger
	- **Energy Based Models (EBMs) provide a probabilistic manner to represent the real distribution as a composition of factors!**
		- Can represent distribution on many factors
		- Factors are combined to represent entire distribution
		- Factors make independence assumptions that are biased, but allow for more generalization
	- Can we gather data to fill in the gaps?
		- **No, generative models cannot fit arbitrarily high-dimensional distributions but rather ones that are simple**
			- Sampling from unconditional distributions generate poor images
		- **Compositional generation is a tractable way to represent high-dimensional distributions**

## Energy Based Models

- EBM: neural network that takes in an input and outputs scalar energy
	- Function parameterizes entire energy landscape
	- Ex. Red truck energy function
		- Green truck -> high energy
		- Red truck -> low energy
	- Energy function can encode utility, membership function, unnormalized PDF
- Sampling from PDF
	- Naively finding low-energy data point in high-dimensional space is difficult
	- Use Langevin Dynamics to draw samples
		- Start with a sample and follow gradient + Gaussian noise![[energy_based_model.png]]
		- Tradeoff: generation speed <-> ability to generalize and generate from new energy landscapes at prediction time
- Training energy functions
	- Given samples from a distribution $p(x)$, want to learn an energy function $E_{\theta}(s)$ that represents $\log p(s)$, so that $p_{\theta}(x) \propto e^{-E_{\theta}(x)}$
	- Contrastive loss: decrease energy for real data and increase energy of all samples from the probability distribution![[ebm_training.png]]
- Diffusion models are energy based models!
	- Score Based Models (SBMs) are EBMs where $\nabla_{x}E(x_t)$ in Langevin Dynamics is learned by a network $\epsilon_{\theta}(x_t)$ by denoising

## Compositionality

- Normally for neural networks, we can sum loss functions to train on multiple objectives of different properties (ex. goal + gait + obstacle)
	- **But, at test time, we want to generate predictions that satisfy a set of properties that we don't know at training time!**
	- Adding predictions cannot combine properties at prediction time
- With EBM landscapes, we can sum predictions at test time!
	- Composing energy functions has different interpretations depending on what the energy function represents
		- Utility: utility addition
		- Set membership
			- Can use set intersection, set union, set difference
			- Can compose these!
		- Probability distribution
			- Product: draw sample from joint distribution
			- Mixture: draw sample from sum of distributions
			- Inversion: draw sample from inversion of distributions
	- Applications ![[ebm_applications.png]]
		- Generating compositional scenes with multiple requirements
		- Adapting image styles
		- Composing models for different locations of an image
		- If each model is a constraint, you can solve problems with many constraints, i.e. for robotic manipulation
- More composition = harder optimization
- How do we discover independent factors of a distribution
	- Train generative model with a factorized structure
	- Ex. learning the distribution of the reconstruction of an image![[ebm_factor_discovery.png]]
	- Resulting factors each encode a specific part of the image
		- In the example above, each function may encode a different shape

## Planning

- Solving long-horizon robotics tasks!
	- Its very hard to get demonstrations of cutting tomatoes in every single house!
	- **Decompose demonstrations into two factors: goal and dynamics**
- Classical robot planning
	- State and action input
	- Optimize a set of actions with respect to rewards
	- Very difficult to do in practice because of **adversarial states**
		- Unseen states can get high reward
- With energy functions, we can capture the dynamics of the trajectory probabilistically in an energy landscape
	- Only if every single state-action pair has consistent dynamics is the energy low
- **Compose trajectory energy functions with cost functions (goal, value function, test-time constraints, etc) to plan!**
	- Reinforcement learning: solving many tasks with one model! ![[ebm_rl.png]]
		- Compositional approach works as well as SOTA offline RL techniques
	- Goal seeking policy with hand-crafted start-goal energy function ![[ebm_goal.png]]
	- Visual observations: diffusion policy
		- Energy function over next sequence of actions
	- Video generation
		- Video trajectory energy function
		- Use images as a unified abstraction of trajectories states and actions across environments
		- Can learn to plan from videos on the internet!
		- Synthesized video plans are highly realistic and detailed

## Multimodal Models

- **Composing pretrained models: every pretrained model encodes an energy function on each data point via its loss!**
	- LLM
		- Input: language
		- Energy function: likelihood loss
	- CLIP
		- Input: (language, image)
		- Energy function: language/vision alignment loss
	- Ex. Given image of a cat, question: what is the cat doing?![[ebm_vqa.png]]
		- LLM and CLIP cannot solve problem on its own
		- Use CLIP energy function to construct landscape of possible language descriptions given the image
		- Use LLM energy function to construct landscape of possible responses given the question
		- Optimize over summed landscape to get the answer!
		- **Zero-shot method to combine information at test time**
- Consider long-horizon task: making a cup of tea
	- Need semantic knowledge, visual knowledge, physical knowledge, control knowledge
	- Combine:
		- Task information from LLM
		- Motion information from video model
		- Kinematic information from egocentric action model
	- We can solve the task if we minimize over all these energy landscapes
	- Challenge: video model is hard to optimize
		- Instead, use a VLM instead of LLM to propose visually grounded plans
	- Combining foundation models ![[ebm_video_planning.png]]
