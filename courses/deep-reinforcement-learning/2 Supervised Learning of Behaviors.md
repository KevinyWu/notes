# Supervised Learning of Behaviors

[[courses/deep-reinforcement-learning/README#[2] Supervised Learning of Behaviors|README]]

Lecture [2.1](https://youtu.be/tbLaFtYpWWU?feature=shared), [2.2](https://youtu.be/YivJ9KDjn-o?feature=shared), [2.3](https://youtu.be/ppN5ORNrMos?feature=shared)

## Notation

  - Notation ![[notation.png]]
	- State $s_t$ is different from observation $o_t$
		- State is a complete and concise representation of state of the world (**fully observed**)
		- Observation is what the agent sees (**partially observed**)
		- State can sometimes be inferred from observation
- **Markov assumption**: $s_t$ contains all relevant information from the past (don't need $s_{t-1}, s_{t-2}, \ldots$ to predict $s_{t+1}$)

## Imitation learning

- Learn policies using supervised learning
- **Behavioral cloning**: learn a policy that mimics an expert's behavior
	- Collect data from expert
	- Train a policy to predict the expert's actions
	- Problems
		- Distribution mismatch between training and test data
		- **Violates i.i.d. assumption: small errors lead to larger and larger errors over time**
		- Let the cost function $c(s_t, a_t) = \begin{cases} 0 & \text{if } a_t = \pi^*(s_t) \\ 1 & \text{otherwise} \end{cases}$
		- Assume $\pi_{\theta}(a \neq \pi^*(s)|s) \leq \epsilon$ for all states $s\in \mathcal{D}_{\text{train}}$
		- Then the sum of expected errors across timesteps is $E\left [\sum_t c(s_t, a_t) \right ]$ is $\mathcal{O}(\epsilon T^2)$
		- **This sum grows quadratically with the number of timesteps!**
	- Paradox: more errors (and recoveries) in training data is beneficial
		- Data augmentation: add "fake" data that illustrates corrections
- Failing to fit the expert
	- Non-markovian behavior: action depends on all past observations
		- I.e. $\pi_{\theta}(a_t | o_1, o_2, \ldots, o_t)$
		- Use the whole history of observations with a sequence model (transformer, LSTM, etc.)
		- Problem: "causal confusion"
	- Multimodal behavior: expert has multiple ways to solve a problem
		- Solution 1: expressive continuous distributions
			- Gaussian mixture models ![[gmm.png]]
		- Latent variable models (ex. CVAE)
			- Predict a latent variable $z$ that is used to choose the distribution of actions
		- Diffusion models ![[diffusion.png]]
- Goal-conditioned behavioral cloning
	- $\pi_{\theta}(a_t | s_t, g)$ where $g$ is the goal
- Changing the algorithm (DAgger)
	- Idea: make $p_{\text{data}}(o_t) = p_{\pi_{\theta}}(o_t)$
	- **DAgger: Dataset Aggregation**
		- Goal: collect training data from $p_{\pi_{\theta}}(o_t)$ instead of $p_{\text{data}}(o_t)$
			1. Train $\pi_{\theta}$ on human data $\mathcal{D} = \{o_1, a_1, \dots, o_N, a_N\}$
			2. Run $\pi_{\theta}(a_t|o_t)$ to get dataset $\mathcal{D}_{\pi} = \{o_1 \dots, o_M\}$
			3. Ask human to label $D_{\pi}$ with actions $a_t$
			4. Aggregate datasets: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_{\pi}$
			5. Repeat
		- Step c. is the problem
- Big problem with imitation learning
	- Humans need to provide data, which is finite
	- Humans bad at providing some kinds of actions
	- Humans learn autonomously, can machines do the same?
