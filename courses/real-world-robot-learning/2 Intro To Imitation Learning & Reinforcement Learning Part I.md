# Intro To Imitation Learning & Reinforcement Learning Part I

#imitation-learning
#reinforcement-learning

[[courses/real-world-robot-learning/README#[2] Intro To Imitation Learning & Reinforcement Learning Part I|README]]

## Perception-Action Loops

- Environment -> observation -> representation -> action
- What robots can learn from data
	- Policies: mapping from perceived state to action
	- Dynamics models: models of how agent actions influence the evolution of the environment state
	- Reward functions: a score indicating how well the robot is performing a task
	- State representations: an encoding of raw sensory inputs
	- "Common-sense knowledge"

## Markov Decision Process

- MDP $(S,A,P,R,\gamma)$
	- Transition function (dynamics model) $P(s'|s,a)$
	- Reward function $r_{t}= R(s,a,s')$
	- Discount factor $\gamma<1$
	- Utility (discounted future reward) $\sum\limits_{t}y^tr_{t+1}$
	- Goal: maximize **expected** utility
- Traditional model-based planning
	- Sample actions and forecast their effects using $P(s'|s,a)$
	- Select action with best forecasted outcome $\sum\limits_{t}y^tr_{t+1}$
- Traditional model-based controllers
	- Rely on near-accurate models of environment
	- Ex. LQR, MPC, H-infinity control, policy iteration
- In practice, we cannot assume knowledge of $P, R$
	- Reinforcement learning
		- Exploration to experience effects of $P, R$
		- Exploration: how to acquire useful experience
		- Credit assignment: how to figure out what experiences are good and bad
	- Imitation learning
		- Teacher shows the way
		- Only need to optimize policy, not gather experience
- Partially observed MDP (POMDP)
	- cannot observe full Markov state

## Imitation Learning Through Behavioral Cloning

- Policy: mapping from input states to probability distribution over output action (or sometimes just deterministic action)
- Behavioral cloning objectives
	- **Supervised maximum likelihood objective to map from states to expert actions**
	- Loss: $-\frac{1}{N}\sum\limits_{i=1}^{N}\left(\sum\limits_{t=1}^{T}\log \pi_{\theta}(a_{i,t}|s_{i,t})\right)$
		- $a_{i,t}$ is expert action, so want the log probability of expert actions given states to be high (therefore making loss low)
		- Gradient: $\frac{1}{N}\sum\limits_{i=1}^{N}\left(\sum\limits_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a_{i,t}|s_{i,t})\right)$
- No connection to reward function
	- If $R$ is bounded, then a policy with error rate $\epsilon$ on expert data incurs a reward penalty $< \mathcal{O}(T^2\epsilon)$
- Violates IID assumption of supervised learning: predictions affect future observations during execution to learned policy
	- Leads to compounding error
	- Need recovery data: DAGGER (interactive online BC)
		- Train policy from expert data
		- Rollout policy to get new dataset of states
		- Ask expert to label each new state with new actions
			- Assumption that expert is always available
		- Aggregate the dataset
- Other policies as experts
	- Instead of human, policy trained on privileged observation, hand scripted policy, or model-based policy operating on system state
	- E.g. train expert on state, then train vision based policy to imitate expert
