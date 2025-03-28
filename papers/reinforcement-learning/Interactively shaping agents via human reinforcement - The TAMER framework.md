# Interactively shaping agents via human reinforcement: The TAMER framework

**Authors**: W. Bradley Knox, Peter Stone

#reinforcement-learning
#rl-human-feedback

[[papers/reinforcement-learning/README#[2009-09] Interactively Shaping Agents via Human Reinforcement The TAMER Framework|README]]

[Paper](https://dl.acm.org/doi/10.1145/1597735.1597738)
[Code](https://github.com/benibienz/TAMER)
[Website](https://www.cs.utexas.edu/~bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html)

## Abstract

> As computational learning agents move into domains that incur real costs (e.g., autonomous driving or financial investment), it will be necessary to learn good policies without numerous high-cost learning trials. One promising approach to reducing sample complexity of learning a task is knowledge transfer from humans to agents. Ideally, methods of transfer should be accessible to anyone with task knowledge, regardless of that person's expertise in programming and AI. This paper focuses on allowing a human trainer to interactively shape an agent's policy via reinforcement signals. Specifically, the paper introduces "Training an Agent Manually via Evaluative Reinforcement," or tamer, a framework that enables such shaping. Differing from previous approaches to interactive shaping, a tamer agent models the human's reinforcement and exploits its model by choosing actions expected to be most highly reinforced. Results from two domains demonstrate that lay users can train tamer agents without defining an environmental reward function (as in an MDP) and indicate that human training within the tamer framework can reduce sample complexity over autonomous learning algorithms.

## Summary

- To deploy RL in the real world, trial-and error is costly, i.e. autonomous driving
	- Humans have domain knowledge to reduce sample complexity
- Shaping problem
	- Agent receives sequence of states and actions
	- Human trainer has predefined performance metric, gives agent positive and negative scalar reinforcement signals
	- How can agent learn best task policy $\pi:S \rightarrow A$
- **This paper uses MDP without environmental reward signal: MDP\R**
- Benefits of learning from human reinforcement
	- Decrease sample complexity
	- Agent can learn in absence of coded evaluation function (i.e. environment reward)
	- Simple mode of communication allows regular people to train policy
	- Shaping enables learning in more complex domains than autonomous learning

## Background

- Learning from advice
	- "Advice" in MDP: suggesting an action when a certain condition is true
	- **"General natural language recognition is unsolved" - closer to solved now?**
		- [Yell At Your Robot: Improving On-the-Fly from Language Corrections](https://yay-robot.github.io/)
		- [Trajectory Improvement and Reward Learning from Comparative Language Feedback](https://liralab.usc.edu/comparative-language-feedback/)
		- [Verifiably Following Complex Robot Instructions with Foundation Models](https://robotlimp.github.io/)
- Learning from demonstration
	- Some tasks are too difficult for human trainer, like flying a helicoptor or multi-agent environments
- Learning from reinforcement (shaping)
	- Clicker training: when trainer only gives positive reinforcement
	- The tamer system is distinct from previous work on human-delivered reinforcement in that it is designed both for a human-agent team and to work in complex domains through function approximation, generalizing to unseen states

## Method

- TAMER framework ![[tamer.png]]
- TAMER seeks to learn a human's reinforcement function $H:S\times A \rightarrow \mathbb{R}$
	- Agent takes action that maximizes $\hat{H}(s,a)$
	- **Optimal policy defined solely by the trainer**
- Motivating insights
	- Challenge upon receiving environment reward: assigning credit from that reward to entire history of past state-action pairs
		- This is difficult when reward is assigned at the end
		- An attentive human trainer can evaluate an action within a small window of its execution: **learning $H$ is a supervised learning problem**
		- $H$ is intuitively a moving target: humans standards change when the agent's policy improves
	- Lower training episodes means learning "true" $H$ will not be reached
		- But ideally we only need to learn $\hat{H}(s,a_{1)}> \hat{H}(s,a_{2)} \Longleftrightarrow H(s,a_{1)} > H(s,a_{1)}$
	- Exploration
		- TAMER is agnostic to exploration, can use any action selection algorithm
		- They greedily choose action expected to receive highest reward
			- How does this lead to good exploration? Some better states may only be reach by going first to an undesirable state
			- **Is myopic approach optimal?**
				- **Myopia puts more burden on the trainer who must micromanage the agents behavior?**
				- **Long-term reward algorithms are easier on the trainer but harder to design?**
	- Comparison to RL
		- RL: agent maximizes discounted sum of future reward
		- TAMER: does NOT seek to maximize discounted sum of future human reinforcement
			- Instead maximizes short-term reinforcement because trainer's reinforcement signal is a direct label on recent state-action pairs
		- $H$ is not an exact replacement for $R$ in an MDP
			- MDP\R: $R$ is removed and $H$ is sole determinant of good and bad behavior
- High-level algorithm ![[tamer_algorithm.png]]
- Credit assignment to a dense state-action history
	- In some task domains, the frequency of time steps is too high for human trainers to respond to specific stateaction pairs before the next one occurs
	- For these faster domains, credit from the human's reinforcement must be appropriately distributed across some subset of the previous time steps
	- **The weight for any time step is its "credit"**
	- TAMER with credit ![[tamer_algorithm_credit.png]]
	- Credit calculation
		- Assume reinforcement signal (received at $t_0$) is targeting a single state-action pair
		- $n$ time steps that might be the target, $t_{1} \dots t_{n}$ where lower subscript is more recent
		- **Credit $c_t$ for time step starting a $t_i$ and ending at $t_{i-1}$ is the probability that the reinforcement signal was given for the event $(s,a)$ that occurred during that time step
			- $f$ is PDF over the delay of the human's reinforcement signal
			- **Have to manually design PDF?**
			- $c_{t}= \int_{t_{i-1}}^{t_i}f(x)dx$
		- In practice, maintain a window of recent time steps (lines 10, 21 in Algorithm 2)

## Results

- Evaluation against environment reward
	- $R$ communicated to the trainer, who was instructed to train his agent to earn the highest sum of environmental reward per trial episode
	- $R$ not used by agent
- In each experiment, users not told anything about the agent, only given two keys on keyboard for positive and negative reinforcement
- Tetris environment
	- Large state space, slow time step frequency
	- Algorithm 1, linear model with gradient descent
	- **Baseline algorithms "unlearn" - why?**
	- Policy search algorithms do best but take hundreds of games
		- Policy search RL: directly optimize policy $\pi(a|s)$, use policy gradient to update policy parameters by computing gradient of expected reward
			- Unstable because each policy update depends on full trajectories and rewards are noisy
			- Need on-policy training, using only the most recent policy to improve
	- TAMER gets good performance after only 3 games, much more than other value-based RL methods
		- Value based RL: learns value function which estimates expected return for a given state or state-action pair, then use this value function to take action of maximum value (indirectly training the policy)
	- What happens after more than 3 games?
		- **"Most trainers stopped giving feedback by the end of the fifth game"?**
- Mountain car environment
	- Continuous 2D state space, fast time step frequency
	- Algorithm 2, linear model with gradient descent
	- Agents shaped for three runs of 20 episodes by each trainer, first run is practice and averaged second and third for results
	- Baselines: Sarsa-3 and Sarsa-20
		- **Since the trainers get 20 episodes, shouldn't we compare to Sarsa-20 only?**
		- **Results are comparable, and only better when averaging across "best 5" trainers?**
		- **Separating the "best 5" and "worst 5" trainers post hoc is biased?**
- Future work
	- Use both $H$ and $R$, the former for early learning and the latter for finetuning
	- How to use TAMER for tasks currently intractable for autonomous learning algorithms
	- How to evaluate TAMER if trainer cannot instantly evaluate behavior
