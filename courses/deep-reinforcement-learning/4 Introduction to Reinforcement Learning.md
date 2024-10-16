# Introduction to Reinforcement Learning

#markov-decision-process
#q-learning

[[courses/deep-reinforcement-learning/README#[4] Introduction to Reinforcement Learning|README]]

Lecture [4.1](https://youtu.be/jds0Wh9jTvE?feature=shared), [4.2](https://youtu.be/Cip5UeGrCEE?feature=shared), [4.3](https://youtu.be/Pua9zO_YmKA?feature=shared), [4.4](https://youtu.be/eG9-F4r5k70?feature=shared), [4.5](https://youtu.be/dFqoGAyofUQ?feature=shared), [4.6](https://youtu.be/hfj9mS3nTLU?feature=shared)

## Markov Decision Process

- Markov chain
	- $\mathcal{M} = \{\mathcal{S}, \mathcal{T}\}$
		- $\mathcal{S}$: state space
		- $\mathcal{T}$: transition operator
			- Let $\mu_{t, i} = p(s_t = i)$
			- Then $\mu_{t}$ is a vector of probabilities
			- $\mathcal{T}_{i,j} = p(s_{t+1} = i | s_t = j)$
			- Then $\mu_{t+1} = \mathcal{T} \mu_t$
	- Consider $p(s_{t+1} | s_t)$
- Markov decision process
	- $\mathcal{M} = \{\mathcal{S}, \mathcal{A}, \mathcal{T}, r\}$
		- Adds action space $\mathcal{A}$ and reward function $r$ to Markov chain definition
		- $\mu_{t,j} = p(s_t = j)$
		- $\eta_{t,k} = p(a_t = k)$
		- $\mathcal{T}_{i,j,k} = p(s_{t+1} = i | s_t = j, a_t = k)$
		- Then $\mu_{t+1, i} = \sum_{j,k} \mathcal{T}_{i,j,k} \eta_{t,k} \mu_{t,j}$
		- $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$
- Partially observed Markov decision process
	- $\mathcal{M} = \{\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{\epsilon}, r\}$
		- Adds observation space $\mathcal{O}$ and emission probability $\mathcal{\epsilon}$ to MDP definition
		- $\mathcal{\epsilon}_{i,j} = p(o_t = i | s_t = j)$

## Q-Learning

- **Goal of RL: maximize expected reward** ![[rl_goal.png]]
	- **Finite horizon case**: $\theta^* = \arg\max_{\theta} E_{\tau \sim p_{\theta}(\tau)} \left [ \sum_{t=1}^{T} r(s_t, a_t) \right ] = \arg \max_{\theta} \sum_{t=1}^{T} E_{(s_t, a_t)\sim p_{\theta}(s_t,a_t)}[r(s_t, a_t)]$
		- $p_{\theta}(s_t,a_t)$ is the state-action marginal
	- **Infinite horizon case**: if $T=\infty$ above
		- **Stationary distribution**: $\mu = \mu \mathcal{T} \leftrightarrow (\mathcal{T}-I)\mu=0$ so $\mu$ is an eigenvector of $\mathcal{T}$ with eigenvalue 1
		- Does $p(s_t, a_t)$ converge to a stationary distribution?
		- If so, then $\theta^* = \arg \max_{\theta} \frac{1}{T} \sum_{t=1}^{T} E_{(s_t, a_t)\sim p_{\theta}(s_t,a_t)}[r(s_t, a_t)]$ converges to $\theta^* = \arg \max_{\theta} E_{(s, a)\sim p_{\theta}(s,a)}[r(s, a)]$ as $T\rightarrow \infty$
	- In RL, we care about expectations (reward functions not necessarily smooth, but $E_{\pi_{\theta}}[r(s, a)]$ is smooth in $\theta$)
- Value functions ![[rl_algo.png]]
	- **Q-function**: $Q^{\pi}(s_t, a_t) = \sum_{t'=t}^{T} E_{\pi_{\theta}}[r(s_{t'}, a_{t'})|s_t, a_t]$
		- Total reward if you take action $a_t$ in state $s_t$ and follow policy $\pi_{\theta}$ thereafter
	- **Value function**: $V^{\pi}(s_t) = \sum_{t'=t}^{T} E_{\pi_{\theta}}[r(s_{t'}, a_{t'})|s_t] = E_{a_t \sim \pi(a_t|s_t)}[Q^{\pi}(s_t, a_t)]$
		- Total reward if you start in state $s_t$ and follow policy $\pi_{\theta}$ thereafter
	- Using Q-functions and value functions in algorithms
		- If we have policy $\pi$ and we know $Q^{\pi}$, then we can improve $\pi$ by choosing letting $\pi'(a|s)=1$ if $a = \arg\max_{a} Q^{\pi}(s, a)$
		- We can compute the gradient to increase the probability of good actions $a$: if $Q^{\pi}(s,a) > V^{\pi}(s)$, then $a$ is better than average, so modify $\pi(a|s)$ to increase the probability of $a$

## Types of Algorithms

- Types of algorithms
    - Policy gradient: directly differentiate $\theta^* = \arg\max_{\theta} E_{\tau \sim p_{\theta}(\tau)} \left [ \sum_{t=1}^{T} r(s_t, a_t) \right ]$
		- ex. REINFORCE, NPG, TRPO, PPO
    - Value-based: estimate value function or Q-function of the optimal policy (no explicit policy)
		- ex. Q-learning, DQN
    - Actor-critic: estimate value function or Q-function of current policy, use it to improve the policy
		- ex. A3C, SAC
    - Model-based: learn a model of the environment, use it to plan or improve policy
		- ex. Dyna, guided policy search
		- Trajectory optimization: just planning, no policy
		- Backpropagate gradients into the policy
		- Use the model to learn a value function
- Tradeoffs
    - Sample efficiency: how many samples are needed to learn a good policy?
    - **Off policy**: able to improve policy without generating new samples from that policy
    - **On policy**: need to generate new samples every time the policy is changed ![[sample_efficiency.png]]
    - Convergence: often don't use gradient descent
		- Many value-based algorithms are not guaranteed to converge
		- Model-based RL minimizes error of fit, but no guarantee that better model leads to better policy
		- Policy gradient: uses gradient descent, but often least efficient
- Common assumptions
    - Full observability
    - Episodic learning
    - Continuity or smoothness
