# An Informal Introduction

[[books/learning-deep-representations/README#[1] Introduction|README]]

[Chapter](https://ma-lab-berkeley.github.io/deep-representation-learning-book/Ch1.html)

## 1.1 Intelligence, Cybernetics, and Artificial Intelligence

- Fundamental task of intelligent beings is to learn and memorize predictable information from massive amounts of sensed data
- Phylogenetic intelligence: learning through the evolution of species
	- Trial and error: nature's RL
- Ontogenetic intelligence: learning mechanisms that allow individuals to learn through its own senses and predictions withing its environment
	- Became possible with emergence of nervous system
- Societal intelligence: human communication through language
- Scientific intelligence: deduction and scientific discovery
- Norbert Weiner: founded cybernetics movement; must deal with nonlinearity to emulate typical learning mechanisms in nature

## 1.2 What to Learn?

### 1.2.1 Predictability

- Scalar case: predictable if sequence $x_{n\in}\mathbb{R}$, next number $x_{n+1}$ can be computed as $x_{n+1} = f(x_n)$
- Multivariable case: ex. Fibonnaci
	- **Autoregression is prediction of next value using past values**: $x_{n+d} = f(x_{n+d-1},\dots, x_{n}); x_{n}\in\mathbb{R}$
- Vector case: $x_{n+1} = g(x_{n})\in\mathbb{R}^d$
	- Vector can be called "state" or "token"
	- $d$ is the degree of recursion
- Controlled prediction: $x_{n+1} = f(x_{n},u_{n)}\in\mathbb{R}^d$
	- Control input $u_{n}\in\mathbb{R}^k$ is a computable predictable sequence
	- Ex. linear dynamical system: $x_{n+1} = Ax_{n}+Bu_{n}$
	- Closed-loop system: when the control input is computable from the state: $u_{n}=h(x_{n})$
		- Sequence again becomes autoregressive predictable sequence, i.e. "autonomous"
		- For linear dynamical system let $u_{n}= Fx_n$
		- Then **Linear autoregression**: $x_{n+1} = Ax_{n}+Bu_{n}= (A + BF)x_n$
- Continuous process (state-space model): $\dot{x}(t) = f(x(t)), x\in\mathbb{R}^d$
	- Controlled process: $\dot{x}(t) = f(x(t), u(t))$

### 1.2.2 Low Dimensionality

- Given many segments drawn from a predictable sequence, and new segment $S_t$, we want to predict future values
	- Generating function $f$ and order $d$ are unknown; how do we identify $f$?
- Suppose we have vectors dimension $D\gg d$
	- If values in the vectors are from a predictable sequence order $d$, the set of all vectors does not occupy $\mathbb{R}^D$
	- Given first $d$ values of $x_i$, remaining values uniquely determined
	- Predictable sequence lies on lower $d$-dimensional surface
- When predicting function $f$ is linear, long segments lie on a low-dimensional linear subspace
	- In this case, we identify $f$ with PCA
- **This observation extends to general predictable sequences: if we can identify the low-dimensional surface on which the segment samples lie, we can identify the predictive function**
	- **All modern learning methods exploit this, implicitly or explicitly**
- Properties of low-dimensionality
	- Space of all images is vast, but actual images are redundant due to strong spatial and temporal correlations among pixel values
	- Constraining an observed $x$ to lie on a low-dim surface, we make its entries highly dependent on each other -> "predictable"
	- Useful tasks
		- Completion: given more than $d$ entries of a sample $x$, the remaining entries can be uniquely determined
		- Denoising: if the entries of a sample $x$ are perturbed by noise, we can recover $x$ by projecting $x$ back onto the surface
		- Error correction: if a small number of unknown entries of $x$ are corrupted, they can be corrected
- Until now, we have considered the deterministic case of data lying on geometric structures
	- Actually, assume data follows a probability distribution with density $p(x)$
	- $p(x)$ is low-dim if density concentrates around a low-dim geometric surface
	- Once we learn $p(x)$, can estimate $x$ from noisy observations $y = f(x) + n$
		- $\hat{x}(y) = E[x|y]$ or $\hat{x}(y) \sim p(x|y)$
- **Main assumption of learning systems**: any learning method should rely on the predictability of the world; so the distribution of high-dim data should have low-dim support

## 1.3 How to Learn?
