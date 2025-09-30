# An Informal Introduction

[[books/learning-deep-representations/README#[1] Introduction|README]]

[Chapter](https://ma-lab-berkeley.github.io/deep-representation-learning-book/Ch1.html)

## 1.1 Intelligence, Cybernetics, and Artificial Intelligence

- Fundamental task of intelligent beings is to learn and memorize predictable information from massive amounts of sensed data
- Phylogenetic intelligence: learning through the evolution of species
	- Trial and error: nature's RL
- Ontogenetic intelligence: learning mechanisms that allow individuals to learn through its own senses and predictions within its environment
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

### 1.3.1 Analytical Approaches

- Simple example: assume data is distributed according to low-dimensional Gaussian (PCA)
	- Also ICA, DL, GPCA
- Central problem of analytical mode families: identify the most compact model within each family that best fits the given data
- Linear Dynamical Systems
	- Weiner filter
	- Kalman filter
		- Finite-dimensional state-space model: $z[n] = Az[n - 1] + Bu[n] + \epsilon[n]$
		- Problem: estimate the system state $z[n]$ from noisy observations of the form $x[n] = Cz[n] + w[n]$ where $w$ is white noise
		- Kalman filter: closed-form optimal causal state estimator that minimizes the minimum-MSE prediction error $\min E[\|x[n] - Cz[n]\|_2^2]$
		- Can introduce linear state feedback, ex. $u[n] = F\hat{z}[n]$ to render the closed-loop system fully autonomous
		- Enables estimation of a dynamical system's state from noisy observations
	- Both filters try to estimate random variable $x_0$ from its noisy observations $x = x_{0}+ \epsilon; x_{0}\sim S$ where $S$ is a low-dim linear subspace
		- Projecting data onto this subspace to obtain optimal denoising operations

### 1.3.2 Empirical Approaches

- Classical artificial neural networks
	- Sufficiently large multi-layer network can learn any finite state machine
		- Finite state machine: finite set of states, transition function that maps current state + input -> next state, output function
		- Hidden layer encodes current state
		- Weights and activations implement transition rules
		- Output layer is output function
	- LeNet: Yann LeCun (Turing Award) used backprop to learn a deep convolutional network for recognizing handwritten digits
		- Prototype for AlexNet and ResNet
	- Compressive autoencoder
		- **Neural networks can learn low-dimensional representations for data with nonlinear distributions**
		- Learn an autoencoder (encoder + decoder)
			- Called "auto" because system is trained to reproduce its own input without labels (unsupervised)
			- Often used for compression, denoising, reducing dimensions before feeding into another model
			- $X \xrightarrow{f} Z \xrightarrow{g} \hat{X}$
			- Enforce consistency between decoded data $\hat{X}$ and original $X$ by minimizing MSE-type reconstruction error: $\min_{f,g} \|X - \hat{X} \|^{2}_{2}$ where $\hat{X} = g(f(X))$
			- How do we avoid $f,g$ being simply identity maps?
				- Need to have some compression in the hidden representation $Z$
				- Geoffrey Hinton: minimize coding length (number of bits needed to represent the data) as a measure of compression
				- See ch. 5
- Modern deep neural networks
	- Classification and recognition
		- ImageNet: large dataset
		- AlexNet: similar to LeNet but larger and replaces sigmoid in LeNet with ReLU (Geoffrey Hinton Turing Award)
		- VGG, GoogLeNet, ResNet, Transformers
	- Reinforcement learning
		- Deep networks to model decision/control policy (best action to take maximize expected reward) or optimal value function (estimate of expected reward from given state)
		- AlphaGo thought to have an action space too vast
			- Possible explanation: optimal value and policy function of Go have low enough intrinsic dimensions to be well approximated by neural networks with limited samples
	- Generation and prediction
		- Previously, we mostly cared about encoding; retaining sufficient statistics for a task
		- In modern foundation models, we also need to decode $Z$
			- $X$ represents data observed from the world good decoder allows us to simulate or predict the world
			- Ex. text-to-image or text-to-video tasks
		- Discriminative approaches
			- For generated images to resemble natural images, need to minimize distance $\min d(X, \hat{X})$
			- Hard to do in high-dim space
			- Instead, Zhuowen Tu: learn a discriminator to separator $\hat{X}, X$; harder to separate, the closer they are
			- Goodfellow: Generative Adversarial Network (Yoshua Bengio Turing Award)
				- Model generator $g$ and discriminator $d$ with deep networks, learned via minimax game: $\min_{g}\max_{d}\mathcal{l}(X,\hat{X})$
					- Discriminator wants to maximize distance (to more clearly separate $X, \hat{X}$)
					- Generator wants to make $\hat{X}$ indistinguishable from $X$
				- With properly chosen loss, this minimax is equivalent to minimizing Jenson-Shannon distance between $X, \hat{X}$, which remains difficult
					- JS-divergence: symmetric version of KL divergence, measures "How different are these two distributions when compared to their average?"
					- KL divergence $KL(P\|Q)$ (relative entropy): how much information is lost when using $Q$ to approximate $P$
				- GAN relies on heuristics and engineering tricks and suffers from instability issues like mode collapsing ($g$ mapping many $z$ values to the same few outputs)
		- Generation via denoising and diffusion
			- Transforms generic Gaussian distribution to an empirical distribution via denoising
			- Diffusion has by now replaced GANs

## 1.4 A Unifying Approach

### 1.4.1 Learning Parsimonious Representations

- Require sequences to be computable, tractable, and scalable
- Complexity of the algorithms should scale well with the data
- Pursuing low-dimensionality via compression
	- Computational cost depends on predicting function $f$
	- Higher degree of regression $d$: more costly to compute
	- **Kolmogorov**: among all programs computing the same sequence, the length of the shortest program measures its complexity
		- $K(S) = \min_{p : \mathcal{U}(p) = S}L(p)$
		- $p$ is a program generating sequence $S$ on universal computer $\mathcal{U}$
			- This definition is non-constructive: does not provide a way to find the shortest possible program
		- **Occam's Razor**: choose the simplest theory explaining the same observation
	- **Shannon**: entropy of distribution provides a measure of complexity
		- $h(S) = -\int p(s)\log p(s) ds$
		- Sequence $S$ drawn from distribution $p(S)$
		- Average number of binary bits needed to encode a sequence (ch. 3)
	- **Coding rate**: for an encoding $\mathcal{E}$, the complexity of the predicting function $f$ can be evaluated as the average coding length $L(\mathcal{E}(S))$ for all sequences that it generates
		- $R(S|\mathcal{E}) = E[L(\mathcal{E}(S))] \approx \frac{1}{N}\sum\limits_{i = 1}^{N}L(\mathcal{E}(S_i))$
		- Goal of learning the data distribution: $\min_{\mathcal{E}, \mathcal{D}} R(S|\mathcal{E}) \quad\text{s.t.}\quad \text{dist}(S, \mathcal{D}(\mathcal{E}(S))) \leq \epsilon$
		- Minimizing the coding rate minimizes an upper bound of the uncomputable Kolmogorov complexity: $K(S) < L(\mathcal{E}(S))$

### 1.4.2 Learning Informative Representations

- Want to encode the low-dim distribution in data in a structured or organized way to be interpretable
- In ML, want representations to retain structure to be useful for prediction
- Layers of a deep network perform operations that incrementally optimize the objective function of interest
	- Role of deep networks can be interpreted as emulating optimizing information gain
		- **Information gain** measures how much uncertainty is reduced (or, equivalently, how much information is gained) when we move from one state of knowledge to another
		- In deep neural net, it is how smaller the conditional entropy is compared to the baseline uncertainty (just $H(Y)$, or the raw entropy of the data)
		- Conditional entropy $H(Y|X)$: how much knowing about $X$ tells you about $Y$
		- Information gain = $R(S|\mathcal{E}_1) - R(S|\mathcal{E}_2) > 0$; how much information $\mathcal{E}_{2}$ has over $\mathcal{E}_1$
	- Layers of the deep architectures can gain statistical geometric interpretations: performing compressive encoding an decoding operations
	- Deep networks become "white boxes" that are mathematically explainable

### 1.4.3 Learning Consistent Representations

- Let $X = \{S_{1,}\dots S_N\}\subset \mathbb{R}^D$
- Let $Z = \mathcal{S}(X)$ be the codes of $X$ via some encoder $\mathcal{E}$
- $Z$ is called the "features" of $X$
- We do not know which encoder $\mathcal{E}$ should be used to retain the most useful information about the distribution of $X$, which is unknown
	- Try to learn encoder that optimizes an empirical measure of parsimony for the learned representation, such as cross-entropy for classification tasks
	- Cross entropy encourages very lossy encoding; learned $Z$ only contains information about class type
	- Lossless coding only practical if $X$ is discrete
- **Autoencoder**: learns a compressive coding scheme $\mathcal{E}$ to identify low-dim structures in $X$ to predict things in the space of $X$
	- Requires an efficient decoding scheme $\mathcal{D}$, ideally the inverse of the encoding
	- Need $\mathcal{D}$ to exist an be efficiently realizable and physically implementable
	- Learn a compact representation $Z$ that can still predict $X$ well; since encoding and decoding must be lossy
		- $\min_{f,g} [\mathcal{l}(X,\hat{X}) +\rho(Z)]$ where $\hat{X} = g(f(X)), Z = f(X)$
		- $\mathcal{l}$ is a distance function that promotes similarity between $X$ and $\hat{X}$ and $\rho(Z)$ is a measure that promotes parsimony and information richness of $Z$
	- PCA is **consistent** (decoder can closely reconstruct original data from encoder); see ch. 2

### 1.4.4 Learning Self-Consistent Representations

- Often expensive to evaluate how close $X$ is to $\hat{X}$, like pixel-to-pixel
- **Closed-loop transcription**: encode $\hat{X}$ to $\hat{Z}$ and checking consistency between $Z, \hat{Z}$
	- Ch. 5: self-consistency in $Z$ implies consistency in $X$
	- $\max_{f}\min_{g} [\mathcal{l}(Z, \hat{Z}) + \rho (Z)]$
	- The decoder learns to **invert** the encoder on its code manifold (minimizing feature inconsistency), while the encoder learns codes that are **worth having** (maximize $\rho$) **and** are **stable under decode-re-encode** (so the decoder's best effort keeps $\mathcal{l}$ small).

## 1.5 Bridging Theory and Practice for Machine Intelligence

- Ch. 4: open-ended encoding
- Ch. 5.1: bi-directional encoding
- Ch. 5.2: closed-loop encoding
- A fundamental characteristic of any intelligent being or system is the ability to continuously improve or gain information on its own
