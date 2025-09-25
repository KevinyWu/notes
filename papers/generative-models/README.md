# Generative Models

## [2020-06] Denoising Diffusion Probabilistic Models

[[Denoising Diffusion Probabilistic Models]]
- a

## [2024-11] Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion

[[Make-An-Agent - A Generalizable Policy Network Generator with Behavior-Prompted Diffusion]]
- Traditional policy learning involves using demonstrations to make states to actions, modeling a narrow behavior distribution
- Can we learn the underlying parameter distribution in parameter space by predicting optimal policy network parameters using suboptimal trajectories from offline data?
- Autoencoding policy networks into latent representations; behavior embeddings learn mutual information between trajectories and future success/failure; diffusion model conditioned on behavior embeddings generate policies
