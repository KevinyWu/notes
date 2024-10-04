# Antonio Loquercio - Learning Vision-Based High-Speed Flight

#tags

[[talks/README#[2022-03-28] Antonio Loquercio - Learning Vision-Based High-Speed Flight|README]]

[Recording](https://www.youtube.com/watch?v=5BA1Wm6SWQY)

## Introduction

- Why high-speed flight?
    - Make drones faster increases range (short battery life)
    - Applications: search and rescue, delivery, exploration
- Problem formulation
    - $\min_{\pi} J(\pi) = E_{\rho(\pi)} \left [ \sum_{k=0}^{T} \mathcal{C}(\tau[k], s[k])\right ]$
    - Minimize expected over observed states $\rho(\pi)$
    - Time-dependent task $\tau[k]$
    - Quadrotor and environment state $s[k]$
    - **Important**: the policy $\pi$ does not not observe state $s[k]$, but only on-board sensor observations
        - **No GPS, no prior knowledge of environment**

## Section 2

- Notes from the second section
