# Antonio Loquercio - Learning Vision-Based High-Speed Flight

#imitation-learning
#autonomous-flight

[[talks/README#[2022-03-28] Antonio Loquercio - Learning Vision-Based High-Speed Flight|README]]

[Recording](https://www.youtube.com/watch?v=5BA1Wm6SWQY)
[Code](https://github.com/uzh-rpg/agile_autonomy)

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
- Traditional pipeline: perception -> planning -> control
    - Limitations: latency, compounding errors in imperfect perception
        - Transparent and easy to interpret, easy to engineer
    - Now: replace all three with neural network
        - Robust to imperfect perception, low latency

## Imitation Learning for Agile Flight

- Traditional navigation, planning, and control algorithms (RTT*, A*, MPC, LQR, etc.) require access to ground-truth state of robot and the environment $s[k]$
- Imitation learning: neural network that only requires on-board sensor
- Data from simulation
    - **Input-output (I/O)**
        - Allows for transfer learning between real and sim
        - **Abstraction function** $f$ reduces gap between sim and real
            - $DW(f(M), f(L))\leq DW(M, L)$
            - Task and sensor dependent
            - Ex. $f(img)$ is the depth map, so real and sim look more similar
    - **Simulator: [Flightmare](https://github.com/uzh-rpg/flightmare?tab=readme-ov-file)**
        - Uses open-source environments from Unity community
        - 230K samples - one sample includes:
            - Stereo image pair
            - Drone state
            - Depth extracted with SGM
            - 10 collision free trajectories
        - 1K obstacle point clouds
    - **Algorithm**
        - Sensor observations are **asynchronous**
            - Stereo depth: 20Hz
            - IMU: 100Hz
            - Reference direction: 50Hz
        - Process all information separately using temporal convolution
            - Concatenate before input to neural network
            - **Predict multiple trajectories**
        - ![[loquercio-agile-flight.png]]
- Controlled experiments
    - Goal-reaching in simulation
    - Robust to noise and latency

## Future Directions

- Robots don't learn, they just crash
- How to efficiently learn from agent's mistakes?
    - Uncertainty estimation: go into safety mode when network is uncertain (i.e. under sensor failure)
