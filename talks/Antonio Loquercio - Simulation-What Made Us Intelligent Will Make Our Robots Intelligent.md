# Antonio Loquercio - Simulation-What Made Us Intelligent Will Make Our Robots Intelligent

#sim2real
#3d-scenes
#learning-from-video
#tactile

[[talks/README#[2024-10-11] Antonio Loquercio - Simulation-What Made Us Intelligent Will Make Our Robots Intelligent|README]]

[Recording](https://www.youtube.com/watch?v=1nNABpXvQfA)

## Mammalian Brain

- Neocortex
    - Theory: it works like simulation: we simulate what we think we see, then we try to ground this simulation to actual observation
- Recognition and simulation are connected for mammalian brain
- Not specific to vision: other senses used in the same way
- **Sensorimotor simulation enhances both high-level skills (planning) and low-level skills (motor control)**
- Controllers like LQR, MPC rely on simple, state-based models

## Previous Sim2Real Work in Robotics

- OpenAI dextrous in-hand manipulation of alphabet blocks
- OpenAI Rubik's cube
- [[Autonomous Drone Racing - A Survey|Drone racing]]
    - Perception system
    - Control policy trained only in simulation with RL
    - Need very accurate physics model of system
        - Impossible to obtain **globally** accurate model
    - **Locally** accurate model (in the neighborhood of the racetrack) is enough
        - **Train policy in simulation, evaluate in real world, compute difference between predicted and actions observed, use this to update local model**
    - Autonomous drones can take tighter turns than humans: **better planners**

## Tactile-Augmented Radiance Fields

- [[Tactile-Augmented Radiance Fields]]
- **Vision-based touch sensors**: camera with gel in front of camera, object presses into gel and creates illumination gradient
- Collect data with a device that has a vision-based touch sensor and a camera, then calibrate both modes to compose the TaRF
    - Use this large vision-tactile dataset to input RGB-D input and predict touch
- Downstream tasks
    - Material localization: given a touch image, detect in visual image what objects feel like this
- Simulated data can augment the dataset
- This is an active research field - NeRF might be overkill

## Self-Simulation Helps Simulating Others

- Social projection theory: people expect others to be similar to themselves
- Learning from videos can help scale up robot data
    - Can learn intuitive physics, contact poses, pre/post-contact trajectories, human preferences, etc
- [[Hand-Object Interaction Pretraining from Videos]]
    - 3D hand-object detection to get 3D trajectory of human hands
    - Retargeting human action to robot action
        - Use high-fidelity dynamics as constraints
        - The optimizer and environment use different dynamics
    - **Simplifying assumption about objects
        - Ignore object dynamics altogether in the optimization, only consider object kinematics
        - We don't know friction, mass, etc. of object
    - Transfer human video to robot simulation
        - Allows for augmentation: randomize location of objects, lighting, initial robot pose, etc.
    - Improves BC with very few demonstrations

## Towards a Unifying Theory for Sim2Real

- Move away from physics-based simulators, and include senses beyond vision
    - Perfect simulators are not scalable
    - Where is the data coming from?
        - People underestimate the amount of data robots can generate on their own
        - Releasing a robust robot in the wild: **self-supervised data collection**
- Leverage duality between generation and action
    - "What I can't predict, I can't act on"
- Don't specialize but condition on the task
    - Changing architecture for task is not scalable
    - [[Hand-Object Interaction Pretraining from Videos]] learns basic manipulation actions from human videos regardless of intent
