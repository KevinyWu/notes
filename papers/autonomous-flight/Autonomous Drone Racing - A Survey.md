# Autonomous Drone Racing - A Survey

**Authors**: Drew Hanover, Antonio Loquercio, Leonard Bauersfeld, Angel Romero, Robert Penicka, Yunlong Song, Giovanni Cioffi, Elia Kaufmann, Davide Scaramuzza

#imitation-learning
#reinforcement-learning
#autonomous-flight

[[papers/autonomous-flight/README#[2024-01] Autonomous Drone Racing - A Survey|README]]

[Paper](http://arxiv.org/abs/2301.01755)

## Abstract

> Over the last decade, the use of autonomous drone systems for surveying, search and rescue, or last-mile delivery has increased exponentially. With the rise of these applications comes the need for highly robust, safety-critical algorithms which can operate drones in complex and uncertain environments. Additionally, flying fast enables drones to cover more ground which in turn increases productivity and further strengthens their use case. One proxy for developing algorithms used in high-speed navigation is the task of autonomous drone racing, where researchers program drones to fly through a sequence of gates and avoid obstacles as quickly as possible using onboard sensors and limited computational power. Speeds and accelerations exceed over 80 kph and 4 g respectively, raising significant challenges across perception, planning, control, and state estimation. To achieve maximum performance, systems require real-time algorithms that are robust to motion blur, high dynamic range, model uncertainties, aerodynamic disturbances, and often unpredictable opponents. This survey covers the progression of autonomous drone racing across model-based and learning-based approaches. We provide an overview of the field, its evolution over the years, and conclude with the biggest challenges and open questions to be faced in the future.

## Introduction

- Drone racing is far from solved
- Drones need to detect opponents and waypoints, calculate their laocation and orientation in 3D space, and compute actions that enable quick navigation
    - Perception, planning, control

## Drone Modeling

- Kinematics
    - Vehicle is 6-DOF rigit body of mass $m$ with diagonal inertia matrix $J=\text{diag}(J_{x}, J_{y}, J_{z})$
- Aerodynamics
    - The most widely used modeling assumption is that the propeller thrust and drag torque are proportional to the square of the rotational speed and that body drag is negligible
    - **Blade Element Momentum (BEM)** model
- Motor and battery
    - Most multicoptors have "throttle" command (PWM control), and rotational speed is a function of the throttle command and
- Camera and IMU
    - Stereo camera combined with IMU (inertial measurement units)
    - **Pinhole model** to estimate the focal length, image center, and distortion parameters from measurements
    - **Kalibr** to calibrate camera-IMU position and orientation and time offset
    - Biggest source of measurement error is the strong high-frequency vibrations introduced by the fast-spinning propellers

## Classical Perception, Planning, and Control Pipeline

- Perception
    - Estimates vehicle state and perceives environment using onboard sensors
    - Camera normally at 30Hz, lower rate than IMU
    - Camera affected by environment conditions: poor illumination, motion blur
    - **Visual-inertial odometry (VIO)**: uses camera and IMU measurements to estimate state (position, orientation, velocity)
        - **Frontend**: uses camera images to estimate the motion of the sensor
            - **Direct methods**: work on raw pixels, track motion of patches through consecutive images
                - Minimize photometric error defined on pixel intensities
                - Robust in featureless scenarios
            - **Feature-based methods**: extract points of interest from raw image pixels
                - Trajectory estimated by tracking these points through consecutive images
                - Robust to changes in brightness
            - Use hybrid of both methods
        - **Backend**: fuses frontend output with IMU measurements
- Planning
    - **Path planning** tackles the problem of finding a geometrical path between a given start and goal position while passing specified waypoints and avoiding obstacles
        - **Sampling-based methods**: do not construct obstacle space, but rely on random sampling of configuration space
            - PRM, RRT
        - **Combinatorial-based methods**: directly represent the obstacle or free space
            - Given some graph representation, A* or Dijkstra's algorithm can find a path
    - **Trajectory planning** uses a found geometric path to either create a collision-free flight corridor to find new waypoints for the trajectory to avoid collisions to constrain the trajectory to stay close to the found path or directly finds time allocation for a given path
        - Polynomial and spline trajectory planning
        - Optimization-based trajectory planning
        - Search-based trajectory planning
        - Sampling-based trajectory planning
- Control
    - High level controller computes desired virtual input like body rates and collective thrust
        - This is passed down to low level controller that controls rotors
    - The performance of model-based controllers degrades when the model they operate on is inaccurate
    - **For drones, defining a good enough model is an arduous process due to highly complex aerodynamic forces, which can be difficult to capture accurately within a real-time capable model**

## Learning-Based Approaches

- Replace planning, controller, and perception stack with neural network
- Require less compute and more robust to system latency and sensor noise
- **Major limitation: sample complexity**
    - Real world data collection very slow and expensive
    - Simulation data requires more engineering for real-world generalization
- Learned perception
    - Goal: use camera images to output useful representations
    - **GateNet**: CNN trained to detect gate center location
    - Event camera with **YOLO** to detect gates
- Learned planning and perception
    - Simplifies perception task: explicit notion of a map is not required
    - Reduces compute cost
    - Leverage large amounts of data to become robust against noise or dynamics
- Learned control
    - RL allows for overcoming many limitations of prior model-based controller designs by learning effective controllers directly from experience
        - **Low level controller trained with PPO outperformed tuned PID controller**
    - Imitation learning more data-efficient than RL
- Learned planning and control
    - Produce the control command directly from state inputs without requiring high-level trajectory planner
    - Deep RL can solve planning and control problem simultaneously
- End-to-end flight
    - Replace all three modules with neural network
    - Has not yet worked in real world: weak generalization to unseen environments
- **Recently, learning-based controllers have shown the ability to adapt zero-shot to large variations in hardware and external disturbances**

## Open-Source Code and Simulators

- Simulators
    - 2016: [RotorS](https://github.com/ethz-asl/rotors_simulator)
        - Lacks photorealistic details
    - 2018: [AirSim](https://github.com/microsoft/AirSim)
        - Photorealism, improves transfer learning to real world
    - 2019: [FlightGoggles](https://github.com/mit-aera/FlightGoggles)
        - Photorealistic renderer in Unity and dynamic simulation in C++
    - 2020: [FlightMare](https://github.com/uzh-rpg/flightmare)
        - Provide hardware-in-theloop simulation functions where a virtual, synthetic camera image can be provided to the drone for use in control and estimation
    - 2021: [Learning to Fly](https://github.com/utiasDSL/gym-pybullet-drones)
        - Good for RL
    - 2023: [Aerial Gym](https://github.com/ntnu-arl/aerial_gym_simulator)
        - GPU-accelerated simulator that allows simulating millions of multirotor vehicles in parallel with nonlinear geometric controllers for attitude, velocity and position tracking
- Datasets
    - [UZH-FPV Drone Racing Dataset](https://fpv.ifi.uzh.ch/)
        - Expert human pilot data
- See Table I in paper for more

## Open Research Questions and Challenges

- Reliable state estimation at high speeds
    - VIO cannot cope with sensor noise
- Flying from purely vision (like human pilots)
- Multiplayer racing
    - Anticipating behavior of opponents
- Safety
    - Collision-free flight
- Transfer to real-world applications
    - Search and rescue, inspection, agriculture, videography, delivery, passenger air vehicles, law enforcement, and defense
    - Challenge: generalization to conditions where the environmental knowledge before deployment is limited
    - Collecting data for **lifelong RL** onboard a drone is notoriously difficult
        - Drone does not have the luxury of remaining in contact with the ground like legged robots and cars, and thus has to immediately know how to hover otherwise a crash will occur
        - **Safe-RL**: exploration without crashing
