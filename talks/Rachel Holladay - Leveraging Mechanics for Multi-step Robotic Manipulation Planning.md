# Rachel Holladay - Leveraging Mechanics for Multi-step Robotic Manipulation Planning

#tamp
#planning

[[talks/README#[2024-09-13] Rachel Holladay - Leveraging Mechanics for Multi-step Robotic Manipulation Planning|README]]

[Recording](https://youtu.be/Yn-krORGwC8?feature=shared)

## Task and Motion Planning

- Goal: solve long-horizon contact-rich tasks
- Most work has been reasoning over geometry
    - **Need to reason over geometry and physics to accomplish long-horizon manipulation tasks**
- Example: cutting cucumber, need to reason over force

## Forceful Manipulation

- Opening a childproof bottle (push and twist)
    - Must hold bottle still
    - Robot must be strong enough
- **Each joint must satisfy wrench (torque and force) constraint**
    - Represent bottle-opening process as kinematic chain
    - Robot joints
        - Manipulator Jacobian transpose maps end-effector wrench to joint torques, parametrized by joint angles
        - Use this to check if all joints satisfy robot's torque limit
    - Frictional joints via contact
        - Robot on bottle, bottle on table
        - Both are planar joints
        - Friction: limit surface
            - **Ellipsoidal approximation**
            - If exerted wrench lies in the boundary of the ellipse, we have enough friction to twist the cap
            - **Not satisfied! bottle slips**
- How to plan with these constraints?
    - **Action: parametrized controller**
        - Characterized by preconditions (what needs to be true about the world to run the action) and effects (what changes in the state)
        - Ex place, move, push
    - Problem: find a sequence of actions and their parameters subject to constraints
        - TAMP framework: PDDLStream planner
        - Must specify the streams in the planner
        - Planner searches for parameters and actions given constraints
        - Ex. to open the bottle, output of planner could be: move(params), pick(params), move(params), pushtwist-tool(params) …

## In-Hand Manipulation

- Often, you need to change the grasp of objects **in-hand**
- Assume there is a wall in environment to push objects against
- Problem
    - Given initial grasp and final grasp and where in environment we can push against, plan a pushing strategy
- **Motion cone**: the set of feasible motions that a rigid body can follow under the action of a frictional push
    - Need to construct a motion cone, then compose these to find a strategy

## Robust Decision Making

- Cutting a cucumber
    - Many ways to grasp knife
    - Different grasp better in different scenarios
- Experiment: perturb physical parameters from first section
    - Cost of an action is the negative log of the probability of success using that action under many different perturbations

## Guarded Planning

- Action uncertainty in dynamic manipulation
    - **Dynamic non-prehensile actions (DNP)**: action without grasping
    - Highly uncertain action!
    - Also hard to model with classical mechanics
- Learn a model
    - Collect sparse amount of demonstrations
    - learn **outcome volumes**: capture the space of resulting configurations given the action
    - Planner generates action sequence, execute first action, look, then repeat
    - Problem: **dead ends**, or states from which reaching the goal is impossible
    - Leverage outcome volumes avoid dead ends
- **GUARD: Guiding Uncertainty Accounting for Risk and Dynamics**
    - Setup
        - Problem: robot moving object to a goal
        - Environment has obstacles and dead ends
        - Assume discrete action space
        - Tabletop assumption: represent configuration of target object as $q_0=(f_{j}, \theta_{k}, (x,y))$
            - Face in contact, discretized planar orientation, planar position
            - Makes computation faster
    - Algorithm
        - Compute action regions, danger zones
        - Graph search using planner to reach goal while avoiding danger
    - Compared to baselines, GUARD never hits dead end (but sometimes reaches timeout)
    - **Philosophy: allow robot to keep acting my never doing anything catastrophic**
