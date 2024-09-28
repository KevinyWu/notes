# Lerrel Pinto - On Building General-Purpose Home Robots

#foundation-models
#robot-learning
#imitation-learning

[[talks/README#[2024-04-09] Lerrel Pinto - On Building General-Purpose Home Robots|README]]

[Recording](https://www.youtube.com/watch?v=ETSKgiW76dg)

## Constructivist View of Robot Learning

- Constructivist: learn by interacting with human knowledge and guidance
    - 1) Build on prior knowledge
    - 2) Construct new knowledge from human data
    - 3) Assimilate and accommodate new information from interaction

## OK-Robot: Zero-Shot Pick and Place

- **OK-Robot**: uses existing pre-trained models
    1. Scan room (i.e. with phone) with RGB-D
    - 2) Use **VoxelMap** algorithm to make a point cloud 3D scene
        - Odometry to get spatial 3D coordinates
    - 3) **OWL-ViT** to create object masks point cloud
    - 4) **CLIP** image encoder gets image representation that maps to a text representation
    - 5) For each CLIP embedding, we can assign a XYZ coordinate
- Thus, given a text query, we can get an XYZ coordinate
    - A* for pathfinding
    - After getting close to the object
        - **LangSAM** to segment the object from the image
        - **AnyGrasp** pretrained grasping model
- What to do after object is moved?
    - Need to update the point cloud
    - No solution currently
- Failure modes
    - VLM incantaions
        - Ex. Model cannot detect "Grey eye glass box" but can detect "grey eyeglass box" (with no space)
    - Grasping models fail with transparent objects (like empty bottles), flat objects (intersect with surface)
- Limitations
    - Can only do two tasks: navigate to objects, and pick-and-place
    - **Solution: collect data using a iPhone mounted on a reach-grabber stick (looks similar to Hello-Robot Stretch)**
        - 10 homes, 109 tasks, 81% success rate
- **Key ingredients for deploying in new homes**
    - 1) Large pretraining dataset: **Homes of New York**
        - 13 hours, 1.5 million frames, 22 homes
    - 2) Good pretrained representation
        - Data augmentation, contrastive loss
        - Used MoCo-V2
    - 3) Fast finetuning system
        - For RGB module: use pretrained ResNet to obtain representation
        - For Depth module: use median filtering to reduce spatial resolution
            - Depth helps know when you are making contact with object
        - 2-layer MLP for action prediction

## Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations

- Failures are inevitable, how do we learn from failures?
- **Use representation as rewards**
    - Encode both expert and agent to encoders
    - Get an expert and non-expert trajectory (one point for each frame)
    - **Optimal transport matching: how similar is agent to expert trajectory?**
    - Use this similarity as reward
    - **Real-world RL with just 1 min of expert data, 20 min of interaction!**
- Limitations
    - Works when prior policy is close to optimal policy
    - If robot doesn't succeed within ~10 minutes of interaction, it probably cannot learn the expert
