# Robotic Foundation Model

- [Robotic Foundation Model](#robotic-foundation-model)
  - [Background](#background)
  - [Literature](#literature)
    - [(Apr 2023) ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](#apr-2023-act-learning-fine-grained-bimanual-manipulation-with-low-cost-hardware)
    - [(Dec 2023) Octo: An Open-Source Generalist Robot Policy](#dec-2023-octo-an-open-source-generalist-robot-policy)
  - [LMN usage](#lmn-usage)

## Background

## Literature

### (Apr 2023) ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

[Code](https://github.com/tonyzhaozh/act)

[Website](https://tonyzhaozh.github.io/aloha/)

### (Dec 2023) Octo: An Open-Source Generalist Robot Policy

[Code](https://github.com/octo-models/octo)

[Website](https://octo-models.github.io/)

## LMN usage

**Directories in ```foundation``` Docker container**

- ```$LMN_CODE_DIR: /lmn/foundation/code/```
- ```$LMN_MOUNT_DIR: /lmn/foundation/mount/```
- ```$LMN_OUTPUT_DIR: /lmn/foundation/output/```
- ```$LMN_SCRIPT_DIR: /lmn/foundation/script/```

**Transfer output files in Docker container to local machine**

- ```echo 'some-output' > $LMN_OUTPUT_DIR/output_file.txt```
- In local directory, output will be in ```/.output/output/```
  - lmn will copy the output to the local machine when the container is closed
- In remote directory, output will be in ```/scratch/kevinywu/lmn/kevinywu/foundation/output/```
  - This will show up immediately, since the container is running on the remote machine
