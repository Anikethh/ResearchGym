# RL Starter Repository

## Installation

This repository provides a starting point for online RL experimentation.

Tested on Python 3.8. If you don't have MuJoCo installed, follow the instructions at `https://github.com/openai/mujoco-py#install-mujoco`.

## Running Instructions

```bash
python synther/online/online.py --env quadruped-walk-v0 --gin_config_files config/online/sac.gin
```
