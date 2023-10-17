# Proximal Policy Optimization and Super Mario Bros

This repository contains both the code and the report on the implementation and understanding of the Proximal Policy Optimization (PPO) algorithm in the game Super Mario Bros

![](img/level1.gif)

## Introduction

Proximal Policy Optimization (PPO) is a type of Reinforcement Learning algorithm that has gained significant attention in recent years. It addresses some of the challenges faced by earlier policy gradient methods, providing more stable and consistent training. 

In this repository, PPO is applied to train an agent to navigate the challenges and adversaries in Super Mario Bros. For a detailed understanding and hands-on examples, it's highly recommended to read the report provided.

## Getting Started

### Prerequisites

1. Python 3.x
2. [pip](https://pip.pypa.io/en/stable/)

### Installation

Follow these steps to get up and running:

1. **Clone the repository:**
    ```bash
    git clone <https://github.com/Alex-Hawking/PPO_Super_Mario_Bros.git>
    cd <PPO_Super_Mario_Bros>
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```
    (Would highly recommend)

3. **Activate the virtual environment:**
    - **Linux/Mac:**
        ```bash
        source venv/bin/activate
        ```
    - **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Directory structure**

    Ensure your directory is structured as below:


    ```
    ├── PPO_Super_Mario_Bros/
    │ ├── main.py
    │ ├── src/
    │ ├── model/
    │ | ├── checkpoints/
    ```

6. **Device setup**

    Ensure you device is correctly set by following the steps at the stop of `src/agent.py`

## Usage

After you've installed the prerequisites and have you directory set up correctly, you can run the agent:

```bash
python main.py
```

This will begin training a model using the default hyperparameters. However you can make changes to the hyperparameters and functionality of the model by changing variables located at the top of `main.py`.

I have included a partially trained model in the checkpoints folder, it should be able to complete level 1 :)

To understand what these do and how they work I would recommending reading the short report I wrote on PPO and its implementation in Super Mario Bros.
