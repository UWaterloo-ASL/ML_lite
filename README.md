# ML_lite
A package for people in the studio to perform tests involve RL agents

## Dependency
* OpenAI baseline https://github.com/openai/baselines
     
     Please check out the versions on Sept 18, not the latest version. As OpenAI has reworked the implementation of DDPG. In order to do so, please follow the instructions:
     1. clone baselines repository
     ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
    2. Go to Baselines Github repo and look through [commits history](https://github.com/openai/baselines/commits).
    Find the commit that you want to use, in this case it is 
    ```
    115b59d28b79523826dd5a81fbc5d6f8ed431c7c
    ```
    3. Checkout specific commit
    ```
    git checkout 115b59d28b79523826dd5a81fbc5d6f8ed431c7c
    ```
    You will see info like following:
    ```
    Note: checking out '115b59d28b79523826dd5a81fbc5d6f8ed431c7c'.
    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by performing another checkout.
    
    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -b with the checkout command again. Example:
          git checkout -b new_branch_name
    HEAD is now at e442f37... show PASS and FAIL only if they are not 0
    ```
    4. install baseline
    ```
    pip install -e .
    ```
    
    Ignore installing Mujoco if required. The Mujoco library is not used in this project.
     
* Tensorflow https://www.tensorflow.org/install/
