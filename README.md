# IAIP-ProtoNet
Prototypical networks (ProtoNet) for few-shot learning used in the Introduction to AI Projects (2023 spring) class team project.

## How to run

To train the ProtoNet on this task, cd into this repo's source root directory and execute:

    $ python train.py

To train with logging messages to a file (e.g., `output.log`), execute:

    $ nohup python -u train.py > output.log 2>&1 &


The script takes the following command line options:

- `-i`, `--episodes`: Number of episodes (iterations) per epoch. (default: 50)

- `-s`, `--support`: Number of support samples per class per episode. (default: 5)

- `-qt`, `--query_train`: Number of query samples for training per class per episode. (default: 5)

- `-qv`, `--query_valid`: Number of query samples for validation per class per episode. (default: 15)

- `-lr`, `learning_rate`: Learning rate for the optimizer. (default: 0.001)

- `-st`, `--step_size`: Step size of StepLR learning rate scheduler. (default: 10)

- `-g`, `--gamma`: Gamma value of StepLR learning rate scheduler. (default: 0.5)

- `-e`, `--epochs`: Number of training epochs. (default: 50)

Running the command without arguments will train the models with the default hyperparamters values.
