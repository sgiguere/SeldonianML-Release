# SeldonianML

Python code implementing algorithms that provide high-probability safety guarantees for solving classification, bandit, and off-policy reinforcement learning problems.
For further details, see our paper, "Preventing undesirable behavior of intelligent machines", Science 2019. 

# Installation

First, install Python 3.x, Numpy (1.16+), and Cython (0.29+).

The remaining dependencies can be installed by executing the following command from the Python directory of : 

	pip install -r requirements.txt

# Usage

Experiments can be launched using the following command from the SeldonianML/Python directory:

     ./experiments/scripts/science_experiments_brazil.bat
     
Once the experiments complete, the figures found in th Science 2019 paper can be generated using the command, 

     python -m experiments.scripts.science_figures_brazil
     
Once completed, the new figures will be saved to `Python/figures/science/*` by default.

# License

SeldonianML is released under the MIT license.
