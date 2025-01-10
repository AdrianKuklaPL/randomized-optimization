# Randomized Optimization and Neural Network Weights Optimization

## Objective
This repository contains analyses of two discrete optimization problems (FlipFlop and Knapsack) using randomized optimization algorithms, as well as neural network weights optimization. The project evaluates the performance of randomized algorithms such as Randomized Hill Climbing, Simulated Annealing, Genetic Algorithm, and MIMIC. Additionally, the impact of these algorithms on neural network weight optimization is explored.

## Technologies Used
Python


## Analyses Performed
### Discrete Optimization:
- Applied Randomized Hill Climbing (RHC), Simulated Annealing (SA), Genetic Algorithm (GA), and MIMIC to FlipFlop and Knapsack problems.
- Evaluated performance across various problem sizes.
- Visualized results for algorithm comparison and performance trends.

### Neural Network Weights Optimization:
- Optimized neural network weights using RHC, SA, and GA.
- Performed cross-validation to identify the best models.
- Evaluated and visualized model performance using metrics such as accuracy and F1-score.

## Libraries Used
The following libraries are required and can be installed via `pip`:
- `ucimlrepo`
- `numpy`
- `pandas`
- `mlrose_ky`
- `torch`
- `skorch`
- `sklearn`
- `pyperch`
- `imblearn`
- `matplotlib`
- `time`

## Datasets
The dataset for neural network analysis is retrieved from the UCI Machine Learning Repository:
- [Drug Consumption (Quantified) Dataset](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)

## Code Structure
### Discrete Optimization:
For the FlipFlop problem, run the scripts in the following order:
1. `1_flipflop_ga/rhc/sa.py` – Implements optimization algorithms.
2. `2_flipflop_problemsizes.py` – Analyzes performance across problem sizes.
3. `3_flipflop_graphs.py` – Visualizes the results.

For the Knapsack problem, run the scripts in the following order:
1. `1_knapsack_ga/rhc/sa/mimic.py` – Implements optimization algorithms.
2. `2_knapsack_problemsizes.py` – Analyzes performance across problem sizes.
3. `3_knapsack_graphs.py` – Visualizes the results.

### Neural Network Weights Optimization:
For neural network analysis, run the scripts in the following order:
1. `1_NN_CV.py` – Performs cross-validation to select the best models.
2. `2_NN_Final_Models.py` – Evaluates final models using selected parameters.

## How to Replicate
1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianKuklaPL/randomized-optimization.git
