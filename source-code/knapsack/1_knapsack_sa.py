import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import SARunner
from mlrose_ky.algorithms.decay import ExpDecay, ArithDecay, GeomDecay

# Define the Knapsack fitness function
np.random.seed(1)
weights = np.random.randint(1, 100, size=100)
values = np.random.randint(1, 50, size=100)
max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

# Create the Knapsack fitness function
fitness = mlrose.Knapsack(weights, values, max_weight_pct)

# Define the optimization problem object for the Knapsack problem
problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

# Create the temperature and decay schedule using Exponential Decay
temperature_list = [0.1, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0, 10000000.0]
decay_list = [ExpDecay(init_temp=t) for t in temperature_list]

# Define a delta convergence threshold
convergence_threshold = 1e-6  # Example threshold

# Define the parameters for the Simulated Annealing algorithm
sa_runner = SARunner(
    problem=problem,
    experiment_name="Knapsack_SA_ExpDecay",
    seed=1,
    iteration_list=2**np.arange(11),  # List of iterations
    temperature_list=temperature_list,  # Temperature values
    decay_list=[ExpDecay],  # Use the ExpDecay schedule
    max_attempts=1000
)

# Run the experiment and capture results
df_run_stats, df_run_curves = sa_runner.run()

# Save results to CSV
df_run_curves.to_csv('knapsack_sa_results.csv', index=False)
print("Results saved to knapsack_sa_results.csv")

# Load the CSV file into a DataFrame
file_path = 'knapsack_sa_results.csv'
df = pd.read_csv(file_path)

# Group by temperature and get the maximum fitness for each temperature
df_max_fitness = df.groupby('Temperature')['Fitness'].max().reset_index()

# Print the result
print(df_max_fitness)


# Iteration                                                               512
# Fitness                                                              1777.0
# FEvals                                                                  552
# Time                                                               0.279909
# State                     [np.int32(0), np.int32(1), np.int32(0), np.int...
# schedule_type                                                     geometric
# schedule_init_temp                                                     50.0
# schedule_decay                                                         0.99
# schedule_min_temp                                                     0.001
# schedule_current_value                                            49.859539
# Temperature                                                            50.0
# max_iters                                                              1024

# Iteration                                                               512
# Fitness                                                              1816.0
# FEvals                                                                  498
# Time                                                               0.125601
# State                     [np.int32(0), np.int32(1), np.int32(0), np.int...
# schedule_type                                                   exponential
# schedule_init_temp                                                     10.0
# schedule_exp_const                                                    0.005
# schedule_min_temp                                                     0.001
# schedule_current_value                                             9.993722
# Temperature               ExpDecay(init_temp=10.0, exp_const=0.005, min_...
# max_iters                                                              1024

# Iteration                                                               512
# Fitness                                                              1795.0
# FEvals                                                                  637
# Time                                                               0.141396
# State                     [np.int32(1), np.int32(1), np.int32(0), np.int...
# schedule_type                                                    arithmetic
# schedule_init_temp                                                     10.0
# schedule_decay                                                       0.0001
# schedule_min_temp                                                     0.001
# schedule_current_value                                             9.999986
# Temperature               ArithDecay(init_temp=10.0, decay=0.0001, min_t...
# max_iters                                                              1024

########## SA Best Model ############

# Generate random weights and values for 1000 items
np.random.seed(1)  # Set seed for reproducibility
weights = np.random.randint(1, 100, size=100)  # Random weights between 1 and 100
values = np.random.randint(1, 50, size=100)  # Random values between 1 and 50
max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

# Define the Knapsack fitness function with 1000 items
fitness = mlrose.Knapsack(weights, values, max_weight_pct)

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

# Create an SA Runner instance to solve the Knapsack problem
sa = SARunner(
    problem=problem,
    experiment_name="knapsack_sa",
    seed=1,
    output_directory=None,  # Specify output directory if needed
    temperature_list=[10],  # Adjust temperature if necessary
    decay_list=[mlrose.ExpDecay],
    iteration_list=2 ** np.arange(13),  # Iterations from 2^0 to 2^12
)

# Run the SA Runner and retrieve its results
df_run_stats, df_run_curves = sa.run()

# Get the best fitness score (max for Knapsack problem)
best_fitness = df_run_stats["Fitness"].max()

# Print the best fitness score
print(f"Best Fitness Score: {best_fitness}")

# Keep only relevant columns
df_filtered = df_run_curves[["Iteration", "Time", "Fitness", "FEvals", "max_iters"]].copy()

# Add a new column 'Algorithm' with the value 'SA'
df_filtered["Algorithm"] = "SA"

# Save the filtered results to a CSV file
df_filtered.to_csv('knapsack_sa_best_filtered_results.csv', index=False)
print("Filtered results saved to knapsack_sa_best_filtered_results.csv")
