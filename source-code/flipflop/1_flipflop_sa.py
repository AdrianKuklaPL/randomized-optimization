import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import SARunner
from mlrose_ky.algorithms.decay import GeomDecay, ArithDecay, ExpDecay

############## Cross Validation ##############

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=1000, fitness_fn=fitness, maximize=True, max_val=2)

# Create an SA Runner instance to solve the Flip Flop problem
sa = SARunner(
    problem=problem,
    experiment_name="flipflop_sa",
    seed=1,
    output_directory=None,  # Specify output directory if needed
    max_attempts=100,
    temperature_list=[0.1, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0, 10000000.0],
    decay_list=[mlrose.ArithDecay],
    iteration_list=2 ** np.arange(11),  # Iterations from 2^0 to 2^11
)

# Run the SA Runner and retrieve its results
df_run_stats, df_run_curves = sa.run()

# Get the best (max) fitness score
best_fitness = df_run_stats["Fitness"].max()

# Get all runs with the best fitness value
best_runs = df_run_stats[df_run_stats["Fitness"] == best_fitness]

# Extract the run with the minimum number of evaluations (FEvals)
minimum_evaluations = best_runs["FEvals"].min()
best_run = best_runs[best_runs["FEvals"] == minimum_evaluations]

# Print the best run details
print(best_run.iloc[0])

# Get the best temperature parameter
best_temperature_param = best_run["Temperature"].iloc[0].init_temp

# Filter run stats for the best run and print relevant details
run_stats_best_run = df_run_stats[df_run_stats["schedule_init_temp"] == best_temperature_param]
print(run_stats_best_run[["Iteration", "Fitness", "FEvals", "Time", "State"]])


# Iteration                                                              1024
# Fitness                                                               744.0
# FEvals                                                                 1611
# Time                                                               0.189778
# State                     [np.int32(1), np.int32(0), np.int32(0), np.int...
# schedule_type                                                    arithmetic
# schedule_init_temp                                                      0.1
# schedule_decay                                                       0.0001
# schedule_min_temp                                                     0.001
# schedule_current_value                                             0.099981
# Temperature               ArithDecay(init_temp=0.1, decay=0.0001, min_te...
# max_iters                                                              1024

# Iteration                                                              1024
# Fitness                                                               746.0
# FEvals                                                                 1615
# Time                                                               0.224633
# State                     [np.int32(1), np.int32(0), np.int32(0), np.int...
# schedule_type                                                   exponential
# schedule_init_temp                                                     0.75
# schedule_exp_const                                                    0.005
# schedule_min_temp                                                     0.001
# schedule_current_value                                             0.749158
# Temperature               ExpDecay(init_temp=0.75, exp_const=0.005, min_...
# max_iters                                                              1024

# Iteration                                                              1024
# Fitness                                                               750.0
# FEvals                                                                 1646
# Time                                                               0.224393
# State                     [np.int32(1), np.int32(0), np.int32(0), np.int...
# schedule_type                                                     geometric
# schedule_init_temp                                                      1.0
# schedule_decay                                                         0.99
# schedule_min_temp                                                     0.001
# schedule_current_value                                             0.997747
# Temperature                                                             1.0
# max_iters                                                              1024

############ SA Best Model ############

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=1000, fitness_fn=fitness, maximize=True, max_val=2)

# Create an SA Runner instance to solve the Flip Flop problem
sa = SARunner(
    problem=problem,
    experiment_name="flipflop_sa",
    seed=1,
    output_directory=None,  # Specify output directory if needed
    temperature_list=[0.75],
    decay_list=[mlrose.GeomDecay],
    iteration_list=2 ** np.arange(13),  # Iterations from 2^0 to 2^12
)

# Run the SA Runner and retrieve its results
df_run_stats, df_run_curves = sa.run()

# Get the best fitness score (max for Flip Flop problem)
best_fitness = df_run_stats["Fitness"].max()

# Print the best fitness score
print(f"Best Fitness Score: {best_fitness}")

# Keep only relevant columns
df_filtered = df_run_curves[["Iteration", "Time", "Fitness", "FEvals", "max_iters"]].copy()

# Add a new column 'Algorithm' with the value 'SA'
df_filtered["Algorithm"] = "SA"

# Save the filtered results to a CSV file
df_filtered.to_csv('flipflop_sa_best_filtered_results.csv', index=False)
print("Filtered results saved to flipflop_sa_best_filtered_results.csv")

########## SA Temperature ############
# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=1000, fitness_fn=fitness, maximize=True, max_val=2)

# Create the temperature and decay schedule
temperature_list = [0.1, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0, 10000000.0]
decay_list = [GeomDecay(init_temp=t) for t in temperature_list]

# Define a delta convergence threshold
convergence_threshold = 1e-6  # Example threshold

# Define the parameters for the Simulated Annealing algorithm
sa_runner = SARunner(
    problem=problem,
    experiment_name="FlipFlop_SA",
    seed=1,
    iteration_list=2**np.arange(11),  # List of iterations
    temperature_list=temperature_list,  # Temperature values
    decay_list=[GeomDecay],  # Use the GeomDecay schedule
    max_attempts=1000
)

# Run the experiment and capture results
df_run_stats, df_run_curves = sa_runner.run()

# Save results to CSV
df_run_curves.to_csv('flipflop_sa_results.csv', index=False)
print("Results saved to flipflop_sa_results.csv")

# Load the CSV file into a DataFrame
file_path = 'flipflop_sa_results.csv'
df = pd.read_csv(file_path)

# Group by temperature and get the maximum fitness for each temperature
df_max_fitness = df.groupby('Temperature')['Fitness'].max().reset_index()

# Print the result
print(df_max_fitness)