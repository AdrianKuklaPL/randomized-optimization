import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import RHCRunner

########### Knapsack Cross Validation ############

# Define the Knapsack fitness function
np.random.seed(1)
weights = np.random.randint(1, 100, size=100)  # Random weights for 100 items
values = np.random.randint(1, 50, size=100)  # Random values for 100 items
max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

fitness = mlrose.Knapsack(weights, values, max_weight_pct)

# Define the optimization problem object for the Knapsack problem
problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

# Create an RHC Runner instance to solve the Knapsack problem with 30 restarts
rhc_runner = RHCRunner(
    problem=problem,
    experiment_name="knapsack_rhc_30_restarts",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(11),  # Iterations from 2^0 to 2^10
    restart_list=[50],
    max_attempts=200  # Maximum attempts without improvement
)

# Run the RHC Runner and retrieve results
df_run_stats, df_run_curves = rhc_runner.run()

# Find the best fitness score (max for Knapsack problem)
best_fitness = df_run_curves["Fitness"].max()

# Get all runs with the best fitness value
best_runs = df_run_curves[df_run_curves["Fitness"] == best_fitness]

# Find the run with the minimum number of evaluations (FEVals)
minimum_evaluations = best_runs["FEvals"].min()
best_run = best_runs[best_runs["FEvals"] == minimum_evaluations]

# Print the fitness score and the number of evaluations
print(f"Fitness Score: {best_fitness}, Function Evaluations: {minimum_evaluations}")

# Keep only relevant columns
df_filtered = df_run_curves[["Iteration", "Time", "Fitness", "FEvals", "max_iters", "Restarts"]].copy()

# Add a new column 'Algorithm' with the value 'RHC'
df_filtered["Algorithm"] = "RHC"

# Save the filtered results to a CSV file
# df_filtered.to_csv('knapsack_rhc_best_filtered_results_combined.csv', index=False)
print("Filtered results saved to knapsack_rhc_best_filtered_results_combined.csv")

# Print the best run details
print(best_run.iloc[0])

# Iteration            69.000000
# Time                  0.011527
# Fitness            1276.000000
# FEvals               83.000000
# Restarts              0.000000
# max_iters          1024.000000
# current_restart       0.000000

# Iteration            40.00000
# Time                  0.00729
# Fitness            1494.00000
# FEvals              331.00000
# Restarts              5.00000
# max_iters          1024.00000
# current_restart       2.00000

# Iteration            75.000000
# Time                  0.042941
# Fitness            1594.000000
# FEvals              932.000000
# Restarts             10.000000
# max_iters          1024.000000
# current_restart       7.000000

# Iteration            75.000000
# Time                  0.031631
# Fitness            1594.000000
# FEvals              932.000000
# Restarts             20.000000
# max_iters          1024.000000
# current_restart       7.000000

# Iteration           140.00000
# Time                  0.02745
# Fitness            1596.00000
# FEvals             4017.00000
# Restarts             50.00000
# max_iters          1024.00000
# current_restart      34.00000

# Iteration             8.000000
# Time                  0.009527
# Fitness            1605.000000
# FEvals             8184.000000
# Restarts            100.000000
# max_iters          1024.000000
# current_restart      73.000000

########### Knapsack Best Model RHC ############

# Generate random weights and values for 1000 items
np.random.seed(1)  # Set seed for reproducibility
weights = np.random.randint(1, 100, size=100)  # Random weights between 1 and 100
values = np.random.randint(1, 50, size=100)  # Random values between 1 and 50
max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

# Define the Knapsack fitness function with 1000 items
fitness = mlrose.Knapsack(weights, values, max_weight_pct)

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

# Create an RHC Runner instance to solve the Knapsack problem with 30 restarts
rhc_runner = RHCRunner(
    problem=problem,
    experiment_name="knapsack_rhc_30_restarts",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(13),  # Iterations from 2^0 to 2^10
    restart_list=[10],
)

# Run the RHC Runner and retrieve results
df_run_stats, df_run_curves = rhc_runner.run()

# Find the best fitness score (max for Knapsack problem)
best_fitness = df_run_curves["Fitness"].max()

# Get all runs with the best fitness value
best_runs = df_run_curves[df_run_curves["Fitness"] == best_fitness]

# Find the run with the minimum number of evaluations (FEVals)
minimum_evaluations = best_runs["FEvals"].min()
best_run = best_runs[best_runs["FEvals"] == minimum_evaluations]

# Print the fitness score and the number of evaluations
print(f"Fitness Score: {best_fitness}, Function Evaluations: {minimum_evaluations}")

# Keep only relevant columns
df_filtered = df_run_curves[["Iteration", "Time", "Fitness", "FEvals", "max_iters","Restarts"]].copy()

# Add a new column 'Algorithm' with the value 'RHC'
df_filtered["Algorithm"] = "RHC"

# Save the filtered results to a CSV file
df_filtered.to_csv('knapsack_rhc_best_filtered_results_combined.csv', index=False)
print("Filtered results saved to knapsack_rhc_best_filtered_results_combined.csv")

# Print the best run details
print(best_run.iloc[0])

########## Extracting Best Results ##########

# Load the CSV file into a DataFrame
file_path = 'knapsack_rhc_best_filtered_results_combined.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Filter the DataFrame
filtered_row = df[(df['Fitness'] == 1605.0) & (df['FEvals'] == 2398.0)]
print(filtered_row)
if not filtered_row.empty:
    # Get the index of the matching row
    index = filtered_row.index[0]  # Get the first matching index
    
    # Ensure the index is large enough to extract  previous rows
    start_index = max(0, index - 175) 
    
    # Slice the DataFrame to get the previous  rows and the identified row
    df_subset = df.iloc[start_index:index+1] 
    
    # Save the subset to a new CSV file
    output_file = 'knapsack_rhc_best_filtered_results.csv'
    df_subset.to_csv(output_file, index=False)
    
    print(f"Filtered results saved to {output_file}")
else:
    print("No matching rows found.")

