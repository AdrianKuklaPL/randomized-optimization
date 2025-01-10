import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import RHCRunner

######### Cross validation ##########

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=20, fitness_fn=fitness, maximize=True, max_val=2)

# Create an RHC Runner instance to solve the Flip Flop problem
rhc_runner = RHCRunner(
    problem=problem,
    experiment_name="flipflop_rhc",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(7),  # Iterations from 2^0 to 2^10
    restart_list=[0, 5, 10, 20, 30],  # Restart values to test
    max_attempts=100  # Maximum attempts without improvement
)

# Run the RHC Runner and retrieve results
df_run_stats, df_run_curves = rhc_runner.run()

# Find the best fitness score (max for Flip Flop problem)
best_fitness = df_run_curves["Fitness"].max()
# Get all runs with the best fitness value
best_runs = df_run_curves[df_run_curves["Fitness"] == best_fitness]

# Find the run with the minimum number of evaluations (FEvals)
minimum_evaluations = best_runs["FEvals"].min()
best_run = best_runs[best_runs["FEvals"] == minimum_evaluations]

# Extract the best restart value
best_restart = best_run["Restarts"].iloc[0]

# Print the best restart value and fitness score
print(f"Best Restart Value: {best_restart}, Best Fitness: {best_fitness}")


########### RHC Best Model ############

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=1000, fitness_fn=fitness, maximize=True, max_val=2)

# Create an RHC Runner instance to solve the Flip Flop problem with 30 restarts
rhc_runner = RHCRunner(
    problem=problem,
    experiment_name="flipflop_rhc_30_restarts",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(13),  # Iterations from 2^0 to 2^10
    restart_list=[30],
    max_attempts=50  # Maximum attempts without improvement
)

# Run the RHC Runner and retrieve results
df_run_stats, df_run_curves = rhc_runner.run()

# Find the best fitness score (max for Flip Flop problem)
best_fitness = df_run_curves["Fitness"].max()

# Get all runs with the best fitness value
best_runs = df_run_curves[df_run_curves["Fitness"] == best_fitness]

# Print the fitness score
print(f"Fitness Score: {best_fitness}")

# Print the final row of the best_runs
print("Final row of best runs:")
print(best_runs.iloc[-1])

df_run_curves.to_csv('flipflop_rhc_best_filtered_results_combined.csv', index=False)
print("Results saved to csv")

# Final row of best runs:
# Iteration           677.000000
# Time                  0.698249
# Fitness             699.000000
# FEvals              781.000000
# Restarts              0.000000
# max_iters          4096.000000
# current_restart       0.000000

# Final row of best runs:
# Iteration          1197.000000
# Time                  0.802764
# Fitness             753.000000
# FEvals             5317.000000
# Restarts              5.000000
# max_iters          4096.000000
# current_restart       4.000000

# Final row of best runs:
# Iteration          1197.00000
# Time                  0.48215
# Fitness             753.00000
# FEvals             5317.00000
# Restarts             10.00000
# max_iters          4096.00000
# current_restart       4.00000

# Final row of best runs:
# Iteration           1328.000000
# Time                   0.317945
# Fitness              754.000000
# FEvals             12733.000000
# Restarts              20.000000
# max_iters           4096.000000
# current_restart       11.000000

# Iteration           1018.000000
# Time                   0.221759
# Fitness              756.000000
# FEvals             27897.000000
# Restarts              30.000000
# max_iters           1024.000000
# current_restart       27.000000

########## Extract results ##########

# Load the CSV file into a DataFrame
file_path = 'flipflop_rhc_best_filtered_results_combined.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Filter the DataFrame
filtered_row = df[(df['Fitness'] == 784.0) & (df['FEvals'] == 12683.0)]
print(filtered_row)
if not filtered_row.empty:
    # Get the index of the matching row
    index = filtered_row.index[0]  # Get the first matching index
    
    # Ensure the index is large enough to extract  previous rows
    start_index = max(0, index - 1277) 
    
    # Slice the DataFrame to get the previous  rows and the identified row
    df_subset = df.iloc[start_index:index+1] 
    
    # Save the subset to a new CSV file
    output_file = 'flipflop_rhc_best_filtered_results.csv'
    df_subset.to_csv(output_file, index=False)
    
    print(f"Filtered results saved to {output_file}")
else:
    print("No matching rows found.")