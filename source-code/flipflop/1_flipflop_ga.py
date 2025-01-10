import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import GARunner

########## Cross validation ##########

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=1000, fitness_fn=fitness, maximize=True, max_val=2)

# Create a GA Runner instance to solve the Flip Flop problem
ga = GARunner(
    problem=problem,
    experiment_name="flipflop_ga",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(11),  # Iterations from 2^0 to 2^10
    population_sizes=[10, 20, 30, 50, 70, 100],  # Population sizes to test
    mutation_rates=[0.1, 0.2, 0.4, 0.6, 0.8],  # Mutation rates to test
)

# Run the GA Runner and retrieve results
df_run_stats, df_run_curves = ga.run()

# Initialize an empty list to store final results from each combination
final_results = []

# Iterate through each population size and mutation rate combination
for (pop_size, mut_rate), group in df_run_curves.groupby(['Population Size', 'Mutation Rate']):
    # Get the row corresponding to the last iteration of this run
    final_iteration = group[group['Iteration'] == group['Iteration'].max()]
    
    # Extract necessary values and append them to the final results
    final_results.append({
        'Fitness': final_iteration['Fitness'].values[0],
        'FEvals': final_iteration['FEvals'].values[0],
        'Iteration': final_iteration['Iteration'].values[0],
        'Time': final_iteration['Time'].values[0],
        'Population Size': pop_size,
        'Mutation Rate': mut_rate
    })

# Convert the final results list into a DataFrame
df_final_runs = pd.DataFrame(final_results)

# Print the DataFrame with the final run details
print(df_final_runs)

# Optionally, save the final DataFrame to a CSV file
df_final_runs.to_csv('flipflop_ga_final_runs.csv', index=False)
print("Final run results saved to flipflop_ga_final_runs.csv")

########### GA Best Model ############

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=1000, fitness_fn=fitness, maximize=True, max_val=2)

# Create a GA Runner instance with the best hyperparameters
ga = GARunner(
    problem=problem,
    experiment_name="flipflop_ga_best",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(13),  # Iterations from 2^0 to 2^10
    population_sizes=[50],  # Best population size
    mutation_rates=[0.8],  # Best mutation rate
)

# Run the GA Runner and retrieve results
df_run_stats, df_run_curves = ga.run()

# Find the best fitness score (max for Flip Flop problem)
best_fitness = df_run_curves["Fitness"].max()

# Print the best fitness score
print(f"Best Fitness Score: {best_fitness}")

# Keep only relevant columns
df_filtered = df_run_curves[["Iteration", "Time", "Fitness", "FEvals", "max_iters"]].copy()

# Add a new column 'Algorithm' with the value 'GA'
df_filtered["Algorithm"] = "GA"

# Save the filtered results to a CSV file
df_filtered.to_csv('flipflop_ga_best_filtered_results.csv', index=False)
print("Filtered results saved to flipflop_ga_best_filtered_results.csv")



