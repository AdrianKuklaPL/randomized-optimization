import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import MIMICRunner

########### Cross validation MIMIC ##########

# Generate random weights and values for 100 items
np.random.seed(1)  # Set seed for reproducibility
weights = np.random.randint(1, 100, size=100)  # Random weights between 1 and 100
values = np.random.randint(1, 50, size=100)  # Random values between 1 and 50
max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

# Define the Knapsack fitness function with 100 items
fitness = mlrose.Knapsack(weights, values, max_weight_pct)

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

# Create a MIMIC Runner instance to solve the Knapsack problem
mimic = MIMICRunner(
    problem=problem,
    experiment_name="knapsack_mimic_100_items",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(11),  # Iterations from 2^0 to 2^10
    population_sizes=[30, 70, 100, 150, 200, 300, 500],  # Population sizes to test
    keep_percent_list=[0.1, 0.2, 0.4, 0.6, 0.8],  # Keep percentage to test
    use_fast_mimic=True  # Use the fast MIMIC algorithm
)

# Run the MIMIC Runner and retrieve results
df_run_stats, df_run_curves = mimic.run()

# Initialize an empty list to store final results from each combination
final_results = []

# Iterate through each population size and keep percent combination
for (pop_size, keep_pct), group in df_run_curves.groupby(['Population Size', 'Keep Percent']):
    # Get the row corresponding to the last iteration of this run
    final_iteration = group[group['Iteration'] == group['Iteration'].max()]
    
    # Extract necessary values and append them to the final results
    final_results.append({
        'Fitness': final_iteration['Fitness'].values[0],
        'FEvals': final_iteration['FEvals'].values[0],
        'Iteration': final_iteration['Iteration'].values[0],
        'Time': final_iteration['Time'].values[0],
        'Population Size': pop_size,
        'Keep Percent': keep_pct
    })

# Convert the final results list into a DataFrame
df_final_runs = pd.DataFrame(final_results)

# Sort the DataFrame by 'Fitness' in descending order and extract the top 5 rows
top_5_fitness = df_final_runs.sort_values(by='Fitness', ascending=False).head(5)

# Display the top 5 rows
print(top_5_fitness)

# Print the DataFrame with the final run details
print(df_final_runs)

# Optionally, save the final DataFrame to a CSV file
df_final_runs.to_csv('knapsack_mimic_final_runs.csv', index=False)
print("Final run results saved to knapsack_mimic_final_runs.csv")

########## MIMIC Best Model ##########
# Generate random weights and values for 1000 items
np.random.seed(1)  # Set seed for reproducibility
weights = np.random.randint(1, 100, size=100)  # Random weights between 1 and 100
values = np.random.randint(1, 50, size=100)  # Random values between 1 and 50
max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

# Define the Knapsack fitness function with 1000 items
fitness = mlrose.Knapsack(weights, values, max_weight_pct)

# Define the optimization problem object
problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

# Create a MIMIC Runner instance with population size 500 and keep percent 0.2
mimic = MIMICRunner(
    problem=problem,
    experiment_name="knapsack_mimic_best",
    output_directory=None,  # Specify output directory if needed
    seed=1,
    iteration_list=2**np.arange(13),  # Iterations from 2^0 to 2^12
    population_sizes=[500],  # Population size
    keep_percent_list=[0.2],  # Keep percent
    use_fast_mimic=True  # Enable fast MIMIC if desired
)

# Run the MIMIC Runner and retrieve results
df_run_stats, df_run_curves = mimic.run()

# Find the best fitness score (max for Knapsack problem)
best_fitness = df_run_curves["Fitness"].max()

# Print the best fitness score
print(f"Best Fitness Score: {best_fitness}")

# Keep only relevant columns
df_filtered = df_run_curves[["Iteration", "Time", "Fitness", "FEvals", "max_iters"]].copy()

# Add a new column 'Algorithm' with the value 'MIMIC'
df_filtered["Algorithm"] = "MIMIC"

# Save the filtered results to a CSV file
df_filtered.to_csv('knapsack_mimic_best_filtered_results.csv', index=False)
print("Filtered results saved to knapsack_mimic_best_filtered_results.csv")
