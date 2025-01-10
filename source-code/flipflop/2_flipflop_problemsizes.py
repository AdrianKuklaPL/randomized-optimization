import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import GARunner

########## GA ##########

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# List of problem sizes to test
problem_sizes = [10, 50, 100, 200, 400, 600, 800, 1000]

# Initialize a list to store results
results_list = []

# Loop over the different problem sizes
for size in problem_sizes:
    print(f"Running GA for problem size: {size}")
    
    # Define the optimization problem object for the current problem size
    problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)
    
    # Create a GA Runner instance with the best hyperparameters
    ga = GARunner(
        problem=problem,
        experiment_name=f"flipflop_ga_size_{size}",
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
    
    # Find the run with the minimum number of evaluations (FEvals)
    best_run = df_run_curves[df_run_curves["Fitness"] == best_fitness].iloc[0]
    
    # Store the best fitness, FEvals, and time for the current problem size
    results_list.append({
        "Problem Size": size,
        "Fitness": best_fitness,
        "FEvals": best_run["FEvals"],
        "Time": best_run["Time"]
    })

# Convert the results list into a DataFrame
df_results = pd.DataFrame(results_list)

# Save the results to a CSV file
df_results.to_csv('flipflop_ga_problem_sizes_results.csv', index=False)
print("Results saved to flipflop_ga_problem_sizes_results.csv")

########## SA ##########
from mlrose_ky.runners import SARunner

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# List of problem sizes to test
problem_sizes = [10, 50, 100, 200, 400, 600, 800, 1000]

# Initialize a list to store results
results_list = []

# Loop over the different problem sizes
for size in problem_sizes:
    print(f"Running SA for problem size: {size}")
    
    # Define the optimization problem object for the current problem size
    problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)
    
    # Create an SA Runner instance with the best hyperparameters
    sa = SARunner(
        problem=problem,
        experiment_name=f"flipflop_sa_size_{size}",
        seed=1,
        output_directory=None,  # Specify output directory if needed
        temperature_list=[0.75],
        decay_list=[mlrose.GeomDecay],
        iteration_list=2**np.arange(13),  # Iterations from 2^0 to 2^12
    )
    
    # Run the SA Runner and retrieve results
    df_run_stats, df_run_curves = sa.run()
    
    # Find the best fitness score (max for Flip Flop problem)
    best_fitness = df_run_curves["Fitness"].max()
    
    # Find the run with the minimum number of evaluations (FEvals)
    best_run = df_run_curves[df_run_curves["Fitness"] == best_fitness].iloc[0]
    
    # Store the best fitness, FEvals, and time for the current problem size
    results_list.append({
        "Problem Size": size,
        "Fitness": best_fitness,
        "FEvals": best_run["FEvals"],
        "Time": best_run["Time"]
    })

# Convert the results list into a DataFrame
df_results = pd.DataFrame(results_list)

# Save the results to a CSV file
df_results.to_csv('flipflop_sa_problem_sizes_results.csv', index=False)
print("Results saved to flipflop_sa_problem_sizes_results.csv")

########## RHC ##########
from mlrose_ky.runners import RHCRunner

# Define the Flip Flop fitness function
fitness = mlrose.FlipFlop()

# List of problem sizes to test
problem_sizes = [10, 50, 100, 200, 400, 600, 800, 1000]

# Initialize a list to store results
results_list = []

# Loop over the different problem sizes
for size in problem_sizes:
    print(f"Running RHC for problem size: {size}")
    
    # Define the optimization problem object for the current problem size
    problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)
    
    # Create an RHC Runner instance with 30 restarts for the current problem size
    rhc_runner = RHCRunner(
        problem=problem,
        experiment_name=f"flipflop_rhc_size_{size}",
        output_directory=None,  # Specify output directory if needed
        seed=1,
        iteration_list=2**np.arange(13),  # Iterations from 2^0 to 2^12
        restart_list=[30],  # Using 30 restarts
        max_attempts=50  # Maximum attempts without improvement
    )
    
    # Run the RHC Runner and retrieve results
    df_run_stats, df_run_curves = rhc_runner.run()
    
    # Find the best fitness score (max for Flip Flop problem)
    best_fitness = df_run_curves["Fitness"].max()
    
    # Find the run with the minimum number of evaluations (FEvals)
    best_run = df_run_curves[df_run_curves["Fitness"] == best_fitness].iloc[0]
    
    # Store the best fitness, FEVals, and time for the current problem size
    results_list.append({
        "Problem Size": size,
        "Fitness": best_fitness,
        "FEvals": best_run["FEvals"],
        "Time": best_run["Time"]
    })

# Convert the results list into a DataFrame
df_results = pd.DataFrame(results_list)

# Save the results to a CSV file
df_results.to_csv('flipflop_rhc_problem_sizes_results.csv', index=False)
print("Results saved to flipflop_rhc_problem_sizes_results.csv")

