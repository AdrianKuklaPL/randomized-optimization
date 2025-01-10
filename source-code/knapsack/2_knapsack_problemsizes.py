import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from mlrose_ky.runners import GARunner, SARunner, RHCRunner, MIMICRunner

# List of problem sizes to test
problem_sizes = [50, 100, 200, 400]

########## GA ##########

# Initialize a list to store results
results_list = []

# Loop over the different problem sizes
for size in problem_sizes:
    print(f"Running GA for problem size: {size}")
    
    # Generate random weights and values for the knapsack problem
    np.random.seed(1)  # Set seed for reproducibility
    weights = np.random.randint(1, 100, size=size)  # Random weights for the current problem size
    values = np.random.randint(1, 50, size=size)    # Random values for the current problem size
    max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

    # Define the Knapsack fitness function
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    # Define the optimization problem object for the current problem size
    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)
    
    # Create a GA Runner instance with the best hyperparameters
    ga = GARunner(
        problem=problem,
        experiment_name=f"knapsack_ga_size_{size}",
        output_directory=None,  # Specify output directory if needed
        seed=1,
        iteration_list=2**np.arange(11),  # Iterations from 2^0 to 2^12
        population_sizes=[100],  # Best population size
        mutation_rates=[0.8],  # Best mutation rate
    )
    
    # Run the GA Runner and retrieve results
    df_run_stats, df_run_curves = ga.run()
    
    # Find the best fitness score (max for Knapsack problem)
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
df_results.to_csv('knapsack_ga_problem_sizes_results.csv', index=False)
print("Results saved to knapsack_ga_problem_sizes_results.csv")

########## SA ##########

# Initialize a list to store results
results_list = []

# Loop over the different problem sizes
for size in problem_sizes:
    print(f"Running SA for problem size: {size}")
    
    # Generate random weights and values for the knapsack problem
    np.random.seed(1)  # Set seed for reproducibility
    weights = np.random.randint(1, 100, size=size)  # Random weights for the current problem size
    values = np.random.randint(1, 50, size=size)    # Random values for the current problem size
    max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

    # Define the Knapsack fitness function
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    # Define the optimization problem object for the current problem size
    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)
    
    # Create an SA Runner instance with the best hyperparameters
    sa = SARunner(
        problem=problem,
        experiment_name=f"knapsack_sa_size_{size}",
        seed=1,
        output_directory=None,  # Specify output directory if needed
        temperature_list=[10],
        decay_list=[mlrose.ExpDecay],
        iteration_list=2**np.arange(11),  # Iterations from 2^0 to 2^12
    )
    
    # Run the SA Runner and retrieve results
    df_run_stats, df_run_curves = sa.run()
    
    # Find the best fitness score (max for Knapsack problem)
    best_fitness = df_run_curves["Fitness"].max()
    
    # Find the run with the minimum number of evaluations (FEVals)
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
df_results.to_csv('knapsack_sa_problem_sizes_results.csv', index=False)
print("Results saved to knapsack_sa_problem_sizes_results.csv")

########## RHC ##########

# Initialize a list to store results
results_list = []

# Loop over the different problem sizes
for size in problem_sizes:
    print(f"Running RHC for problem size: {size}")
    
    # Generate random weights and values for the knapsack problem
    np.random.seed(1)  # Set seed for reproducibility
    weights = np.random.randint(1, 100, size=size)  # Random weights for the current problem size
    values = np.random.randint(1, 50, size=size)    # Random values for the current problem size
    max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

    # Define the Knapsack fitness function
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    # Define the optimization problem object for the current problem size
    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)
    
    # Create an RHC Runner instance with 30 restarts for the current problem size
    rhc_runner = RHCRunner(
        problem=problem,
        experiment_name=f"knapsack_rhc_size_{size}",
        output_directory=None,  # Specify output directory if needed
        seed=1,
        iteration_list=2**np.arange(11),  # Iterations from 2^0 to 2^12
        restart_list=[10],
        max_attempts=50  # Maximum attempts without improvement
    )
    
    # Run the RHC Runner and retrieve results
    df_run_stats, df_run_curves = rhc_runner.run()
    
    # Find the best fitness score (max for Knapsack problem)
    best_fitness = df_run_curves["Fitness"].max()
    
    # Find the run with the minimum number of evaluations (FEVals)
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
df_results.to_csv('knapsack_rhc_problem_sizes_results.csv', index=False)
print("Results saved to knapsack_rhc_problem_sizes_results.csv")

########## MIMIC ##########

# Initialize a list to store results
results_list = []

# Loop over the different problem sizes
for size in problem_sizes:
    print(f"Running MIMIC for problem size: {size}")
    
    # Generate random weights and values for the knapsack problem
    np.random.seed(1)  # Set seed for reproducibility
    weights = np.random.randint(1, 100, size=size)  # Random weights for the current problem size
    values = np.random.randint(1, 50, size=size)    # Random values for the current problem size
    max_weight_pct = 0.6  # Maximum weight percentage allowed in the knapsack

    # Define the Knapsack fitness function
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    # Define the optimization problem object for the current problem size
    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)
    
    # Create a MIMIC Runner instance with the specified hyperparameters
    mimic = MIMICRunner(
        problem=problem,
        experiment_name=f"knapsack_mimic_size_{size}",
        output_directory=None,  # Specify output directory if needed
        seed=1,
        iteration_list=2**np.arange(11),  # Iterations from 2^0 to 2^10
        population_sizes=[500],  # Population size
        keep_percent_list=[0.2],  # Keep percent
        use_fast_mimic=True  # Use the fast MIMIC algorithm
    )
    
    # Run the MIMIC Runner and retrieve results
    df_run_stats, df_run_curves = mimic.run()
    
    # Find the best fitness score (max for Knapsack problem)
    best_fitness = df_run_curves["Fitness"].max()
    
    # Find the run with the minimum number of evaluations (FEVals)
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
df_results.to_csv('knapsack_mimic_problem_sizes_results.csv', index=False)
print("Results saved to knapsack_mimic_problem_sizes_results.csv")