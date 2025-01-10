import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files into DataFrames
file_path_ga = 'knapsack_ga_best_filtered_results.csv'
file_path_sa = 'knapsack_sa_best_filtered_results.csv'
file_path_rhc = 'knapsack_rhc_best_filtered_results.csv'
file_path_mimic = 'knapsack_mimic_best_filtered_results.csv'

df_ga = pd.read_csv(file_path_ga)
df_sa = pd.read_csv(file_path_sa)
df_rhc = pd.read_csv(file_path_rhc)
df_mimic = pd.read_csv(file_path_mimic)

######### Fitness vs Iteration #########

# Plot Fitness vs Iteration for GA
plt.figure(figsize=(10, 6))
plt.plot(df_ga['Iteration'], df_ga['Fitness'], label='GA', marker='', color='blue')

# Plot Fitness vs Iteration for SA
plt.plot(df_sa['Iteration'], df_sa['Fitness'], label='SA', marker='', color='green')

# Plot Fitness vs Iteration for RHC
plt.plot(df_rhc['Iteration'], df_rhc['Fitness'], label='RHC', marker='', color='red')

# Plot Fitness vs Iteration for MIMIC
plt.plot(df_mimic['Iteration'], df_mimic['Fitness'], label='MIMIC', marker='', color='purple')

# Add labels, title, and legend
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Fitness vs Iteration for GA, SA, RHC, and MIMIC on Knapsack')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

############# FEvals vs Iteration #############

# Plot FEvals vs Iteration for GA
plt.figure(figsize=(10, 6))
plt.plot(df_ga['Iteration'], df_ga['FEvals'], label='GA', marker='', color='blue')

# Plot FEvals vs Iteration for SA
plt.plot(df_sa['Iteration'], df_sa['FEvals'], label='SA', marker='', color='green')

# Plot FEvals vs Iteration for RHC
plt.plot(df_rhc['Iteration'], df_rhc['FEvals'], label='RHC', marker='', color='red')

# Plot FEvals vs Iteration for MIMIC
plt.plot(df_mimic['Iteration'], df_mimic['FEvals'], label='MIMIC', marker='', color='purple')

# Add labels, title, and legend
plt.xlabel('Iteration')
plt.ylabel('FEvals')
plt.title('FEvals vs Iteration for GA, SA, RHC, and MIMIC on Knapsack')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

############# FEvals vs Time #############

# Plot FEvals vs Time for GA
plt.figure(figsize=(10, 6))
plt.plot(df_ga['Time'], df_ga['FEvals'], label='GA', marker='', color='blue')

# Plot FEvals vs Time for SA
plt.plot(df_sa['Time'], df_sa['FEvals'], label='SA', marker='', color='green')

# Plot FEvals vs Time for RHC
plt.plot(df_rhc['Time'], df_rhc['FEvals'], label='RHC', marker='', color='red')

# Plot FEvals vs Time for MIMIC
plt.plot(df_mimic['Time'], df_mimic['FEvals'], label='MIMIC', marker='', color='purple')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('FEvals')
plt.title('FEvals vs Time for GA, SA, RHC, and MIMIC on Knapsack')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Load the CSV files into DataFrames
file_path_ga = 'knapsack_ga_problem_sizes_results.csv'
file_path_sa = 'knapsack_sa_problem_sizes_results.csv'
file_path_rhc = 'knapsack_rhc_problem_sizes_results.csv'
file_path_mimic = 'knapsack_mimic_problem_sizes_results.csv'

df_ga = pd.read_csv(file_path_ga)
df_sa = pd.read_csv(file_path_sa)
df_rhc = pd.read_csv(file_path_rhc)
df_mimic = pd.read_csv(file_path_mimic)  # Load MIMIC data

######### Fitness vs Problem Size #########
plt.figure(figsize=(10, 6))

# Plot Fitness vs Problem Size for GA
plt.plot(df_ga['Problem Size'], df_ga['Fitness'], label='GA', marker='o', color='blue')

# Plot Fitness vs Problem Size for SA
plt.plot(df_sa['Problem Size'], df_sa['Fitness'], label='SA', marker='o', color='green')

# Plot Fitness vs Problem Size for RHC
plt.plot(df_rhc['Problem Size'], df_rhc['Fitness'], label='RHC', marker='o', color='red')

# Plot Fitness vs Problem Size for MIMIC
plt.plot(df_mimic['Problem Size'], df_mimic['Fitness'], label='MIMIC', marker='o', color='purple')

# Add labels, title, and legend
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.title('Fitness vs Problem Size for GA, SA, RHC, and MIMIC on Knapsack')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

######### FEvals vs Problem Size #########
plt.figure(figsize=(10, 6))

# Plot FEvals vs Problem Size for GA
plt.plot(df_ga['Problem Size'], df_ga['FEvals'], label='GA', marker='o', color='blue')

# Plot FEvals vs Problem Size for SA
plt.plot(df_sa['Problem Size'], df_sa['FEvals'], label='SA', marker='o', color='green')

# Plot FEvals vs Problem Size for RHC
plt.plot(df_rhc['Problem Size'], df_rhc['FEvals'], label='RHC', marker='o', color='red')

# Plot FEvals vs Problem Size for MIMIC
plt.plot(df_mimic['Problem Size'], df_mimic['FEvals'], label='MIMIC', marker='o', color='purple')

# Add labels, title, and legend
plt.xlabel('Problem Size')
plt.ylabel('FEvals')
plt.title('FEvals vs Problem Size for GA, SA, RHC, and MIMIC on Knapsack')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

######### Time vs Problem Size #########
plt.figure(figsize=(10, 6))

# Plot Time vs Problem Size for GA
plt.plot(df_ga['Problem Size'], df_ga['Time'], label='GA', marker='o', color='blue')

# Plot Time vs Problem Size for SA
plt.plot(df_sa['Problem Size'], df_sa['Time'], label='SA', marker='o', color='green')

# Plot Time vs Problem Size for RHC
plt.plot(df_rhc['Problem Size'], df_rhc['Time'], label='RHC', marker='o', color='red')

# Plot Time vs Problem Size for MIMIC
plt.plot(df_mimic['Problem Size'], df_mimic['Time'], label='MIMIC', marker='o', color='purple')

# Add labels, title, and legend
plt.xlabel('Problem Size')
plt.ylabel('Time')
plt.title('Run Time vs Problem Size for GA, SA, RHC, and MIMIC on Knapsack')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()