import pandas as pd
import matplotlib.pyplot as plt

######### Fitness vs Iteration #########

# Load the CSV files into DataFrames
file_path_ga = 'flipflop_ga_best_filtered_results.csv'
file_path_csv1 = 'flipflop_sa_best_filtered_results.csv'
file_path_csv2 = 'flipflop_rhc_best_filtered_results.csv'

df_ga = pd.read_csv(file_path_ga)
df_csv1 = pd.read_csv(file_path_csv1)
df_csv2 = pd.read_csv(file_path_csv2)

# Plot Fitness vs Iteration for GA
plt.figure(figsize=(10, 6))
plt.plot(df_ga['Iteration'], df_ga['Fitness'], label='GA', marker='', color='blue')

# Plot Fitness vs Iteration for SA
plt.plot(df_csv1['Iteration'], df_csv1['Fitness'], label='SA', marker='', color='green')

# Plot Fitness vs Iteration for RHC
plt.plot(df_csv2['Iteration'], df_csv2['Fitness'], label='RHC', marker='', color='red')

# Add labels, title, and legend
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Fitness vs Iteration for GA, SA, and RHC on Flip Flop')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

############# FEvals vs Iteration #############

# Plot Fitness vs Iteration for GA
plt.figure(figsize=(10, 6))
plt.plot(df_ga['Iteration'], df_ga['FEvals'], label='GA', marker='', color='blue')

# Plot Fitness vs Iteration for SA
plt.plot(df_csv1['Iteration'], df_csv1['FEvals'], label='SA', marker='', color='green')

# Plot Fitness vs Iteration for RHC
plt.plot(df_csv2['Iteration'], df_csv2['FEvals'], label='RHC', marker='', color='red')

# Add labels, title, and legend
plt.xlabel('Iteration')
plt.ylabel('FEvals')
plt.title('FEvals vs Iteration for GA, SA, and RHC on Flip Flop')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

############# FEvals vs Time #############
# Plot Fitness vs Iteration for GA
plt.figure(figsize=(10, 6))
plt.plot(df_ga['Time'], df_ga['FEvals'], label='GA', marker='', color='blue')

# Plot Fitness vs Iteration for SA
plt.plot(df_csv1['Time'], df_csv1['FEvals'], label='SA', marker='', color='green')

# Plot Fitness vs Iteration for RHC
plt.plot(df_csv2['Time'], df_csv2['FEvals'], label='RHC', marker='', color='red')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('FEvals')
plt.title('FEvals vs Time for GA, SA, and RHC on Flip Flop')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Load the CSV files into DataFrames
file_path_ga = 'flipflop_ga_problem_sizes_results.csv'
file_path_sa = 'flipflop_sa_problem_sizes_results.csv'
file_path_rhc = 'flipflop_rhc_problem_sizes_results.csv'

df_ga = pd.read_csv(file_path_ga)
df_sa = pd.read_csv(file_path_sa)
df_rhc = pd.read_csv(file_path_rhc)

######### Fitness vs Problem Size #########
plt.figure(figsize=(10, 6))

# Plot Fitness vs Problem Size for GA
plt.plot(df_ga['Problem Size'], df_ga['Fitness'], label='GA', marker='o', color='blue')

# Plot Fitness vs Problem Size for SA
plt.plot(df_sa['Problem Size'], df_sa['Fitness'], label='SA', marker='o', color='green')

# Plot Fitness vs Problem Size for RHC
plt.plot(df_rhc['Problem Size'], df_rhc['Fitness'], label='RHC', marker='o', color='red')

# Add labels, title, and legend
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.title('Fitness vs Problem Size for GA, SA, and RHC on Flip Flop')
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

# Add labels, title, and legend
plt.xlabel('Problem Size')
plt.ylabel('FEvals')
plt.title('FEvals vs Problem Size for GA, SA, and RHC on Flip Flop')
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

# Add labels, title, and legend
plt.xlabel('Problem Size')
plt.ylabel('Time')
plt.title('Run Time vs Problem Size for GA, SA, and RHC on Flip Flop')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
