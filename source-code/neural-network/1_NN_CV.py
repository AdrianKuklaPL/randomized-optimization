import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from pyperch.neural.sa_nn import SAModule  
from pyperch.neural.ga_nn import GAModule  
from imblearn.over_sampling import SMOTE
from skorch.callbacks import EpochScoring
from sklearn.metrics import f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

############## Simulated Annealing Cross Validation ##############

# Set the seed for reproducibility
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

# Fetch dataset - drug consumption (quantified)
from ucimlrepo import fetch_ucirepo
drug_consumption_quantified = fetch_ucirepo(id=373)

# Data (as pandas dataframes)
X = drug_consumption_quantified.data.features 
y = drug_consumption_quantified.data.targets

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a binary classification problem (users vs. non-users for cannabis)
class_mapping = {
    'CL0': 'Non-user', 'CL1': 'Non-user',
    'CL2': 'User', 'CL3': 'User', 'CL4': 'User', 'CL5': 'User', 'CL6': 'User'
}

y['cannabis'] = y['cannabis'].map(class_mapping)
Y = np.where(y['cannabis'] == 'Non-user', 0, 1)

# Convert to float32 and int64
X_scaled = X_scaled.astype(np.float32)
Y = Y.astype(np.int64)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply SMOTE
smote = SMOTE(random_state=seed)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Define the neural network
net = NeuralNetClassifier(
    module=SAModule,
    module__layer_sizes=(12, 32, 16, 2),
    module__dropout_percent=0.1,
    module__activation=nn.Tanh(),
    module__output_activation=nn.Softmax(dim=-1),
    module__t_min=0.001,
    module__t=10000,
    module__cooling=0.95,
    max_epochs=100,
    verbose=1,
    criterion=nn.CrossEntropyLoss(),
    lr=0.001,
    batch_size=256,
    callbacks=[
        EpochScoring(scoring='f1_weighted', name='train_f1_weighted', on_train=True),
        EpochScoring(scoring='f1_weighted', name='valid_f1_weighted'),
    ],
    iterator_train__shuffle=True
)

# Set up GridSearchCV
net.set_params(train_split=False, verbose=0)

default_params = {
    'module__layer_sizes': [(12, 32, 16, 2)],
    'max_epochs': [50],
    'module__t_min': [0.001],
    'module__dropout_percent': [0.1],
    'lr': [0.001],
    'batch_size': [256],
}

grid_search_params = {
    'module__t': [5000, 10000, 20000, 30000, 40000],
    'module__cooling': [0.2, 0.4, 0.6, 0.8, 0.95],
    **default_params,
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
gs = GridSearchCV(net, grid_search_params, n_jobs=-1, refit=True, cv=cv, scoring='f1_weighted', verbose=2)

# Fit the grid search
gs.fit(X_train_resampled, Y_train_resampled)

print("Best score: {:.3f}, Best params: {}".format(gs.best_score_, gs.best_params_))

# Create a data table of grid search results
results = gs.cv_results_
params = results['params']
scores = results['mean_test_score']

# Extract cooling rates and temperatures
cooling_rates = [p['module__cooling'] for p in params]
temperatures = [p['module__t'] for p in params]

# Create a DataFrame
df_results = pd.DataFrame({
    'Cooling Rate': cooling_rates,
    'Temperature': temperatures,
    'F1 Score': scores
})

# Pivot the DataFrame to create a 2D table
df_pivot = df_results.pivot(index='Cooling Rate', columns='Temperature', values='F1 Score')

# Sort the index and columns for better readability
df_pivot = df_pivot.sort_index(ascending=False)
df_pivot = df_pivot.sort_index(axis=1)

print("\nGrid Search Results:")
print(df_pivot.to_string())

# Create a heatmap of the results
plt.figure(figsize=(12, 8))
plt.imshow(df_pivot, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='F1 Score')
plt.title('F1 Scores for Different Cooling Rates and Temperatures')
plt.xlabel('Temperature')
plt.ylabel('Cooling Rate')
plt.xticks(range(len(df_pivot.columns)), df_pivot.columns, rotation=45)
plt.yticks(range(len(df_pivot.index)), df_pivot.index)

# Add text annotations to the heatmap
for i in range(len(df_pivot.index)):
    for j in range(len(df_pivot.columns)):
        plt.text(j, i, f'{df_pivot.iloc[i, j]:.3f}', 
                 ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# Use the best estimator for predictions
best_net = gs.best_estimator_

# Predict probabilities on the test set
Y_pred_proba = best_net.predict_proba(X_test)
Y_pred = best_net.predict(X_test)

# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("\nFinal Model Evaluation:")
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Best score: 0.710, Best params: {'batch_size': 256, 'lr': 0.001, 'max_epochs': 50, 'module__cooling': 0.8,
# 'module__dropout_percent': 0.1, 'module__layer_sizes': (12, 32, 16, 2), 'module__t': 30000, 'module__t_min': 0.001}
# However, consistently the 30k population has outperformed other populations and
# the 0.4 cooling rate has outperformed other cooling rates.

# Grid Search Results:
# Temperature      5000      10000     20000     30000     40000
# Cooling Rate
# 0.95          0.627908  0.457091  0.495260  0.576559  0.400120
# 0.80          0.604231  0.600031  0.542992  0.709730  0.657211
# 0.60          0.585423  0.637367  0.629517  0.666095  0.624673
# 0.40          0.655053  0.661064  0.694009  0.708335  0.702874
# 0.20          0.652259  0.648480  0.656064  0.634920  0.517608


############## Genetic Algorithm Cross validation ##############

# Set the seed for reproducibility
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

# Fetch dataset - drug consumption (quantified)
from ucimlrepo import fetch_ucirepo
drug_consumption_quantified = fetch_ucirepo(id=373)

# Data (as pandas dataframes)
X = drug_consumption_quantified.data.features 
y = drug_consumption_quantified.data.targets

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a binary classification problem (users vs. non-users for cannabis)
class_mapping = {
    'CL0': 'Non-user', 'CL1': 'Non-user',
    'CL2': 'User', 'CL3': 'User', 'CL4': 'User', 'CL5': 'User', 'CL6': 'User'
}

y['cannabis'] = y['cannabis'].map(class_mapping)
Y = np.where(y['cannabis'] == 'Non-user', 0, 1)

# Convert to float32 and int64
X_scaled = X_scaled.astype(np.float32)
Y = Y.astype(np.int64)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

# Apply SMOTE
smote = SMOTE(random_state=seed)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Define the neural network
net = NeuralNetClassifier(
    module=GAModule,
    module__layer_sizes=(12, 32, 16, 2),
    module__dropout_percent=0.1,
    module__activation=nn.Tanh(),
    module__output_activation=nn.Softmax(dim=-1),
    module__population_size=100,
    module__to_mate=90,
    module__to_mutate=1,
    max_epochs=30,  # Reduced for speed
    verbose=1,
    criterion=nn.CrossEntropyLoss(),
    lr=0.001,
    batch_size=256,
    callbacks=[
        EpochScoring(scoring='f1_weighted', name='train_f1_weighted', on_train=True),
        EpochScoring(scoring='f1_weighted', name='valid_f1_weighted'),
    ],
    iterator_train__shuffle=True
)

# Set up GridSearchCV
net.set_params(train_split=False, verbose=0)

default_params = {
    'max_epochs': [30],
    'module__layer_sizes': [(12, 32, 16, 2)],
    'module__dropout_percent': [0.1],
    'lr': [0.001],
    'batch_size': [256],
    'module__to_mutate': [1],
    'module__population_size': [100],
    'module__to_mate': [90],
}

grid_search_params = {
    'module__to_mutate': [1,3,5],
    **default_params,
}

gs = GridSearchCV(net, grid_search_params, n_jobs=-1, refit=True, cv=3, scoring='f1_weighted', verbose=2)

# Fit the grid search
gs.fit(X_train_resampled, Y_train_resampled)

print("Best score: {:.3f}, Best params: {}".format(gs.best_score_, gs.best_params_))

# Use the best estimator for predictions
best_net = gs.best_estimator_

# Predict probabilities on the test set
Y_pred_proba = best_net.predict_proba(X_test)
Y_pred = best_net.predict(X_test)

# Extract results
results = pd.DataFrame(gs.cv_results_)

# Create a DataFrame focusing on the parameters we're interested in
focused_results = pd.DataFrame({
    'module__to_mutate': results['param_module__to_mutate'],
    'Mean F1 Score': results['mean_test_score'],
    'Std F1 Score': results['std_test_score']
})

# Sort by Mean F1 Score in descending order
focused_results = focused_results.sort_values('Mean F1 Score', ascending=False)

# Display the results
print("\nResults for each module__to_mutate value:")
print(focused_results.to_string(index=False))


# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))


# Best score: 0.650, Best params: {'batch_size': 256, 'lr': 0.001, 'max_epochs': 30, 'module__dropout_percent': 0.1,
#  'module__layer_sizes': (12, 32, 16, 2), 'module__population_size': 100, 'module__to_mate': 90, 'module__to_mutate': 1}
# Weighted F1 Score: 0.6037540645984646
# Accuracy: 0.5915119363395226
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.42      0.65      0.51       124
#            1       0.77      0.56      0.65       253

#     accuracy                           0.59       377
#    macro avg       0.59      0.61      0.58       377
# weighted avg       0.65      0.59      0.60       377

# Best score: 0.620, Best params: {'batch_size': 256, 'lr': 0.001, 'max_epochs': 30, 'module__dropout_percent': 0.1, 'module__layer_sizes': (12, 32, 16, 2), 'module__population_size': 100, 'module__to_mate': 90, 'module__to_mutate': 1}

# Grid Search Results:
#                                                                                                        Parameters  Mean F1 Score  Std F1 Score  Rank
# batch_size: 256, lr: 0.001, max_epochs: 30, dropout_percent: 0.1, population_size: 100, to_mate: 90, to_mutate: 1       0.620286      0.032318     1
# Weighted F1 Score: 0.6037540645984646
# Accuracy: 0.5915119363395226
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.42      0.65      0.51       124
#            1       0.77      0.56      0.65       253

#     accuracy                           0.59       377
#    macro avg       0.59      0.61      0.58       377
# weighted avg       0.65      0.59      0.60       377