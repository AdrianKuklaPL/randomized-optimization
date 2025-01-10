import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from pyperch.neural.backprop_nn import BackpropModule
from pyperch.neural.rhc_nn import RHCModule
from pyperch.neural.sa_nn import SAModule
from pyperch.neural.ga_nn import GAModule  
from imblearn.over_sampling import SMOTE
from skorch.callbacks import EpochScoring, EarlyStopping
from sklearn.metrics import f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
from skorch.dataset import ValidSplit
import time


########### Backpropagation Neural Network ############

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
    module=BackpropModule,
    module__layer_sizes=(12, 32, 16, 2),
    module__dropout_percent=0.1,
    module__activation=nn.Tanh(),
    module__output_activation=nn.Softmax(dim=-1),
    max_epochs=300,
    verbose=1,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    lr=0.001,
    batch_size=256,
    train_split=ValidSplit(0.2, random_state=seed),
    callbacks=[
        EpochScoring(scoring='f1_weighted', name='train_f1_weighted', on_train=True),
        EpochScoring(scoring='f1_weighted', name='valid_f1_weighted'),
        EarlyStopping(
            monitor='valid_f1_weighted',  # Metric to monitor
            patience=50,  # Number of epochs to wait for improvement
            threshold=0.0001,  # Minimum change to qualify as an improvement
            threshold_mode='rel',  # Relative change
            lower_is_better=False,  # For metrics where higher is better (e.g., F1 score)
        )
    ],
    iterator_train__shuffle=True
)

# Start timing
start_time = time.time()

# Fit the model
net.fit(X_train_resampled, Y_train_resampled)

# End timing
end_time = time.time()

# Predict probabilities on the test set
Y_pred_proba = net.predict_proba(X_test)
Y_pred = net.predict(X_test)


# Plot the F1 score over epochs
plt.figure(figsize=(10,5))
plt.plot(net.history[:, 'train_f1_weighted'], label="Train Weighted F1")
plt.plot(net.history[:, 'valid_f1_weighted'], label="Validation Weighted F1")
plt.xlabel('Epoch')
plt.ylabel('Weighted F1 Score')
plt.legend()
plt.title('Weighted F1 Score Over Epochs - Backpropagation Neural Network')
plt.show()

from sklearn.model_selection import learning_curve

# Generate the learning curve with weighted F1 score
train_sizes, train_scores, test_scores = learning_curve(
    net, X_scaled, Y, train_sizes=np.linspace(0.1, 1.0, 5), cv=3, scoring='f1_weighted'
)

# Calculate mean and standard deviation of training and test scores
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='cyan')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='darkorchid')
plt.plot(train_sizes, train_scores_mean, label="Training Weighted F1 Score", color='cyan')
plt.plot(train_sizes, test_scores_mean, label="Validation Weighted F1 Score", color='darkorchid')
plt.title("Learning Curve - Backpropagation Neural Network")
plt.xlabel("Training Size")
plt.ylabel("Weighted F1 Score")
plt.grid(visible=True)
plt.legend(loc="best", frameon=False)
plt.show()


# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")

# Backpropagation Neural Network
# Total training time: 3.18 seconds
# Weighted F1 Score: 0.8038827403274489
# Accuracy: 0.7984084880636605
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.64      0.86      0.74       124
#            1       0.92      0.77      0.84       253

#     accuracy                           0.80       377
#    macro avg       0.78      0.81      0.79       377
# weighted avg       0.83      0.80      0.80       377


##############  RHC Neural Network  ##############

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

# Define the neural network using RHCModule
net = NeuralNetClassifier(
    module=RHCModule,
    module__layer_sizes=(12, 32, 16, 2),
    module__activation=nn.Tanh(),
    module__dropout_percent=0.1,
    module__output_activation=nn.Softmax(dim=-1),
    max_epochs=300,
    verbose=1,
    criterion=nn.CrossEntropyLoss(),
    lr=0.001,
    batch_size=256,
    train_split=ValidSplit(0.2, random_state=seed),
    callbacks=[
        EpochScoring(scoring='f1_weighted', name='train_f1_weighted', on_train=True),
        EpochScoring(scoring='f1_weighted', name='valid_f1_weighted'),
        EarlyStopping(
            monitor='valid_f1_weighted',  # Metric to monitor
            patience=50,  # Number of epochs to wait for improvement
            threshold=0.0001,  # Minimum change to qualify as an improvement
            threshold_mode='rel',  # Relative change
            lower_is_better=False,  # For metrics where higher is better (e.g., F1 score)
        )
    ],
    iterator_train__shuffle=True
)

# Start timing
start_time = time.time()

# Fit the model
net.fit(X_train_resampled, Y_train_resampled)

# End timing
end_time = time.time()

# Predict probabilities on the test set
Y_pred_proba = net.predict_proba(X_test)
Y_pred = net.predict(X_test)

# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Plot the F1 score over epochs
plt.figure(figsize=(10,5))
plt.plot(net.history[:, 'train_f1_weighted'], label="Train Weighted F1")
plt.plot(net.history[:, 'valid_f1_weighted'], label="Validation Weighted F1")
plt.xlabel('Epoch')
plt.ylabel('Weighted F1 Score')
plt.legend()
plt.title('Weighted F1 Score Over Epochs - RHC Neural Network')
plt.show()

from sklearn.model_selection import learning_curve

# Generate the learning curve with weighted F1 score
train_sizes, train_scores, test_scores = learning_curve(
    net, X_scaled, Y, train_sizes=np.linspace(0.1, 1.0, 5), cv=3, scoring='f1_weighted'
)

# Calculate mean and standard deviation of training and test scores
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='cyan')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='darkorchid')
plt.plot(train_sizes, train_scores_mean, label="Training Weighted F1 Score", color='cyan')
plt.plot(train_sizes, test_scores_mean, label="Validation Weighted F1 Score", color='darkorchid')
plt.title("Learning Curve - RHC Neural Network")
plt.xlabel("Training Size")
plt.ylabel("Weighted F1 Score")
plt.grid(visible=True)
plt.legend(loc="best", frameon=False)
plt.show()

# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")

# RHC Neural Network
# Weighted F1 Score: 0.7390696158679139
# Accuracy: 0.7320954907161804
# Total training time: 10.70 seconds
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.55      0.94      0.70       124
#            1       0.96      0.63      0.76       253

#     accuracy                           0.73       377
#    macro avg       0.76      0.79      0.73       377
# weighted avg       0.83      0.73      0.74       377

############ Simulated Annealing Neural Network ############

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
    module__t=30000,
    module__cooling=0.4,
    max_epochs=300,
    verbose=1,
    criterion=nn.CrossEntropyLoss(),
    lr=0.001,
    batch_size=256,
    train_split=ValidSplit(0.2, random_state=seed),
    callbacks=[
        EpochScoring(scoring='f1_weighted', name='train_f1_weighted', on_train=True),
        EpochScoring(scoring='f1_weighted', name='valid_f1_weighted'),
        EarlyStopping(
            monitor='valid_f1_weighted',  # Metric to monitor
            patience=50,  # Number of epochs to wait for improvement
            threshold=0.0001,  # Minimum change to qualify as an improvement
            threshold_mode='rel',  # Relative change
            lower_is_better=False,  # For metrics where higher is better (e.g., F1 score)
        )
    ],
    iterator_train__shuffle=True
)


# Start timing
start_time = time.time()

# Fit the model
net.fit(X_train_resampled, Y_train_resampled)

# End timing
end_time = time.time()

# Predict probabilities on the test set
Y_pred_proba = net.predict_proba(X_test)
Y_pred = net.predict(X_test)

# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Plot the F1 score over epochs
plt.figure(figsize=(10,5))
plt.plot(net.history[:, 'train_f1_weighted'], label="Train Weighted F1")
plt.plot(net.history[:, 'valid_f1_weighted'], label="Validation Weighted F1")
plt.xlabel('Epoch')
plt.ylabel('Weighted F1 Score')
plt.legend()
plt.title('Weighted F1 Score Over Epochs - Simulated Annealing Neural Network')
plt.show()

from sklearn.model_selection import learning_curve

# Generate the learning curve with weighted F1 score
train_sizes, train_scores, test_scores = learning_curve(
    net, X_scaled, Y, train_sizes=np.linspace(0.1, 1.0, 5), cv=3, scoring='f1_weighted'
)

# Calculate mean and standard deviation of training and test scores
train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='cyan')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='darkorchid')
plt.plot(train_sizes, train_scores_mean, label="Training Weighted F1 Score", color='cyan')
plt.plot(train_sizes, test_scores_mean, label="Validation Weighted F1 Score", color='darkorchid')
plt.title("Learning Curve - SA Neural Network")
plt.xlabel("Training Size")
plt.ylabel("Weighted F1 Score")
plt.grid(visible=True)
plt.legend(loc="best", frameon=False)
plt.show()

# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")

# Simulated Annealing Neural Network
# 0.4 Cooling rate and 30000 temperature
# Weighted F1 Score: 0.7285409792890761
# Accuracy: 0.7214854111405835
# Total training time: 5.59 seconds
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.54      0.94      0.69       124
#            1       0.95      0.62      0.75       253

#     accuracy                           0.72       377
#    macro avg       0.75      0.78      0.72       377
# weighted avg       0.82      0.72      0.73       377

############### Genetic Algorithm Neural Network ###############
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
    max_epochs=300,
    verbose=1,
    criterion=nn.CrossEntropyLoss(),
    lr=0.001,
    batch_size=256,
    train_split=ValidSplit(0.2, random_state=seed),
    callbacks=[
        EpochScoring(scoring='f1_weighted', name='train_f1_weighted', on_train=True),
        EpochScoring(scoring='f1_weighted', name='valid_f1_weighted'),
        EarlyStopping(
            monitor='valid_f1_weighted',  # Metric to monitor
            patience=50,  # Number of epochs to wait for improvement
            threshold=0.0001,  # Minimum change to qualify as an improvement
            threshold_mode='rel',  # Relative change
            lower_is_better=False,  # For metrics where higher is better (e.g., F1 score)
        )
    ],
    iterator_train__shuffle=True,
)


# Start timing
start_time = time.time()

# Fit the model
net.fit(X_train_resampled, Y_train_resampled)

# End timing
end_time = time.time()

# Predict probabilities on the test set
Y_pred_proba = net.predict_proba(X_test)
Y_pred = net.predict(X_test)

# Plot the F1 score over epochs
plt.figure(figsize=(10,5))
plt.plot(net.history[:, 'train_f1_weighted'], label="Train Weighted F1")
plt.plot(net.history[:, 'valid_f1_weighted'], label="Validation Weighted F1")
plt.xlabel('Epoch')
plt.ylabel('Weighted F1 Score')
plt.legend()
plt.title('Weighted F1 Score Over Epochs - Genetic Algorithm Neural Network')
plt.show()

# Evaluate the model performance
f1 = f1_score(Y_test, Y_pred, average='weighted')
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")

# Genetic Algorithm Neural Network
# Weighted F1 Score: 0.5437856701463881
# Accuracy: 0.5305039787798409
# Total training time: 208.69 seconds
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.37      0.60      0.46       124
#            1       0.72      0.49      0.59       253

#     accuracy                           0.53       377
#    macro avg       0.54      0.55      0.52       377
# weighted avg       0.60      0.53      0.54       377

###############  END  ###############