import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def wind_driven_optimization(population_size, max_iterations, data, target, fitness_function):
  """
  Implements a Wind Driven Optimization (WDO) algorithm.

  Args:
      population_size: Number of particles in the population.
      max_iterations: Maximum number of iterations for the optimization.
      data: Input data for the ANN.
      target: Target variable (radiation) for the ANN.
      fitness_function: Function to evaluate particle fitness.

  Returns:
      Best particle's position (weights and biases for the ANN).
  """

  # Define particle structure (weights and biases)
  particle_dim = (data.shape[1] + 1) * (hidden_layers * 2 + 1)  # Account for all layers
  particle_positions = np.random.rand(population_size, particle_dim)

  # Initialize personal best positions and fitness values
  best_positions = np.copy(particle_positions)
  best_fitness_values = np.zeros(population_size)
  for i in range(population_size):
    best_fitness_values[i] = fitness_function(data, target, particle_positions[i])

  # Main optimization loop
  for iteration in range(max_iterations):

    # Update velocity based on current position, personal best, and neighborhood information
    for i in range(population_size):
      velocity = np.zeros_like(particle_positions[i])
      # Implement wind term (exploratory force) here (example using random noise)
      wind_term = np.random.uniform(-wind_strength, wind_strength, particle_dim)
      velocity += wind_term

      # Personal best term
      personal_best_term = learning_rate * (best_positions[i] - particle_positions[i])

      # Neighborhood term (information exchange)
      # (Implement based on neighborhood selection strategy, e.g., ring topology)
      neighborhood_term = np.zeros_like(particle_positions[i])
      # ... (calculate neighborhood_term based on neighboring particles' information)

      velocity += personal_best_term + neighborhood_term

      # Update particle position with velocity clamping
      particle_positions[i] += np.clip(velocity, -velocity_limit, velocity_limit)

    # Evaluate fitness of updated positions
    for i in range(population_size):
      fitness_value = fitness_function(data, target, particle_positions[i])
      if fitness_value < best_fitness_values[i]:
        best_fitness_values[i] = fitness_value
        best_positions[i] = np.copy(particle_positions[i])

  # Return the best particle's position
  return best_positions[np.argmin(best_fitness_values)]

def fitness_function(data, target, particle_position):
  """
  Calculates the mean squared error (MSE) between predicted and actual radiation.

  Args:
      data: Input data for the ANN.
      target: Target variable (radiation) for the ANN.
      particle_position: Position of a particle (weights and biases).

  Returns:
      Mean squared error (MSE).
  """

  # Extract weights and biases from particle position
  weights, biases = reshape_particle_position(particle_position)

  # Build ANN with the weights and biases
  model = build_ann(weights, biases)

  # Make predictions
  predictions = model.predict(data)

  # Calculate MSE
  mse = np.mean((predictions - target) ** 2)
  return mse

def reshape_particle_position(particle_position):
  """
  Reshapes the particle position into weights and biases for the ANN.

  Args:
      particle_position: Position of a particle (1D array).

  Returns:
      Tuple of lists containing weights and biases (separate lists for each layer).
  """

  # Implement logic to reshape based on ANN architecture (example for 2 hidden layers)
  weights = []
  biases = []
  weight_index = 0
  for layer in range(1, hidden_layers + 1):  # Skip input layer
    # Weights for current layer
    num_weights = (data.shape[1] + 1) if layer == 1 else hidden_neurons
    weights.append(particle_position[weight_index:weight_index + num_weights * hidden_neurons])
    weight_index += num_weights * hidden_neurons

    # Biases for current layer
    biases.append(particle_position[weight_index:weight_index + hidden_neurons])
    weight_index += hidden_neurons

  return weights, biases

def build_ann(weights, biases):
  """
  Builds a Feed-Forward ANN with the provided weights and biases.

  Args:
      weights: List of weight matrices for each layer (excluding input layer).
      biases: List of bias vectors for each layer (excluding input layer).

  Returns:
      Compiled Feed-Forward ANN model.
  """

  model = Sequential()

  # Input layer (no weights or biases)
  model.add(Dense(units=input_dim, activation='relu', input_shape=(data.shape[1],)))

  # Hidden layers based on weights and biases
  for i in range(hidden_layers):
    model.add(Dense(units=hidden_neurons, activation='relu', use_bias=False))  # No bias for hidden layers with separate bias lists
    model.add(Dense(units=hidden_neurons, activation='relu'))

  # Output layer
  model.add(Dense(units=1))  # Single output neuron for radiation prediction

  # Compile the model
  model.compile(optimizer='adam', loss='mse')  # Adam optimizer and mean squared error loss

  return model

# Hyperparameter definitions (adjust as needed)
learning_rate = 0.1
wind_strength = 0.2
velocity_limit = 1.0  # Clamping limit for velocity updates

# Load data from CSV file (replace 'your_data.csv' with the actual filename)
data = pd.read_csv('your_data.csv')

# Separate features and target
features = ['Data', 'Time', 'Temperature', 'Pressure', 'Humidity', 'WindDirection', 'WindSpeed']
target = 'Radiation'  # Assuming 'Radiation' is the target column
X = data[features]
y = data[target]

# Preprocess data (handle time feature and scaling)
def preprocess_data(X, y):
  # Extract hour from 'Time' column (or adjust time handling as needed)
  X['Hour'] = pd.to_datetime(X['Time'], format='%H:%M:%S').dt.hour

  # Drop the original 'Time' column
  X.drop('Time', axis=1, inplace=True)

  # Scale features (excluding hour)
  scaler = MinMaxScaler()
  scaled_X = scaler.fit_transform(X.iloc[:, 1:])  # Exclude hour for scaling

  # Combine hour and scaled features
  preprocessed_X = np.hstack((X['Hour'].values.reshape(-1, 1), scaled_X))

  return preprocessed_X, y

X, y = preprocess_data(X.copy(), y.copy())  # Avoid modifying original dataframes

input_dim = X.shape[1]  # Number of input features (including hour)
hidden_layers = 2  # Number of hidden layers (example)
hidden_neurons = 16  # Number of neurons per hidden layer (example)

# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# WDO optimization
population_size = 50
max_iterations = 100
best_particle_position = wind_driven_optimization(population_size, max_iterations, X_train, y_train, fitness_function)

# Extract weights and biases from the best particle
weights, biases = reshape_particle_position(best_particle_position)

# Build the ANN with the optimized weights and biases
model = build_ann(weights, biases
