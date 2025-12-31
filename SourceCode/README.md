The source code is developed in Python 3 and follows an object-oriented programming approach.

Below is a comprehensive guide designed to facilitate your understanding of the code structure. See `ThaCo3/README.md` file for the execution command, through the main script `thaco.py`.

- **Code Structure**:
  - `Parent_Network`: This class initializes the neural network. It is responsible for creating arrays for training and test datasets, initializing neural network layers and noise inputs, interconnecting them, and setting the network parameters.
  - `Parent_Train`: Dedicated to loading the training dataset into arrays, this class divides the data into features and labels.
  - `Parent_Test`: Similar to `Parent_Train`, this class loads the test dataset into arrays and separates them into features and labels.
  - `Parent_Layer`: This class is responsible for setting up the network layers.
  - `Parent_Noise`: This class initializes noise layers and contains functions to inject noise into the network across different brain states.
  - `Parent_Connections`: This class provides functions to interconnect layers, modulate parameters for changing brain states, and manage I/O of network synaptic states.
  - `Parent_BrainStates`: This class is designed for configuring brain states and simulating network dynamics.
  - `Parent_Network`: this class allows to construct a `var_dict` dictionary for variable parameters that are candidates for alteration during phase space exploration, to generate a distinct save path for different configurations of the network, and finally to conduct simulations on a single core or on multiple cores. Note that multi-core simulations rely on SLURM workload manager.
