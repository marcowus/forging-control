# ---------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------

# Create random numbers
import random

# DataFrames
import pandas as pd                                   

# Numpy
import numpy as np 

# Sklearn 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score # Metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler # Normalize data

# PyTorch 
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom plotly identifier
from plotly.io._base_renderers import BrowserRenderer, open_html_in_browser
from plotly.io._renderers import renderers

# Casadi - Optimization
from casadi import *

# Progress bar
from alive_progress import alive_bar

# Time library
from time import time  # Measure time

# MPC tools
from do_mpc.tools import Timer   
import do_mpc 

# Tables
from tabulate import tabulate

# Print information
import logging

# ----------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------- 

# Save log messages into a .log file
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Configure logging messages
logging.basicConfig(level=logging.INFO, format='%(message)s', 
                    handlers=[logging.FileHandler(f"my_log.log", mode='w'), stream_handler])
logger = logging.getLogger()

# ----------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------  
class SequenceDataset(Dataset):
    """
    Dataset class for sequential data.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the column to shift.
    target : list
        The target column in the dataset.
    features : list
        The list of feature columns (independent variables) used to predict the target.
    recurrent_features : list
        Columns used as recurrent features.
    lookback : int
        Number of previous timesteps used for recurrent features.
    
    Returns
    -------
        x : torch.Tensor
            Static features (independent variables) for the current timestep.
        y : torch.Tensor
            Target value at the next timestep.
        z : torch.Tensor 
            Features for the current and past time steps.
    """
    def __init__(self, dataframe:pd.DataFrame, target:list, features:list, recurrent_features, lookback=5):

        # Load parameters
        self.features = features
        self.target = target
        self.lookback = lookback

        # Load tensors
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.Z = torch.tensor(dataframe[recurrent_features].values).float()
        self.L = len(dataframe)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Return input-output pair for the given index."""
        # Static features (input)
        x = self.X[i] 

        # Recurrent features (LSTM inputs)
        if i >= self.lookback - 1:
            i_start = i - self.lookback + 1
            z = self.Z[i_start:(i + 1), :]
        else:
            padding = self.Z[0].repeat(self.lookback - i - 1, 1)
            z = self.Z[0:(i + 1), :]
            z = torch.cat((padding, z), 0)
        
        # Target values (output)
        if i < self.L - 1:
            y = self.y[i+1]
        else:
            y = self.y[-1]
        
        # Ensure valid index
        if i >= len(self): raise IndexError # Raise Error
    
        return x, y, z

class CreateDataset(Dataset):
    """
    Dataset class for creating sequences of time-series data suitable for recurrent neural networks.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the input features and target column.
    target : list
        List containing the target column name(s) to predict.
    features : list
        List of column names used as independent variables (features) to predict the target.
    lookback : int, optional
        Number of past timesteps included in each input sequence (default is 5).
    prediction_length : int, optional
        Number of future timesteps the model should predict (default is 1).

    Returns
    -------
    x : torch.Tensor
        Feature tensor of shape `(lookback, num_features)` representing the input sequence.
    y : torch.Tensor
        Target tensor of shape `(prediction_length,)` representing the sequence of future target values.
    """
    def __init__(self, dataframe:pd.DataFrame, target:list, features:list, lookback=5, prediction_length=1):

        # Load parameters
        self.features = features
        self.target = target
        self.lookback = lookback
        self.prediction_length = prediction_length

        # Load tensors
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.L = len(dataframe)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, i): 
        """
        Retrieve the input-output pair for a given index, managing padding for edge cases.

        Parameters
        ----------
        i : int
            Index of the dataset sample.

        Returns
        -------
        x : torch.Tensor
            Tensor of features over the specified lookback period.
        y : torch.Tensor
            Tensor of target values for the specified prediction length.
        """
        # Extract features (input)
        if i >= self.lookback - 1:
            i_start = i - self.lookback + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.lookback - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        
        # Extract targets (output)
        if i >= (self.L-self.prediction_length):
            padding = self.y[self.L-1].repeat(i - self.L + self.prediction_length + 1).unsqueeze(dim=0)
            y = self.y[i:self.L-1]
            y = torch.cat((padding, y), 0)
        else:
            i_end = i + self.prediction_length + 1
            y = self.y[i+1:i_end]
    
        return x, y
    
       
# ----------------------------------------------------------------
# FEEDFORWARD NEURAL NETWORK
# ----------------------------------------------------------------    
class FNNModel(nn.Module):
    """
    Feedforward Neural Network (FNN) for general use cases.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Number of neurons in each hidden layer.
    output_dim : int
        Number of output features.
    width_dim : int
        Number of hidden layers.
    activation_fn : callable, optional
        Activation function (default is `Tanh`).
    bias : bool, optional
        Whether to use bias in layers (default is `False`).
    
    Returns
    ----------
    out : torch.Tensor
        Neural network prediction with shape (batch_size, output_dim).
    """
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, width_dim:int, activation_fn=nn.ReLU, bias=True):
        """Method to initialize the feedforward Neural Network.""" 
        super(FNNModel, self).__init__()

        # Store parameters
        self.width_dim = width_dim
        self.activation = activation_fn()
        self.constraint = nn.Hardtanh()

        # Store layers
        self.fc_inp = nn.Linear(input_dim, hidden_dim, bias = bias)   # Linear function - Input
        self.fc_int = nn.Linear(hidden_dim, hidden_dim, bias = bias)  # Linear function - Intermediate
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias = False)  # Linear function - Output
        
        # Initialization
        nn.init.xavier_normal_(self.fc_inp.weight)
        nn.init.xavier_normal_(self.fc_int.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

        nn.init.zeros_(self.fc_inp.bias) 
        nn.init.zeros_(self.fc_int.bias) 
        
    def forward(self, x):   
        """ Forward pass through the network. 
        
        Parameters
        ----------
        x : torch.Tensor
            Features passed to the network.
            
        Returns
        ----------
        out : torch.Tensor 
            Neural network prediction with shape `(batch_size, output_dim)`.
        """
        # From input to intermediate layer    
        out = self.fc_inp(x)          
        out = self.activation(out)    

        # Between intermediate layers
        for _ in range(self.width_dim - 1):
            out = self.fc_int(out)     
            out = self.activation(out)

        # From intermediate to output layer    
        out = self.fc_out(out)        

        # Apply Constraint
        out = self.constraint(out)
        
        return out


# ----------------------------------------------------------------
# LONG SHORT TERM MEMORY NEURAL NETWORK
# ----------------------------------------------------------------  
class LSTMModel(nn.Module):
    """
    LSTMModel defines a stacked LSTM neural network for sequence modeling tasks.
    
    Parameters
    ----------
    input_dim : int 
        Number of input features per time step.
    hidden_dim : int
        Number of hidden neurons in each LSTM layer.
    output_dim : int 
        Number of output features.
    layer_dim : int 
        Number of stacked LSTM layers.
    bias : bool, optional
        Whether to use bias in LSTM layers (`default=False`).
    
    Returns
    ----------
    out : torch.Tensor 
        Neural network prediction with shape `(batch_size, output_dim)`.
    """
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, layer_dim:int, bias=False, device:torch.device = "cpu"):
        super(LSTMModel, self).__init__()

        # Model parameters
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bias=bias)
        # OBS: batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)

        # Fully connected (readout) layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def initialize_hidden_states(self, batch_size:int, device:torch.device):
        """
        Initialize hidden and cell states with zeros.
        
        Parameters
        ----------
        batch_size : int
            Current batch size.
        device : torch.device
            Device for tensor allocation (`cuda:0` or `cpu`).
        
        Returns
        ----------
        h0 : torch.Tensor
            Initialized hidden and cell states.
        c0 : torch.Tensor
            Initialized hidden and cell states.
        """
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device).requires_grad_()
        return h0, c0

    def forward(self, x:torch.Tensor, device:torch.device):
        """
        Forward pass for the LSTM model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, seq_length, input_dim)`.
        
        Returns
        ----------
        out : torch.Tensor
            Neural network prediction with shape `(batch_size, output_dim)`.
        """
        # Get batch size and device
        batch_size, _, _ = x.size()

        # Initialize hidden and cell states
        h0, c0 = self.initialize_hidden_states(batch_size, device)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Extract hidden state of the last time step and pass through fully connected layer
        out = self.fc(out[:, -1, :])  # Shape: (batch_size, output_dim)

        return out
    
# ----------------------------------------------------------------
# DATA
# ---------------------------------------------------------------- 
class Data:
    """Class to deal with data pre-processing in before training begins."""
    
    @staticmethod
    def random_simulation(seed:int):
        """
        Make simulations deterministic by setting a fixed random seed for all sources of randomness.

        Parameters
        ----------
        seed : int
            The seed value to use for random number generation, ensuring reproducibility 
                    across random, and PyTorch libraries, including CUDA operations.
        """
        # Set seeds for random, and torch
        random.seed(seed)
        torch.manual_seed(seed)

        # If using CUDA, set seed for all CUDA devices and configure PyTorch to use deterministic algorithms
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) 

    @staticmethod
    def get_scaler(scaler:str):
        """
        Retrieve an instance of the specified scaler for data normalization.

        Parameters
        ----------
        scaler : str
            The name of the scaler to retrieve. Supported options include:
            - `minmax` : MinMaxScaler
            - `standard` : StandardScaler
            - `maxabs` : MaxAbsScaler
            - `robust` : RobustScaler

        Returns
        -------
        scaler_instance : object
            An instantiated scaler object. Raises a `ValueError` if an invalid scaler name is provided.

        Raises
        ------
        ValueError
            If an unsupported scaler name is given.
        """
        scalers = {
            "minmax":   MinMaxScaler,
            "standard": StandardScaler,
            "maxabs":   MaxAbsScaler,
            "robust":   RobustScaler,
        }

        # Retrieve and instantiate the scaler
        scaler_name = scaler.lower()
        if scaler_name not in scalers:
            raise ValueError(f"Scaler '{scaler}' is not supported. Choose from {list(scalers.keys())}.")
        
        return scalers[scaler_name]()
    
    @staticmethod
    def shift_commands(df:pd.DataFrame, col:str, fill_value=0):
        """
        Shift a specified column in a DataFrame by one row, filling the resulting 
        empty space with a specified value. Returns a modified copy of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to shift.
        col : str 
            The name of the column to shift.
        fill_value : int, optional
            The value to use for filling the shifted position. Defaults to `0`.

        Returns
        -------
        df : pd.DataFrame
            A DataFrame with the specified column shifted.
        
        Raises
        ------
        KeyError: 
            If the specified column is not in the DataFrame.
        """
        # Check if the column exists in the DataFrame
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        # Shift the specified column and fill with the provided value
        df[col] = df[col].shift(periods=1, fill_value=fill_value)

        return df
    
    @staticmethod
    def get_individual_dataset(df:pd.DataFrame, target:list, features:list, model_features:list, t_traj:int, lookback = 1):
        """
        Create individual datasets for a given DataFrame and sequence length.

        Parameters
        ----------
        df : DataFrame
            Input scaled DataFrame.
        target : list
            Columns from the DataFrame serving as output to the neural network.
        features : list
            Columns from the DataFrame serving as input to the neural network.
        t_traj : int
            Sequence length per trajectory.
        lookback : int 
            Lookback window size for recurrent datasets.

        Returns
        -------
        datasets : list
            List of datasets used to train the feed-forward neural network.
        datasets_recurrent : list
            List of datasets used to train the LSTM.
        """
        datasets = []
        datasets_recurrent = []

        for i in range (len(df) // t_traj):
            # Select the data from the DataFrame with the appropriate sequence length
            data = df.iloc[i*t_traj:(i+1)*t_traj]

            # Recurrent DataSets
            datasets_recurrent.append(CreateDataset(data, target, features, lookback)) 
            
            # Regular DataSets
            datasets.append(SequenceDataset(data, target, features, model_features, lookback)) 

        return datasets, datasets_recurrent
    
    @staticmethod
    def append_prediction(dataframe:pd.DataFrame, prediction:torch.Tensor, target:list):
        """
        Append neural network predictions to a given DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input scaled DataFrame.
        prediction : torch.Tensor or np.ndarray
            Columns from the DataFrame serving as output to the neural network.
        target : list
            Target that the neural network is trying to predict.

        Returns
        -------
        dataframe : pd.DataFrame
            DataFrame with the column `NN({target})` appended with the values in `prediction`.
        """
        # Initialize empty list
        prediction_col = []

        # Loop to concatenate the scaled prediction to the DataFrame
        for idx, col in enumerate(target):
            col_name = f"NN({col})"
            prediction_col.append(col_name)

            # Append NN prediction to the DataFrame
            dataframe[col_name] = prediction[:,idx]

            # Shift the commands by 1 (quantity being predicted)
            dataframe = Data.shift_commands(dataframe, col_name, fill_value = 0)

        return dataframe
    
    @staticmethod
    def show_tabulate(results_dict:dict, file = '', feasibility = False, save_flag = True, debug_flag = False):
        """
        Display and save a table with simulation results.

        Parameters
        ----------
        results_dict : dict
            Dictionary with the results of each closed-loop simulation.
        file : str
            Name of the file to save table results.
        feasibility : bool, optional
            Feasibility Recovery flag.
        save_flag : bool, optional
            Flag to save the table locally.
        debug_flag : bool, optional
            Flag that indicates if debugging mode is `on`.
        """
        # Build table to terminal
        table = tabulate(results_dict, results_dict.keys(), tablefmt="fancy_grid", floatfmt=".3f", showindex= True)
        print(table)

        # Build table to be stocked locally
        if save_flag and not debug_flag:
            table = tabulate(results_dict, results_dict.keys(), tablefmt="tsv", showindex= True)
            if feasibility:
                text_file=open(f"Tables/Feasibility/" + file, "w")
            else:
                text_file=open(f"Tables/Normal/" + file, "w")

            text_file.write(table) # Write table to file
            text_file.close()      # Close table to file
    

# ----------------------------------------------------------------
# NEURAL NETWORK
# ----------------------------------------------------------------   
class NeuralNetwork:
    """Class to manage feed-forward neural networks, including training, validation, prediction, and evaluation."""

    @staticmethod
    def train_model(data_loader:DataLoader, simulator:nn.Module, model:nn.Module, loss_function:nn.Module, optimizer: torch.optim.Optimizer, device:torch.device, enable_noise = False):
        """
        Train the neural network model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        simulator : nn.Module
            LSTM used to predict future states of the system.
        model : nn.Module
            Feed-forward neural network used to predict the next command.
        loss_function : nn.Module
            Loss function used in the backpropagation.
        optimizer : torch.optim.Optimizer
            Optimizer used to find the network parameters (e.g., `AdamW`).
        device : torch.device
            Device in which training is performed (`cuda:0` or `cpu`).
        enable_noise : bool
            Flag that determines whereas noise should be applied to the simulator predictions during training.

        Returns
        ----------
        avg_loss : float
            Average training loss over all batches.

        loss_features : dict
            Dictionary containing the keys:
            - `loss`: loss for each training data point, 
            - `command`: command loss for each data point, 
            - `error`: error loss for each data point.
            - `prediction`: command predicted for each data point.
        """
        model.train()   # Set model to training mode
        total_loss = 0  # Initialize cumulative loss

        # Declare empty lists
        list_loss, prediction_loss = [], []
        command_loss, error_loss = [], []

        # Training loop       
        for X, _, z in data_loader: 
            # Move inputs and targets to the device
            X, z = X.to(device), z.to(device)

            # Reset gradients
            optimizer.zero_grad()  

            # Forward pass    
            output = model(X)   

            # Compute loss 
            loss, loss_features = loss_function(simulator, model, X, output, z, device, enable_noise)

            # Append features to list
            list_loss.append(loss_features.get('loss'))
            command_loss.append(loss_features.get('command'))
            error_loss.append(loss_features.get('error'))
            prediction_loss.append(loss_features.get('prediction'))

            # Backpropagation
            loss.backward()  

            # Update weights
            optimizer.step()  

            # Accumulate loss
            total_loss += loss.item()  

        # Transform list to tensor
        list_loss = torch.cat(list_loss, dim=0) 
        command_loss = torch.cat(command_loss, dim=0) 
        error_loss = torch.cat(error_loss, dim=0) 
        prediction_loss = torch.cat(prediction_loss, dim=0)

        # Add loss features to dictionary
        loss_features = {'loss' : list_loss  , 'command'   : command_loss, 
                         'error': error_loss , 'prediction': prediction_loss}

        # Compute average loss
        avg_loss = total_loss / len(data_loader)

        return avg_loss, loss_features
    
    @staticmethod
    def validate_model(data_loader:DataLoader, model:nn.Module, loss_function:nn.Module, device:torch.device):
        """
        Validate the neural network model.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader for the validation dataset.
        model : nn.Module 
            Feed-forward neural network used to predict the next command.
        loss_function : nn.Module
            Loss function used in the backpropagation.
        device : torch.device
            Device in which training is performed (`cuda:0` or `cpu`).

        Returns
        -------
        avg_loss : float
            Average training loss over all batches.
        """
        model.eval()  # Set model to evaluation mode
        total_loss = 0
        
        # Validation loop
        with torch.no_grad():  # Disable gradient computation for validation         
            for X, y, _ in data_loader: 
                # Move inputs and targets to the device
                X, y = X.to(device), y.to(device)

                # NN prediction
                output = model(X)  
                
                # Compute loss
                total_loss += loss_function(output, y).item()

        # Compute average loss
        avg_loss = total_loss / len(data_loader)

        return avg_loss
    
    @staticmethod
    def predict(data_loader:DataLoader, model:nn.Module):
        """
        Generate predictions using the neural network on a given data_loader.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader for prediction.
        model : nn.Module
            Feed-forward neural network used to predict the next command.

        Returns
        -------
        output : torch.Tensor
            Concatenated predictions for all batches.
        """

        model.eval()  # Set model to evaluation mode
        predictions = []

        # Prediction loop
        with torch.no_grad():
            for X, _, z in data_loader:  
                predictions.append(model(X))  # Collect predictions 

        # Concatenate along the batch dimension
        output = torch.cat(predictions, dim=0) 

        return output 
    
    @staticmethod
    def metrics(truth: np.ndarray, out:np.ndarray, results_dict: dict):
        """
        Compute and display evaluation metrics for the neural network's predictions.

        Parameters
        ----------
        truth : np.ndarray
            Ground truth target values.
        out : np.ndarray 
            Predicted values from the neural network.
        results_dict : dict
            Dictionary to append test results.

        Returns
        -------
        results (dict):
            Dictionary containing the keys:
            - `MAE`: Mean Average Error, 
            - `RMSE`: Root Mean Squared Error,
            - `R2`: R2 Coefficient.
        """
        results = {'mae': mean_absolute_error(truth, out),
                   'rmse': root_mean_squared_error(truth, out),
                   'r2': r2_score(truth, out)
        }

        # Print results
        logger.info(f"- Mean Average Error (MAE) = {results['mae']:.4f}.")
        logger.info(f"- Root Mean Squared Error (RMSE) = {results['rmse']:.4f}.")
        logger.info(f"- R2 Coefficient = {results['r2']:.4f}.")

        # Append results
        results_dict['MAE'].append(results['mae'])
        results_dict['RMSE'].append(results['rmse'])
        results_dict['R2'].append(results['r2'])
    
    @staticmethod
    def other_metrics(input: np.ndarray, timer:np.ndarray, results_dict: dict):
        """
        Compute and display the average input and time complexity for the neural network's predictions.

        Parameters
        ----------
        input : np.ndarray
            Control inputs predicted by the neural network.
        timer : np.ndarray
            Time statistics of the closed-loop simulation.
        results_dict : dict
            Dictionary to append test results.

        Returns
        -------
        results : dict
            Dictionary containing the keys:
            - `Command`: average control input. 
            - `Mean_time`: average computational time.
            - `median_time`: median computational time.
            - `Std_time`: standard deviation of the computational time.
            - `25_perc`: 25 percentile of the computational time,
            - `75_perc`: 75 percentile of the computational time.
        """
        # Computational time and input effort
        logger.info(f"- Average Command = {np.mean(np.abs(input)):.4f}.")
        logger.info(f"- Runtime = {np.mean(timer)*1000:.4f} +- {np.std(timer)*1000:.4f} ms. Median = {np.percentile(timer, 50)*1000:.4f} with [{np.percentile(timer, 25)*1000:.4f}, {np.percentile(timer, 75)*1000:.4f}] ms")

        # Append results to dictionary
        results_dict['Command'].append(np.mean(np.abs(input)))
        results_dict['Mean_time'].append(np.mean(timer)*1000)
        results_dict['Std_time'].append(np.std(timer)*1000)
        results_dict['Median_time'].append(np.percentile(timer, 50)*1000)
        results_dict['25_perc'].append(np.percentile(timer, 25)*1000)
        results_dict['75_perc'].append(np.percentile(timer, 75)*1000)

    @staticmethod
    def train_loop(controller:nn.Module, simulator:nn.Module, train_loader:DataLoader, val_loader:DataLoader, loss_function:nn.Module, optimizer:torch.optim.Optimizer, n_epochs:int, device:torch.device, enable_noise = False):
        """
        Loop to train a neural network using unsupervised learning.

        Parameters
        ----------
        controller : nn.Module
            Feed-forward neural network used to predict the next command.
        simulator : nn.Module
            LSTM used to predict the next states in the MPC loss.
        train_loader : DataLoader
            DataLoader for training the feed-froward network.
        val_loader : DataLoader
            DataLoader for validation during training.
        loss_function : nn.Module
            Loss function used in the backpropagation.
        optimizer : torch.optim.Optimizer
            Optimizer used to find the network parameters (e.g., AdamW).
        n_epochs : int
            Number of times that all data points in the dataset have passed through the optimization.
        device : torch.device
            Device in which training is performed (`cuda:0` or `cpu`).
        enable_noise : bool
            Flag that determines whereas noise should be applied to the simulator predictions during training.

        Returns
        -------
        controller : nn.Module
            Feed-forward neural network with the updated weights.
        vec_t_loss : list
            List containing the average training loss at each epoch.
        vec_v_loss : list
            List containing the average validation loss at each epoch.
        comp_time : float
            Time taken to train the neural network.
        loss_features : dict
            Dictionary containing the keys:
            - `loss`: loss for each training data point, 
            - `command`: command loss for each data point, 
            - `error`: error loss for each data point.
            - `prediction`: command predicted for each data point.
        """
        logger.info("\n-------- \nTRAINING \n--------")
        validation_loss = NeuralNetwork.validate_model(val_loader, controller, nn.MSELoss(), device)

        # Variables initialization
        vec_t_loss, vec_v_loss = [], []

        # Loss features
        loss_matrix, prediction_loss = [], []
        command_loss, error_loss = [], []

        # Set timer 
        t0 = time()

        # Training loop
        with alive_bar(n_epochs, title='TRAINING', enrich_print = False) as bar:
            for ix_epoch in range(n_epochs):
                
                # Training function
                training_loss, loss_features = NeuralNetwork.train_model(train_loader, simulator, controller, loss_function, optimizer, device, enable_noise)

                # Validation loss
                validation_loss = NeuralNetwork.validate_model(val_loader, controller, nn.MSELoss(), device)

                # Print results
                if ix_epoch % (n_epochs/10) == 0:
                    logger.info(f"\n[{100*ix_epoch/n_epochs:.1f}%] Training loss: {training_loss:.4f},  Validation loss: {validation_loss:.4f}")
                elif ix_epoch == n_epochs-1:
                    logger.info(f"\n[100%] Training loss: {training_loss:.4f},  Validation loss: {validation_loss:.4f}")
                
                # Append losses
                vec_t_loss.append(training_loss)
                vec_v_loss.append(validation_loss)

                # Append cost vector
                loss_matrix.append(loss_features['loss'].detach())
                command_loss.append(loss_features['command'].detach())
                error_loss.append(loss_features['error'].detach())
                prediction_loss.append(loss_features['prediction'].detach())

                # Update progress bar
                bar()   

        # Transform list to tensor
        loss_matrix  = torch.stack(loss_matrix, dim=0)
        command_loss = torch.stack(command_loss, dim=0)
        error_loss   = torch.stack(error_loss, dim=0)
        prediction_loss = torch.stack(prediction_loss, dim=0)

        # Add loss features to a dictionary
        loss_features = {'loss' : loss_matrix, 'command'   : command_loss, 
                         'error': error_loss , 'prediction': prediction_loss}

        # Computational time
        comp_time = time() - t0
        logger.info(f"\nTotal time: {comp_time:.2f}s.")

        return controller, vec_t_loss, vec_v_loss, comp_time, loss_features
    
    @staticmethod
    def tvp_fun(t_now:float, ref_step:float, bias_work:int, bias_return:int, epsilon = 10**(-7)):
        """
        Function to compute the time-varying parameters of the MPC controller.
        In our case, we use the generate the reference at each time step of 
        the simulation.

        Parameters
        ----------
        t_now : float
            Current time in the closed-loop simulation.
        ref_step : float
            Reference that the system must follow.
        bias_work : int
            Number that is added to the `seed` to randomly generate the
            next reference of the working motion.
        bias_return : int
            Number that is added to the `seed` to randomly generate the
            next reference of the return motion.
        epsilon : float
            Constant to avoid precision errors when selecting the working 
            and return references.

        Returns
        -------
            ref (float): 
                Reference that the system should follow.
        """
        if ((t_now + epsilon)%ref_step) < ref_step/2: 
            # Select seed
            random.seed((t_now+epsilon)//ref_step + bias_work) 

            # Reference
            ref = 0.8*random.random() + 0.1
        else:
            # Select seed
            random.seed((t_now+epsilon)//ref_step + bias_return) 

            # Reference
            ref = -0.8*random.random() - 0.1

        return ref
    
    @staticmethod
    def simulator_make_step(X:np.ndarray, model:nn.Module, scalers:dict, noise: np.ndarray):
        """
        Function to compute the predict the next state of the system using a LSTM.

        Parameters
        ----------
        X : np.ndarray
            Array with the current state and command applied to the system.
        model : nn.Module
            LSTM network used to perform the prediction.
        scalers : dict
            Dictionary containing the scalers of the LSTM network.
        noise : np.ndarray
            Process noise for the simulator (NN).

        Returns
        -------
        output : np.ndarray
            Predicted state of the system.
        """
        # Evaluation mode
        model.eval()

        # Prediction
        with torch.no_grad():

            # Convert to 3D tensor
            X_new = torch.tensor(X).float()

            # Compute the output
            y_star = model(X_new, "cpu")
            # print(f"y_star =  {y_star}")

            # Add noise
            # print(f"noise = {torch.tensor(noise).float().unsqueeze(dim=0)}")
            y_star = y_star + torch.tensor(noise).float()
            # print(f"y_star + noise =  {y_star}")
            # print("\n")

            # Unscale the output
            output = scalers['output'].inverse_transform(y_star)

        return output
    
    @staticmethod
    def loop(N_traj:int, T_traj:int, Ts:float, controller:nn.Module, simulator:do_mpc.simulator.Simulator, simulator_LSTM:nn.Module, init_state:dict, 
             scalers:dict, model_scalers:dict, bias_work:float, bias_return:float, lookback:int, bar_title:str, process_std: np.ndarray, meas_std: np.ndarray,
             feasibility = False):
        """
        Function that implements the closed-loop simulation using a feed-forward neural 
        network as controller.

        Parameters
        ----------
            N_traj : int 
                Number of complete trajectories (working and return) in the simulation.
            T_traj : float
                Duration of the trajectory.
            Ts : float
                Time step of the controller.
            controller : nn.Module
                Feed-forward neural network to control the system.
            simulator : do_mpc.simulator.Simulator
                Simulator of the `do_mpc` acting like the physical system.
            simulator_LSTM : nn.Module
                LSTM network used to predict the next state of the system.
            init_state : dict
                Dictionary with the initial states of the system.
            scalers : dict
                Dictionary containing the scalers of the feed-forward network.
            model_scalers : dict
                Dictionary containing the scalers of the LSTM network.
            bias_work : float
                Number that is added to the `seed` to randomly generate the
                next reference of the working motion.
            bias_return : float
                Number that is added to the `seed` to randomly generate the
                next reference of the return motion.
            lookback : int 
                Lookback window size for recurrent datasets.
            bar_title :s tr
                Title displayed in the alive bar.
            process_std : np.ndarray
                Standard deviation of the process noise for each state.
            meas_std : np.ndarray
                Standard deviation of the measurement noise for each state.
            feasibility : bool, optional
                Activated the Feasibility Recovery strategy.

        Returns
        -------
            simulator : do_mpc.simulator.Simulator
                Simulator of the `do_mpc` acting like the physical system with the closed-loop data.
            results : dict
                Closed-loop trajectories from the `do_mpc` simulator.
            results_LSTM : dict
                Closed-loop trajectories from the LSTM simulator.
            timer : do_mpc.tools._timer.Timer
                Timer containing the computational cost of the controller.
            feasibility_results : dict
                Dictionary with the optimization results.
        """
        # Empty trajectories for the states command of the system (simulator)
        dataset_y     = np.zeros((N_traj, T_traj + 1))
        dataset_y_dot = np.zeros((N_traj, T_traj + 1))
        dataset_p1    = np.zeros((N_traj, T_traj + 1))
        dataset_p2    = np.zeros((N_traj, T_traj + 1))
        dataset_z     = np.zeros((N_traj, T_traj + 1))

        # Empty trajectories for the states command of the system (LSTM)
        dataset_y_dot_LSTM = np.zeros((N_traj, T_traj + 1))
        dataset_p1_LSTM    = np.zeros((N_traj, T_traj + 1))
        dataset_p2_LSTM    = np.zeros((N_traj, T_traj + 1))
        dataset_z_LSTM     = np.zeros((N_traj, T_traj + 1))

        # Create empty trajectories for the reference and the command
        dataset_ref = np.zeros((N_traj, T_traj))
        dataset_u   = np.zeros((N_traj, T_traj)) 

        # Empty trajectories for the Feasibility Recovery statistics 
        if feasibility:
            feas_inter_count = np.zeros((N_traj, T_traj + 1))
            feas_alpha_du    = np.zeros((N_traj, T_traj + 1))
            feas_alpha_pr    = np.zeros((N_traj, T_traj + 1))
            feas_d_norm      = np.zeros((N_traj, T_traj + 1))
            feas_inf_du      = np.zeros((N_traj, T_traj + 1))
            feas_inf_pr      = np.zeros((N_traj, T_traj + 1))
            feas_mu          = np.zeros((N_traj, T_traj + 1))
            feas_obj         = np.zeros((N_traj, T_traj + 1))
            feas_reg_size    = np.zeros((N_traj, T_traj + 1))

            feas_t_wall_callback_fun = np.zeros((N_traj, T_traj + 1))
            feas_t_wall_nlp_f        = np.zeros((N_traj, T_traj + 1))
            feas_t_wall_nlp_g        = np.zeros((N_traj, T_traj + 1))
            feas_t_wall_nlp_grad     = np.zeros((N_traj, T_traj + 1))
            feas_t_wall_nlp_grad_f   = np.zeros((N_traj, T_traj + 1))
            feas_t_wall_nlp_jac_g    = np.zeros((N_traj, T_traj + 1))

        # Set initial state
        y_init     = init_state.get('y', 0)
        y_dot_init = init_state.get('y_dot', 0)
        p1_init    = init_state.get('p1', 0)
        p2_init    = init_state.get('p2', 0)
        z_init     = init_state.get('z', 0)

        # Get the range of the inner loop - Simulator
        inner_loop = int(Ts/simulator.settings.t_step)

        # Duration of the reference
        T_ref = Ts*T_traj

        # Set a timer
        timer = Timer()

        # Noise seed
        # np.random.seed(42)

        # NN loop
        with alive_bar(N_traj, title = bar_title, enrich_print = False) as bar:
            for idx in range(N_traj):

                # Create the initial state vector
                x0 = np.array([y_init, y_dot_init, p1_init, p2_init, z_init]) 

                # Affect initial state to trajectory array (simulator)
                dataset_y[idx,0]     = y_init
                dataset_y_dot[idx,0] = y_dot_init
                dataset_p1[idx,0]    = p1_init
                dataset_p2[idx,0]    = p2_init
                dataset_z[idx,0]     = z_init

                # Affect initial state to trajectory array (LSTM)
                dataset_y_dot_LSTM[idx,0] = y_dot_init
                dataset_p1_LSTM[idx,0]    = p1_init
                dataset_p2_LSTM[idx,0]    = p2_init
                dataset_z_LSTM[idx,0]     = z_init

                # Set the initial state
                simulator.x0 = x0  

                # Change shape of initial condition (LSTM)
                x_next_LSTM = np.array([x0[1], x0[2], x0[3], x0[4]])

                # Change shape of initial condition (simulator)
                x0 = np.expand_dims(x0, axis=1)
                warm_start = {'u': np.zeros(3), 'lam_g': np.zeros(18)}

                # Main loop
                for t in range(T_traj):

                    # Set timer
                    timer.tic() 

                    # Get new reference from function 'my_tvp_fun'
                    t_now = (idx*T_traj + t)*Ts
                    ref = NeuralNetwork.tvp_fun(t_now, T_ref, bias_work, bias_return)

                    # NN input
                    input_NN = np.expand_dims(np.array([x0[1,0], x0[4,0], ref]), axis=0)

                    # Compute the NN command 
                    u0, sol, warm_start = FeasibilityRecovery.NN_make_step(input_NN, controller, scalers, x0, warm_start, feasibility)

                    # Stop timer
                    timer.toc()

                    # Process noise
                    w0 = np.random.normal(loc=0.0, scale=process_std).reshape(-1, 1)

                    # Measurement noise
                    v0 = np.random.normal(loc=0.0, scale=meas_std).reshape(-1, 1)

                    # Inner loop - Simulator
                    for _ in range(inner_loop):
                        x0 = simulator.make_step(u0, v0 = v0, w0 = w0)

                    # Squeeze the state vector
                    x_old = x0.squeeze()

                    # Add states to trajectory (simulator)
                    dataset_y[idx,t+1]     = x_old[0]
                    dataset_y_dot[idx,t+1] = x_old[1]
                    dataset_p1[idx,t+1]    = x_old[2]
                    dataset_p2[idx,t+1]    = x_old[3]
                    dataset_z[idx,t+1]     = x_old[4]

                    # Generate LSTM input
                    x_next_LSTM = np.concatenate((x_next_LSTM, u0[0]))
                    x_next_LSTM = np.expand_dims(x_next_LSTM, axis=0)
                    x_next_LSTM = model_scalers['input'].transform(x_next_LSTM)

                    # Different case scenario
                    if t == 0:
                        x_next_LSTM = np.repeat(x_next_LSTM, lookback, axis=0)
                        input_simulator = np.expand_dims(x_next_LSTM, axis=0)
                    else:
                        x_next_LSTM = np.expand_dims(x_next_LSTM, axis=0)
                        input_simulator = input_simulator[:,1:lookback,:]
                        input_simulator = np.concatenate((input_simulator, x_next_LSTM), axis=1)
                    
                    process_std_NN = np.array([0.0, 0.0, 0.0, 0.0])
                    w0_NN = np.random.normal(loc=0.0, scale=process_std_NN)

                    # # Process noise std - Simulator NN
                    # if np.any(process_std): 
                    #     process_std_NN = np.array([0.01, 0.01, 0.01, 0.01])
                    # else: 
                    #     process_std_NN = np.array([0.0, 0.0, 0.0, 0.0])

                    # # Process noise - Simulator NN
                    # w0_NN = np.random.normal(loc=0.0, scale=process_std_NN)

                    # Call Simulator NN
                    x_next_LSTM = NeuralNetwork.simulator_make_step(input_simulator, simulator_LSTM, model_scalers, w0_NN)

                    # Squeeze the state vector (LSTM)
                    x_next_LSTM = x_next_LSTM.squeeze()

                    # Add states to trajectory (LSTM)
                    dataset_y_dot_LSTM[idx,t+1] = x_next_LSTM[0]
                    dataset_p1_LSTM[idx,t+1]    = x_next_LSTM[1]
                    dataset_p2_LSTM[idx,t+1]    = x_next_LSTM[2]
                    dataset_z_LSTM[idx,t+1]     = x_next_LSTM[3]

                    # Retrieve reference trajectory       
                    dataset_ref[idx,t] = ref
                
                    # Add commands to trajectory
                    dataset_u[idx,t] = u0

                    if sol:
                        # Add optimization results to trajectory
                        feas_inter_count[idx,t+1] = sol.stats()['iter_count']
                        feas_alpha_du[idx,t+1]    = sol.stats()['iterations']['alpha_du'][-1]
                        feas_alpha_pr[idx,t+1]    = sol.stats()['iterations']['alpha_pr'][-1]
                        feas_d_norm[idx,t+1]      = sol.stats()['iterations']['d_norm'][-1]
                        feas_inf_du[idx,t+1]      = sol.stats()['iterations']['inf_du'][-1]
                        feas_inf_pr[idx,t+1]      = sol.stats()['iterations']['inf_pr'][-1]
                        feas_mu[idx,t+1]          = sol.stats()['iterations']['mu'][-1]
                        feas_obj[idx,t+1]         = sol.stats()['iterations']['obj'][-1]
                        feas_reg_size[idx,t+1]    = sol.stats()['iterations']['regularization_size'][-1]

                        feas_t_wall_callback_fun[idx,t+1] = sol.stats()['t_wall_callback_fun']
                        feas_t_wall_nlp_f[idx,t+1]        = sol.stats()['t_wall_nlp_f']
                        feas_t_wall_nlp_g[idx,t+1]        = sol.stats()['t_wall_nlp_g']
                        feas_t_wall_nlp_grad[idx,t+1]     = sol.stats()['t_wall_nlp_grad']
                        feas_t_wall_nlp_grad_f[idx,t+1]   = sol.stats()['t_wall_nlp_grad_f']
                        feas_t_wall_nlp_jac_g[idx,t+1]    = sol.stats()['t_wall_nlp_jac_g']

                # Set the initial command
                controller.u0 = 0.0  # MPC
                simulator.u0 = 0.0   # Simulator 

                # Update progress bar
                bar()

        # Add the results of the simulator in a dictionary
        results = {'y' : dataset_y  , 'y_dot': dataset_y_dot, 
                   'p1': dataset_p1 , 'p2'   : dataset_p2   ,
                   'z' : dataset_z  , 'ref'  : dataset_ref  , 
                   'u' : dataset_u}
        
        # Add the results of the LSTM simulator in a dictionary
        results_LSTM = {'y_dot': dataset_y_dot_LSTM, 'p1': dataset_p1_LSTM, 
                        'p2'   : dataset_p2_LSTM   , 'z' : dataset_z_LSTM }
        
        if feasibility:
            feasibility_results = {
                'iter_count': feas_inter_count      , 'alpha_du'    : feas_alpha_du, 
                'alpha_pr'  : feas_alpha_pr         , 'd_norm'      : feas_d_norm  ,
                'inf_du'    : feas_inf_du           , 'inf_pr'      : feas_inf_pr  , 
                'mu'        : feas_mu               , 'obj'         : feas_obj     ,
                'reg_size'  : feas_reg_size         , 'callback_fun': feas_t_wall_callback_fun,
                'nlp_f'     : feas_t_wall_nlp_f     , 'nlp_g'       : feas_t_wall_nlp_g       ,
                'nlp_grad'  : feas_t_wall_nlp_grad  , 'nlp_grad_f'  : feas_t_wall_nlp_grad_f  ,
                'nlp_jac_g'   : feas_t_wall_nlp_jac_g   
            }
        else:
            feasibility_results = 0.0
        
        return simulator, results, results_LSTM, timer, feasibility_results
    
    @staticmethod
    def test(loader:DataLoader, df:pd.DataFrame, df_scaled: pd.DataFrame, controller:nn.Module, target:dict, scalers:dict):
        """
        Append the neural network prediction to the original DataFrame 

        Parameters
        ----------
        loader : DataLoader
            DataLoader for prediction.
        df : DataFrame
            Unscaled DataFrame.
        df_scaled : DataFrame
            Scaled DataFrame.
        controller : nn.Module
            Feed-forward neural network used to predict the next command.
        target : dict
            Columns from the DataFrame serving as output to the neural network.
        scalers : dict
            Dictionary containing the scalers of the feed-forward network.
            
        Returns
        -------
        df : DataFrame
            Unscaled DataFrame with the neural network prediction appended.
        df_scaled : DataFrame
            Scaled DataFrame with the neural network prediction appended.
        """
        # Scaled prediction
        scaled_prediction = NeuralNetwork.predict(loader, controller)

        # Unscaled prediction
        unscaled_dataframe = scalers['output'].inverse_transform(scaled_prediction)

        # Append prediction to scaled DataFrame 
        df_scaled  = Data.append_prediction(df_scaled, scaled_prediction , target)

        # Append prediction to unscaled DataFrame 
        df = Data.append_prediction(df, unscaled_dataframe, target)

        return df, df_scaled


# ----------------------------------------------------------------
# MPC LOSS
# ----------------------------------------------------------------   
class MPCLoss(nn.Module):
    """
    Custom Loss Function that mimics the MPC cost function.
    
    Parameters
    ----------
    prediction_horizon : int 
        Size of the prediction horizon.
    alpha : int  
        Scaling factor of the command variation in comparison to the error.
    """
    def __init__(self, prediction_horizon = 10, alpha = 0.1):
        super(MPCLoss, self).__init__()
        self.N = prediction_horizon    # Prediction Horizon
        self.alpha = alpha             # Scaling factor - Command
        self.activation = nn.ReLU()    
        
    def forward(self, simulator:nn.Module, controller:nn.Module, input_controller:torch.Tensor, output_controller:torch.Tensor, states:torch.Tensor, device:torch.device, enable_noise = False):  
        """
        Calling the MPC loss function.
        
        Parameters
        ----------
        simulator : nn.Module
            LSTM to predict the future states of the system.
        controller : nn.Module
            Feed-forward neural network to predict the next commands.
        input_controller : torch.Tensor  
            Input of the controller (feed-forward neural network).
        output_controller : torch.Tensor   
            Output of the controller (feed-forward neural network).
        states : torch.Tensor
            Input of the simulator (LSTM). 
        device : torch.device
            Device in which training is performed (`cuda:0` or `cpu`).
        enable_noise : bool
            Flag that determines whereas noise should be applied to the simulator predictions during training.


        Returns
        ----------
        loss : torch.Tensor
            Average of the loss computed for each data point in the batch. 
        loss_features : dict
            Dictionary containing the keys:
                - `loss`: loss for each training data point, 
                - `command`: command loss for each data point, 
                - `error`: error loss for each data point.
                - `prediction`: command predicted for each data point.
        """
        # Initialization of the cost function
        cost         = torch.zeros((self.N,len(input_controller)), device = device)
        command_cost = torch.zeros_like(cost, device = device)
        error_cost   = torch.zeros_like(cost, device = device)

        # Extract the Reference
        ref = input_controller[:,-1]
        
        # Input of the NN-Simulator
        input_simulator = states.clone()
        input_simulator[:, -1, -1] = output_controller.squeeze()

        # Predict future states (NN Simulator)
        x0 = simulator(input_simulator, device)
        if enable_noise:
            noise = torch.randn_like(x0, device = device) * 0.01
            x0 += noise

        # Command loss - N = 0
        command_cost[0] = self.alpha*torch.square(input_simulator[:,-2,-1] - input_simulator[:,-1,-1])

        # Error loss - N = 0
        error_cost[0] = torch.square(x0[:,0] - ref)

        # Constraint loss - N = 0
        constraint_cost = self.activation(-x0[:,1]) + self.activation(-x0[:,2]) + (self.activation(x0[:,1] - 2.122366)) + (self.activation(x0[:,2] - 1.036233))

        # Compute global cost
        cost[0] = error_cost[0] + command_cost[0] + constraint_cost

        # Set the predicted command as u_next
        u_next = output_controller.clone()
        prediction_vector = u_next.clone()

        # Loop over the Prediction Horizon - N
        for j in range (self.N-1):
            
            # Input - Controller
            input_controller = torch.stack((x0[:,0], x0[:,3], ref), dim = 1)
                         
            # Get the old command
            u0 = u_next.clone()

            # Compute the next command
            u_next = controller(input_controller)
            
            # Input - Simulator
            x_next = torch.cat((x0, u_next), dim=1).unsqueeze(dim=1)
            input_simulator = torch.cat((input_simulator[:,1:10,:], x_next), dim=1)

            # Compute future states
            x0 = simulator(input_simulator, device)
            if enable_noise:
                noise = torch.randn_like(x0, device = device) * 0.01
                x0 += noise

            # Error loss - N = j + 1
            error_cost[j+1] = torch.square(x0[:,0] - ref)

            # Command loss - N = j + 1
            command_cost[j+1] = self.alpha*torch.square(u0.squeeze() - u_next.squeeze())

            # Accumulate cost function
            constraint_cost = self.activation(-x0[:,1]) + self.activation(-x0[:,2]) + (self.activation(x0[:,1] - 2.122366)) + (self.activation(x0[:,2] - 1.036233))
            
            # Total cost - N = j + 1
            cost[j+1] = error_cost[j+1] + command_cost[j+1] + constraint_cost

            # Get the prediction
            prediction_vector = torch.cat((prediction_vector,u_next), dim=1)

        # Cost vector
        cost_vector         = torch.sum(cost,dim=0)/self.N
        command_cost_vector = torch.sum(command_cost,dim=0)/self.N
        error_cost_vector   = torch.sum(error_cost,dim=0)/self.N

        # MPC loss - Average of all cost functions in the batch  
        loss = torch.mean(cost_vector)

        # Flatten the prediction vector
        prediction_vector = torch.flatten(prediction_vector)

        # Add loss features to dictionary
        loss_features = {'loss' : cost_vector      , 'command': command_cost_vector, 
                         'error': error_cost_vector, 'prediction': prediction_vector}

        return loss, loss_features
    
# ----------------------------------------------------------------
# FEASIBILITY RECOVERY
# ---------------------------------------------------------------- 
class FeasibilityRecovery:
    """Class to deal with the Feasibility Recovery Strategy."""
    
    def feasibility_recover(u_NN:np.float64, x_init:np.ndarray, warm_start: dict, feasibility_variables:dict):
        """
        Function to perform to solve the optimization within the Feasibility Recovery Framework.
        
        Parameters
        ----------
        u_NN : np.float64
            Command computed by the Neural Network.
        x_init : np.ndarray
            Current position of the system.
        warm_start : dict
            Dictionary with the warm start details:
            * `u` (casadi.MX): optimization variables.
            * `lam_g` (casadi.MX): Lagrange multipliers.
        feasibility_variables : dict
            Dictionary to get the optimization details:
            * `opti` (casadi.Opti): optimization object.
            * `u_param` (casadi.MX): parameter related to the command computed by the NN.
            * `x_param` (casadi.MX): parameter related to the system current state.
            * `opti_var` (casadi.MX): optimization variable (command).
        
        Returns
        -------
        result : np.ndarray
            Feasible command resulting from the Feasibility Recovery approach.
        sol : casadi.casadi.OptiSol  
            Object with the optimization solution.
        warm_start : np.ndarray  
            Dictionary with the warm start details.
        """
        # Get optimization object
        opti = feasibility_variables['optimization']

        # Get optimization parameters
        network_command = feasibility_variables['u_param']
        initial_state = feasibility_variables['x_param']

        # Get optimization variable
        u = feasibility_variables['opti_var']

        # Get slack variables
        s = feasibility_variables['slack_variables']

        # Set parameters
        opti.set_value(network_command, u_NN)
        opti.set_value(initial_state, x_init)

        # Set initial guess
        opti.set_initial(u, warm_start['u'][0])
        opti.set_initial(s, warm_start['u'][1:3])
        opti.set_initial(opti.lam_g, warm_start['lam_g'])

        # Bounds over the input control
        LB_U, UB_U = -0.2, 0.2 

        # Solve the optimization problem
        try:
            sol = opti.solve()

            # Warm start
            warm_start = {'u': sol.value(opti.x), 'lam_g': sol.value(opti.lam_g)}

            # Change the result shape
            result = np.array([sol.value(u)], ndmin=2)
        except:
            print("\nError!!!")
            print("Infeasibilities: \n", opti.debug.show_infeasibilities())
            print("Solution found = ", opti.debug.value(u))
            print("Initial guess: ", opti.debug.value(u,opti.initial()))
            print("Return status: ", opti.return_status())
            if opti.debug.value(u) < UB_U and opti.debug.value(u) > LB_U:
                result = np.clip(np.array([opti.debug.value(u)], ndmin=2), LB_U, UB_U)
            else:
                result = np.clip(warm_start['u'][0], LB_U, UB_U)
            print("Control input applied", result)
            warm_start = {'u': np.zeros(3), 'lam_g': np.zeros(18)}
            sol = None
        
        return result, sol, warm_start

    def NN_make_step(X:np.ndarray, model:nn.Module, scalers:dict, x_init:np.ndarray, warm_start:dict, feasibility = None):
        """
        Function to predict the next command of that should be applied to the system a feed-forward NN.

        Parameters
        ----------
        X : np.ndarray
            Array with the current state and command applied to the system.
        model : nn.Module
            Feed-forward neural network used as the controller.
        scalers : dict
            Dictionary containing the scalers of the NN controller.
        x_init : np.ndarray
            Current position of the system.
        feasibility_variables : dict
            Dictionary to get the optimization details:
            - `opti` (casadi.Opti): optimization object.
            - `u_param` (casadi.MX): parameter related to the command computed by the NN.
            - `x_param` (casadi.MX): parameter related to the system current state.
            - `opti_var` (casadi.MX): optimization variable (command).

        Returns
        -------
        output : np.ndarray
            Predicted state of the system.
        sol : casadi.casadi.OptiSol    
            Object with the optimization solution.
        warm_start : np.ndarray  
            Dictionary with the warm start details:
        """
        # Evaluation mode
        model.eval()

        # Prediction
        with torch.no_grad():
            # Scale the input
            X_new = scalers['input'].transform(X)
            ref = scalers['y_dot'].transform(np.array([[X[0,-1]]]))
            X_new[0,-1] = ref[0,-1]

            # Compute the output
            y_star = model(torch.tensor(X_new).float())

            # Unscale the output
            output = scalers['output'].inverse_transform(y_star)

            # Feasibility Recovery
            if feasibility:
                # Get feasible output
                output, sol, warm_start = FeasibilityRecovery.feasibility_recover(output[0,0], x_init[:,0], warm_start, feasibility)
            else:
                sol = 0.0

        return output, sol, warm_start
    
    def forging_model(x: MX, u: MX):
        """
        Function to predict the next command of that should be applied to the system a feed-forward NN.

        Parameters
        ----------
        X casadi.casadi.MX
            Symbolic variable containing the states of the system.
        u casadi.casadi.MX
            Symbolic variable containing the command that will be optimized.

        Returns
        -------
        xdot : casadi.casadi.MX
            Symbolic function containing the expression of state evolution describing the system to
            integrate over time using RK4.
        """
        # Forging variables
        y     = x[0]
        y_dot = x[1]
        p1    = x[2]
        p2    = x[3]
        z     = x[4]

        # Parameters of the hydraulic press
        M  = 90000          # Mass of the moving parts [kg]
        B  = 25000          # Viscous damping coefficient [Ns/m]
        FT = 200000         # Sliding friction force [N]
        D1 = 0.6            # Diameter of the working plunger [m]
        D2 = 0.5            # Diameter of the return plunger [m]
        A1 = np.pi*D1**2/4  # Effective area of the working plunger [m^2]
        A2 = np.pi*D2**2/4  # Effective area of the return plunger [m^2]
        G  = 9.81           # Gravity acceleration [m/s^2]

        # Pressure parameters
        KB   = 22*10**9      # Bulk modulus
        V1_0 = 0.3           # Volume of the chamber in the working cylinders 
        V2_0 = 0.1           # Volume of the chamber in the return cylinders 
        KL_1 = 8*10**(-13)   # Coefficient of external leakage flow in the working cylinders
        KL_2 = 14*10**(-14)  # Coefficient of external leakage flow in the return cylinders

        # Flow rate parameters
        CD  = 0.63           # Valve discharge coefficient 
        RHO = 858            # Density of the oil [kg/m^3]
        D   = 0.006          # Diameter of the control valve hole [m]

        ################# Equations #################

        # Pressure parameters of the hydraulic press
        PS = 32*10**6       # Supply pressure [Pa]
        PT = 101325         # Return pressure [Pa] (1 atm)

        # Geometric parameters
        MU = 0.3            # Coefficient of friction stress
        K  = 1.115          # Deformation strengthening indicator
        W0 = 0.2            # Original width [m]
        H0 = 0.5            # Original height [m]
        B0 = 0.1            # Original bite length [m]

        # Geometric equations
        A  = 0.14 + 0.36*(B0/W0) - 0.054*(B0/W0)**2     # Spreading coefficient (Tomlinson ans Stringer)
        h1 = H0 - y
        w1 = W0*(H0/h1)**A                              # Deformed width [m]
        b1 = B0*(1 + 0.67*(H0/h1*W0/w1 - 1))            # Deformed bite length [m]

        # Characteristics of the part
        T = 900                                         # Deformation temperature [K]   

        # Servo valve parameters
        T1 = 0.005                                      # Time constant of the servo valve   

        # Auxiliary term 
        delta_h = y                                     # Deformation of the part

        # Forging coefficients
        Kd = K*(1 + MU*b1/(2*delta_h) + delta_h/(4*b1)) # Measures the influence of friction and heat transfer on the contact surface of the die
        Ad = w1*b1                                      # Contact surface of the forging and the die for a single forging draft

        ################## Deformation Force ###############

        # Material constants for C45 40 carbon steel
        M0 = 1200*10**6
        M1 = -0.0025
        M2 = -0.0587
        M3 =  0.1165
        M4 = -0.0065   

        # Article strain and strain rate
        e = log(H0/(H0 - y))
        e_dot = y_dot/(H0 - y) 

        # Article Force
        Fd_article = if_else(logic_and(y > 0, y_dot >= 0), Kd*Ad*M0*exp(M1*T)*e**M2*e_dot**M3*exp(M4/e), 0)

        ############ Oil flow in the servo-valves ############

        # Working Cylinders
        qvPB_work = np.pi*D*z*CD*sqrt(2/RHO*fabs(PS - p1))*sign(PS - p1)
        qvAT_work = np.pi*D*z*CD*sqrt(2/RHO*fabs(p2 - PT))*sign(p2 - PT)
        
        # Return Cylinders
        qvPB_return = np.pi*D*z*CD*sqrt(2/RHO*fabs(p1 - PT))*sign(p1 - PT)
        qvAT_return = np.pi*D*z*CD*sqrt(2/RHO*fabs(PS - p2))*sign(PS - p2)

        # Select the oil flow equation according to return and working motions
        qvPB = if_else(z >= 0, qvPB_work, qvPB_return)
        qvAT = if_else(z >= 0, qvAT_work, qvAT_return)

        # Evolution of the volume in the chamber
        V1 = V1_0/2 + A1*y
        V2 = V2_0/2 - A2*y

        # Friction force
        Ft = if_else(fabs(y_dot) <= 0.5, FT*y_dot/0.5, FT)

        # Right-hand side expressions of all states
        rhs_y     = y_dot
        rhs_y_dot = (3*np.pi*D1**2*p1/4 - np.pi*D2**2*p2/2 - B*y_dot - Ft - Fd_article)/M + G
        rhs_p1    = KB/V1*(qvPB/3 - A1*y_dot - KL_1*p1)
        rhs_p2    = KB/V2*(-qvAT/2 + A2*y_dot - KL_2*p2)
        rhs_z     = -z/T1 + u/T1

        # Xdot expression
        xdot = vertcat(rhs_y, rhs_y_dot, rhs_p1, rhs_p2, rhs_z)

        return xdot

    @staticmethod
    def Ruge_Kuta(TS: float, f):
        """
        Function to compute a integrator using the Ruge Kuta method.
        
        Parameters
        ----------
        TS : float 
            Simulation time step.
        f : casadi.casadi.Function
            Function encoding the open-die forging ODE.
        
        Returns
        -------
        F : casadi.casadi.Function   
            Function encoding the open-die forging integrator.
        """
        M = 4
        DT = TS/M

        # Variable declaration
        X0 = MX.sym('X0', 5)
        U = MX.sym('U')
        X = X0

        for idx in range(M):
            # Call the model
            k1 = f(X, U)
            k2 = f(X + DT/2 * k1, U)
            k3 = f(X + DT/2 * k2, U)
            k4 = f(X + DT * k3, U)

            # Add results
            X = X + DT/6*(k1 + 2*k2 + 2*k3 + k4)

        # Integrator Function
        F = Function('F', [X0, U], [X],['x0','u'],['xf'])
        F = F.expand() # Transform MX into SX function

        return F


# ----------------------------------------------------------------
# MPC
# ----------------------------------------------------------------   
class MPC:
    """Class to deal with MPC."""
    @staticmethod
    def loop(N_traj:int, T_traj:float, controller:do_mpc.controller._mpc.MPC, simulator:do_mpc.simulator.Simulator, init_state:dict, timer:do_mpc.tools._timer.Timer, bar_title:str, process_std: np.ndarray, meas_std: np.ndarray):
        """
        Function which implements the closed-loop simulation using a MPC controller.
        
        Parameters
        ----------
        N_traj : int
            Number of complete trajectories (working and return) in the simulation.
        T_traj : float
            Duration of the trajectory.
        controller : do_mpc.controller._mpc.MPC
            MPC controller used in the closed-loop simulation.
        simulator : do_mpc.simulator.Simulator
            Simulator of *do_mpc* acting as the physical system.
        init_state : dict
            Dictionary with the initial states of the system.
        timer : do_mpc.tools._timer.Timer
            Timer that will contain the computational cost of the controller.
        bar_title : str
            Title displayed in the alive bar.
        process_std : np.ndarray
            Standard deviation of the process noise for each state.
        meas_std : np.ndarray
            Standard deviation of the measurement noise for each state.
        
        Returns
        -------
        controller : do_mpc.controller._mpc.MPC
            MPC controller containing the controller's data after the simulation.
        simulator : do_mpc.simulator.Simulator
            Simulator containing the system's data after the simulation.
        MPC_results : dict
            Closed-loop trajectories from the `do_mpc` simulator.
        opt_results : dict
            Optimization indicators from the IPOPT solver.
        timer : do_mpc.tools._timer.Timer
            Timer containing the computational cost of the controller.
        """

        # Empty trajectories for the states command of the system 
        dataset_y     = np.zeros((N_traj, T_traj + 1))
        dataset_y_dot = np.zeros((N_traj, T_traj + 1))
        dataset_p1    = np.zeros((N_traj, T_traj + 1))
        dataset_p2    = np.zeros((N_traj, T_traj + 1))
        dataset_z     = np.zeros((N_traj, T_traj + 1))

        # Empty trajectories for the reference and the command
        dataset_ref = np.zeros((N_traj, T_traj))
        dataset_Fd  = np.zeros((N_traj, T_traj))
        dataset_u   = np.zeros((N_traj, T_traj))

        # Empty optimization trajectory
        dataset_iter = np.zeros((N_traj, T_traj))
        dataset_mu       = np.zeros((N_traj, T_traj))
        dataset_obj      = np.zeros((N_traj, T_traj))
        dataset_reg_size = np.zeros((N_traj, T_traj))
        dataset_d_norm   = np.zeros((N_traj, T_traj))
        dataset_inf_du   = np.zeros((N_traj, T_traj))
        dataset_inf_pr   = np.zeros((N_traj, T_traj))

        # Set initial state
        y_init     = init_state.get('y', 0)
        y_dot_init = init_state.get('y_dot', 0)
        p1_init    = init_state.get('p1', 0)
        p2_init    = init_state.get('p2', 0)
        z_init     = init_state.get('z', 0)

        # Get the range of the inner loop - Simulator
        inner_loop = int(controller.settings.t_step/simulator.settings.t_step)

        # Noise seed
        # np.random.seed(42)

        # Main Loop
        with alive_bar(N_traj, title = bar_title, enrich_print = False) as bar:
            for idx in range(N_traj):

                # Create the initial state vector
                x0 = np.array([y_init, y_dot_init, p1_init, p2_init, z_init]) 

                # Affect initial state to trajectory array
                dataset_y[idx,0]     = y_init
                dataset_y_dot[idx,0] = y_dot_init
                dataset_p1[idx,0]    = p1_init
                dataset_p2[idx,0]    = p2_init
                dataset_z[idx,0]     = z_init

                # Set the initial state
                controller.x0 = x0  # MPC
                simulator.x0  = x0  # Simulator 

                # Set initial guess
                controller.set_initial_guess()  

                # Main loop
                for t in range(T_traj):
                    
                    # Compute the MPC command
                    timer.tic()  # Set timer
                    u0 = controller.make_step(x0)
                    timer.toc()  # Stop timer

                    # Process noise
                    w0 = np.random.normal(loc=0.0, scale=process_std).reshape(-1, 1)

                    # Measurement noise
                    v0 = np.random.normal(loc=0.0, scale=meas_std).reshape(-1, 1)

                    # Inner loop - Simulator
                    for _ in range(inner_loop):
                        x0 = simulator.make_step(u0, v0 = v0, w0 = w0).squeeze()

                    # Add states to trajectory
                    dataset_y[idx,t+1]     = x0[0]
                    dataset_y_dot[idx,t+1] = x0[1]
                    dataset_p1[idx,t+1]    = x0[2]
                    dataset_p2[idx,t+1]    = x0[3]
                    dataset_z[idx,t+1]     = x0[4]

                    # Retrieve reference trajectory       
                    dataset_ref[idx,t] = controller.data['_tvp'][-1,0]
                    dataset_Fd[idx,t]  = controller.data['_aux'][-1,1]
                
                    # Add commands to trajectory
                    dataset_u[idx,t] = u0

                    # Add optimization variables to trajectory
                    # dataset_iter[idx,t]     = controller.solver_stats['iter_count']
                    # dataset_mu[idx,t]       = controller.solver_stats['iterations']['mu'][-1]
                    # dataset_obj[idx,t]      = controller.solver_stats['iterations']['obj'][-1]
                    # dataset_reg_size[idx,t] = controller.solver_stats['iterations']['regularization_size'][-1]
                    # dataset_d_norm[idx,t]   = controller.solver_stats['iterations']['d_norm'][-1]
                    # dataset_inf_du[idx,t]   = controller.solver_stats['iterations']['inf_du'][-1]
                    # dataset_inf_pr[idx,t]   = controller.solver_stats['iterations']['inf_pr'][-1]

                # Set the initial command
                controller.u0 = 0.0  # MPC 
                simulator.u0 = 0.0   # Simulator 
                
                # Update progress bar
                bar()

        MPC_results = {'y' : dataset_y , 'y_dot': dataset_y_dot, 
                       'p1': dataset_p1, 'p2'   : dataset_p2   ,
                       'z' : dataset_z , 'ref'  : dataset_ref  , 
                       'u' : dataset_u , 'F_d'  : dataset_Fd   }
        
        opt_results = {'iter'   : dataset_iter  , 'mu'      : dataset_mu, 
                       'obj'    : dataset_obj   , 'reg_size': dataset_reg_size,
                       'd_norm' : dataset_d_norm, 'inf_du'  : dataset_inf_du, 
                       'inf_pr' : dataset_inf_pr }

        return controller, simulator, MPC_results, opt_results, timer

# ----------------------------------------------------------------
# SAVE NN DATA
# ---------------------------------------------------------------- 
class Save_Network_Data:
    """
    Class to store and manage hyperparameters and configurations for training a neural network.

    Parameters
    ----------
    batch_size : int
        Number of batches.
    n_epochs : int
        Number of epochs .
    input_dim : int
        Number of inputs of the neural network.
    hidden_dim : int
        Number of neurons in the hidden layer.
    width_dim : int
        Number of hidden layers.
    output_dim : int
        Number of outputs to predict.
    learning_rate : float
        Learning rate of the optimizer.
    features : list
        List of string with the features names (inputs of the network).
    target : list
        List of string with the targets names (outputs of the network) . 
    optimizer : torch.optim.Optimizer
        Optimizer used to find the network parameters (e.g., `AdamW`).
    """
    def __init__(self, batch_size:int, n_epochs:int, input_dim:int, hidden_dim:int, width_dim:int, output_dim:int, learning_rate:float, features:list, target:list, optimizer:torch.optim.Optimizer):

        self.batch_size     = batch_size    # Number of batches 
        self.n_epochs       = n_epochs      # Number of epochs 
        self.input_dim      = input_dim     # Number of inputs 
        self.hidden_dim     = hidden_dim    # Number of neurons in the hidden layer
        self.width_dim      = width_dim     # Number of hidden layers
        self.output_dim     = output_dim    # Number of outputs 
        self.learning_rate  = learning_rate # Learning rate
        self.optimizer      = optimizer     # Optimizer
        self.features       = features      # Features
        self.target         = target        # Targets 


# ----------------------------------------------------------------
# GRAPHICS
# ---------------------------------------------------------------- 
class Graphics:
    """Class to deal with graphics and plotting."""

    def plot(data_series:list, axis_options:list, title='', tab_title = '', rows=1, cols=1, 
             subplot_titles=None, show = True, save_fig=False, slider_info=None, 
             add_hover=True, **kwargs):
        """
        Plots multiple graphs with different plot types on subplots using Plotly.

        Parameters
        ----------
        data_series : list of dicts
            List of dictionaries, where each dict represents a dataset with keys `x`, `y`, `type`,
            `name`, `row`, `col`, and optionally `kwargs`.                   
        axis_options : list of dicts
            List of dictionaries, where each dict represents the axis options of a subplot with keys 
            `x_label`, `y_label`, `col`, `rol`, and optionally `x_kwargs` and `y_kwargs`.
        title : str 
            Title of the entire figure.
        tab_title : str
            Title of the HTML tab open in the navigator.
        rows : int
            Number of rows in the subplot grid.
        cols : int 
            Number of columns in the subplot grid.
        subplot_titles : list
            List of titles for the individual subplots.
        show : bool
            Whether to show the plot immediately.
        save_fig : bool
            Whether to save the figure locally.
        **kwargs: 
            Additional layout parameters passed to Plotly.

        Returns
        ----------
        fig : Figure
            Plotly figure object with subplots.
        """
        # Create figure object with subplots
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        # Loop through each data series
        for series in data_series:

            # X-Y Data
            x_data = series['x']
            y_data = series['y']

            # Plot type
            plot_type = series.get('type', 'line')  # Default to 'line' if not specified

            # Name of the series
            name = series.get('name', 'Series') # Default to 'Series' if not specified

            # Rows and Columns 
            row = series.get('row', 1)  # Default to 1 if not specified
            col = series.get('col', 1)  # Default to 1 if not specified
            
            # Other arguments
            trace_kwargs = series.get('kwargs', {})
            
            # Add the trace based on the plot type
            if plot_type == 'line': # Line
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=name, **trace_kwargs), row = row, col = col)
            # Markers (scatter)
            elif plot_type == 'markers':
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=name, **trace_kwargs), row = row, col = col)
            # Bar
            elif plot_type == 'bar':
                fig.add_trace(go.Bar(x=x_data, y=y_data, name=name, **trace_kwargs), row = row, col = col)
            # V-line
            elif plot_type == 'v_line':
                fig.add_vline(x = x_data, line_width=2, line_dash="dash", line_color="black", row = row, col = col)
            # H-line
            elif plot_type == 'h_line':
                fig.add_hline(y = y_data, line_width=2, line_dash="dash", line_color="black", row = row, col = col)
            # Unsupported type
            else:
                raise ValueError(f"Unsupported plot_type: {plot_type}")

        # Set the layout for title, axis labels, etc.
        fig.update_layout(
            title = title,
            font = dict(size=18, color="black"),
            template = "seaborn",
            legend_font = dict(size=16, color="black"),
            **kwargs  # Additional layout settings passed via kwargs
        )

        # Loop through each axis
        for axis in axis_options:

            # Name of the series
            x_label = axis.get('x_label', 'X Axis')
            y_label = axis.get('y_label', 'Y Axis')

            # Rows and Columns 
            row = axis.get('row', 1)  # Default row
            col = axis.get('col', 1)  # Default col

            # Other arguments
            x_axis_kwargs = axis.get('x_kwargs', {})
            y_axis_kwargs = axis.get('y_kwargs', {})

            # Update axis labels
            fig.update_xaxes(title_text = x_label, row = row, col = col, title_font=dict(size=18), tickfont=dict(size=16), **x_axis_kwargs)
            fig.update_yaxes(title_text = y_label, row = row, col = col, title_font=dict(size=18), tickfont=dict(size=16), **y_axis_kwargs)
        
        # Add hover text
        if add_hover:
            fig.update_layout(hovermode = 'x unified')
            fig.update_traces(hoverinfo = 'y+name', hovertemplate = '%{y:.4f}')

        # Change subplot titles font
        if subplot_titles:
            fig.update_annotations(font_size=18)

        # Sliders
        if slider_info:
            # Number of graphics visible initially
            N_traj = slider_info.get('N_traj', 1) 

            N_plots = len(data_series)//N_traj

            # Step definition
            slider_steps = []
            for idx in range(N_traj):
                step = dict(method="update", args=[{"visible": [False] * len(data_series)}], label=str(idx+1))
                
                for num in range(N_plots):
                    step["args"][0]["visible"][N_plots*idx + num] = True

                # Append step
                slider_steps.append(step)

            # Make the first traces visible
            for idx in range (N_plots):
                fig.data[idx].visible = True

            # Defined sliders
            sliders = [dict(active= 0, currentvalue={"prefix": "Trajectory : "}, pad={"t": 50}, steps = slider_steps)]

            # Add slider to the layout
            fig.update_layout(sliders = sliders)

        # Save figure
        if save_fig:
            fig.write_image("results/Images/" + str(tab_title) + ".png")

        # Display the figure if requested
        if show:
            fig.show(renderer="titleBrowser", browser_tab_title = tab_title)

        return fig

# ----------------------------------------------------------------
# CUSTOM PLOTLY IDENTIFIER
# ---------------------------------------------------------------- 
class TitleBrowserRenderer(BrowserRenderer):
    """
    Custom Plotly renderer that sets a custom browser tab title when displaying plots in the browser.
    """
    def __init__(
        self,
        config=None,
        auto_play=False,
        using=None,
        new=0,
        autoraise=True,
        post_script=None,
        animation_opts=None,
    ):
        super().__init__(
            config, auto_play, using, new, autoraise, post_script, animation_opts
        )

    browser_tab_title = "Undefined"

    def render(self, fig_dict):
        from plotly.io import to_html

        html = (
            """
<title>
"""
            + self.browser_tab_title
            + """
</title>
"""
            + to_html(
                fig_dict,
                config=self.config,
                auto_play=self.auto_play,
                include_plotlyjs=True,
                include_mathjax="cdn",
                post_script=self.post_script,
                full_html=True,
                animation_opts=self.animation_opts,
                default_width="100%",
                default_height="100%",
                validate=False,
            )
        )
        open_html_in_browser(html, self.using, self.new, self.autoraise)


renderers["titleBrowser"] = TitleBrowserRenderer()