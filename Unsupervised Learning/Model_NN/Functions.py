# ---------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------

# Create random numbers
import random

# Measure time
from time import time  

# Numpy
import numpy as np 

# DataFrames
import pandas as pd     

# Sklearn 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 

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

# MPC tools
import do_mpc 

# Progress bar
from alive_progress import alive_bar

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
    
    Returns
    -------
    x : torch.Tensor
        Static features (independent variables) for the current timestep.
    y : torch.Tensor
        Target value at the next timestep.
    """
    def __init__(self, dataframe:pd.DataFrame, target:list, features:list):

        """Initialize the dataset."""
        self.features = features
        self.target = target
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.L = len(dataframe)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Return input-output pair for the given index."""
        # Features at index i
        x = self.X[i]       # Input
        
        # Target at next time step
        if i < self.L - 1:
            y = self.y[i+1]
        # Target at last value
        else:
            y = self.y[-1]
        
        # Ensure valid index
        if i >= len(self): raise IndexError # Raise Error
    
        return x, y

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

        """Initialize the dataset."""
        self.features = features
        self.target = target
        self.lookback = lookback
        self.prediction_length = prediction_length
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
        if i >= self.lookback - 1:
            i_start = i - self.lookback + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.lookback - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        
        # Handle target sequence with padding for indices near the end
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
class FNNSimulator(nn.Module):
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
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, width_dim:int, activation_fn=nn.Tanh, bias=False):
        """Method to initialize the feedforward Neural Network.""" 
        super(FNNSimulator, self).__init__()

        # Store parameters
        self.width_dim = width_dim
        self.activation = activation_fn()

        # Store layers
        self.fc_inp = nn.Linear(input_dim, hidden_dim, bias = bias)   # Linear function - Input
        self.fc_int = nn.Linear(hidden_dim, hidden_dim, bias = bias)  # Linear function - Intermediate
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias = bias)  # Linear function - Output
        
        
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
    """Class for handling data preprocessing before training begins."""
    
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
            "minmax"  : MinMaxScaler,
            "standard": StandardScaler,
            "maxabs"  : MaxAbsScaler,
            "robust"  : RobustScaler,
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

        # Loop to concatenate the scaled prediction to the DataFram
        for idx, col in enumerate(target):
            col_name = f"NN({col})"
            prediction_col.append(col_name)

            # Append NN prediction to the DataFrame
            dataframe[col_name] = prediction[:,idx]

            # Shift the commands by 1 (quantity being predicted)
            dataframe = Data.shift_commands(dataframe, col_name, fill_value = 0)

        return dataframe
    
    @staticmethod
    def get_individual_dataset(df:pd.DataFrame, target:list, features:list, t_traj:int, lookback = 1):
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
            datasets_recurrent.append(CreateDataset(data, target=target, features=features, lookback = lookback)) 
            
            # Regular DataSets
            datasets.append(SequenceDataset(data, target=target, features=features)) 

        return datasets, datasets_recurrent

# ----------------------------------------------------------------
# NEURAL NETWORK
# ----------------------------------------------------------------   
class NeuralNetwork:
    """Class to manage feedforward neural networks, including training, validation, prediction, and evaluation."""

    @staticmethod
    def train_model(data_loader:DataLoader, model:nn.Module, loss_function:nn.Module, optimizer:torch.optim.Optimizer, device:torch.device):
        """
        Train the neural network model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        model : nn.Module
            Feed-forward neural network used to predict the next command.
        loss_function : nn.Module
            Loss function used in the backpropagation.
        optimizer : torch.optim.Optimizer
            Optimizer used to find the network parameters (e.g., `AdamW`).
        device : torch.device
            Device in which training is performed (`cuda:0` or `cpu`).

        Returns
        -------
        total_loss : float
            Average training loss over all batches.
        """
        model.train()   # Set model to training mode
        total_loss = 0  # Initialize cumulative loss

        # Training loop       
        for X, y in data_loader: 

            # Move inputs and targets to the device
            X, y = X.to(device), y.to(device)

            # Reset gradients
            optimizer.zero_grad()  

            # Forward pass    
            output = model(X, device)      

            # Compute loss 
            loss = loss_function(output, y.squeeze())  

            # Backpropagation
            loss.backward()  

            # Update weights
            optimizer.step()  

            # Accumulate loss
            total_loss += loss.item()  

        return total_loss / len(data_loader) # Average loss
    
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
            for X, y in data_loader: 

                # Move inputs and targets to the device
                X, y = X.to(device), y.to(device)

                # NN prediction
                output = model(X, device)  
                
                # Compute loss
                total_loss += loss_function(output, y.squeeze()).item()

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
            Feed-forward neural network used to predict the next state.

        Returns
        -------
        output : torch.Tensor
            Concatenated predictions for all batches.
        """
        model.eval()  # Set model to evaluation mode
        predictions = []

        # Prediction loop
        with torch.no_grad():
            for X, _ in data_loader:  
                predictions.append(model(X, "cpu"), )  # Collect predictions 

        # Concatenate along the batch dimension
        output = torch.cat(predictions, dim=0) 

        return output 
    
    @staticmethod
    def metrics(truth: np.ndarray, out:np.ndarray, results_dict: dict = None):
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
        results = {'mae' : mean_absolute_error(truth, out),
                   'rmse': root_mean_squared_error(truth, out),
                   'r2'  : r2_score(truth, out)
        }

        # Print results
        logger.info(f"- Mean Average Error (MAE) = {results['mae']:.4f}.")
        logger.info(f"- Root Mean Squared Error (RMSE) = {results['rmse']:.4f}.")
        logger.info(f"- R2 Coefficient = {results['r2']:.4f}.")

        # Append results
        if results_dict:
            results_dict['MAE'].append(results['mae'])
            results_dict['RMSE'].append(results['rmse'])
            results_dict['R2'].append(results['r2'])

    @staticmethod
    def other_metrics(input: np.ndarray, timer:np.ndarray, results_dict: dict = None):
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

        # Append results
        if results_dict:
            results_dict['Command'].append(np.mean(np.abs(input)))
            results_dict['Mean_time'].append(np.mean(timer)*1000)
            results_dict['Std_time'].append(np.std(timer)*1000)
            results_dict['Median_time'].append(np.percentile(timer, 50)*1000)
            results_dict['25_perc'].append(np.percentile(timer, 25)*1000)
            results_dict['75_perc'].append(np.percentile(timer, 75)*1000)

    @staticmethod
    def NN_make_step(X:np.ndarray, model:nn.Module, scalers:dict):
        """
        Function to predict the next state of the open-die forging process using a neural network.
        
        Parameters
        ----------
        X : np.ndarray
            Input for the neural network.
        model : nn.Module
            Feed-forward neural network to predict the future states of the system.
        scalers : dict 
            Dictionary containing the scalers of the feed-forward network.
        
        Returns
        -------
        output : np.ndarray    
            Feasible command.
        """

        # Evaluation mode
        model.eval()

        # Prediction
        with torch.no_grad():

            # Compute the output
            y_star = model(torch.tensor(X).float(), "cpu")

            # Unscale the output
            output = scalers['output'].inverse_transform(y_star)
            
        return output
    
    @staticmethod
    def train_loop(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, loss_function:nn.Module, optimizer:torch.optim.Optimizer, n_epochs:int, device:torch.device):
        """
        Loop to training a neural network using undupervised learning.

        Parameters
        ----------
        model : nn.Module
            Feed-forward neural network used to predict the next state.
        train_loader : DataLoader
            DataLoader for training the feed-froward network.
        val_loader : DataLoader 
            DataLoader for validation during training.
        loss_function : nn.Module 
            Loss function used in the backpropagation.
        optimizer : torch.optim.Optimizer 
            Optimizer used to find the network parameters (e.g., `AdamW`).
        n_epochs : int 
            Number of times that all data points in the dataset have passed through the optimization.
        device : torch.device
            Device in which training is performed (`cuda:0` or `cpu`).

        Returns
        -------
        model : nn.Module
            Feed-forward neural network with the updated weights.
        vec_t_loss : list
            List containing the average training loss at each epoch.
        vec_v_loss : list
            List containing the average validation loss at each epoch.
        comp_time : float
            Time taken to train the neural network.
        """
        logger.info("\n-------- \nTRAINING - MODEL \n--------")
        validation_loss = NeuralNetwork.validate_model(val_loader, model, loss_function, device)

        # Variables initialization
        vec_t_loss, vec_v_loss = [], []

        # Set timer 
        t0 = time()

        # Training loop
        with alive_bar(n_epochs, title='TRAIN', enrich_print = False) as bar:
            for ix_epoch in range(n_epochs):
                
                # Training function
                training_loss = NeuralNetwork.train_model(train_loader, model, loss_function, optimizer, device)

                # Validation loss
                validation_loss = NeuralNetwork.validate_model(val_loader, model, loss_function, device)
                
                # Print results
                if ix_epoch % (n_epochs/10) == 0:
                    logger.info(f"\n[{100*ix_epoch/n_epochs:.1f}%] Training loss: {training_loss:.4f},  Validation loss: {validation_loss:.4f}")
                elif ix_epoch == n_epochs-1:
                    logger.info(f"\n[100%] Training loss: {training_loss:.4f},  Validation loss: {validation_loss:.4f}")
                
                # Append losses
                vec_t_loss.append(training_loss)
                vec_v_loss.append(validation_loss)

                # Update progress bar
                bar()   

        # Computational time
        comp_time = time() - t0
        logger.info(f"\nTotal time: {comp_time:.2f}s.")

        return model, vec_t_loss, vec_v_loss, comp_time
    
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
# MPC
# ----------------------------------------------------------------   
class MPC:
    """Class to deal with MPC."""

    @staticmethod
    def loop(N_traj:int, T_traj:float, controller:do_mpc.controller._mpc.MPC, simulator:do_mpc.simulator.Simulator, init_state:dict, timer:do_mpc.tools._timer.Timer, bar_title:str, process_std: np.ndarray, meas_std: np.ndarray, model:nn.Module, scalers:dict, lookback = 1):
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
        model : nn.Module
            Feed-forward neural network to predict the future states of the system.
        scalers : dict
            Dictionary containing the scalers of the feed-forward network.
        lookback : int
            Lookback window size for recurrent datasets.
        
        Returns
        -------
        controller : do_mpc.controller._mpc.MPC
            MPC controller containing the controller's data after the simulation.
        simulator : do_mpc.simulator.Simulator
            Simulator containing the system's data after the simulation.
        MPC_results : dict
            Closed-loop trajectories from the `do_mpc` simulator.
        opt_results : dict
            Optmization indicators from the IPOPT solver.
        timer : do_mpc.tools._timer.Timer
            Timer containing the computational cost of the controller.
        """

        # Empty trajectories - States (Simulator)
        dataset_y     = np.zeros((N_traj, T_traj + 1))
        dataset_y_dot = np.zeros((N_traj, T_traj + 1))
        dataset_p1    = np.zeros((N_traj, T_traj + 1))
        dataset_p2    = np.zeros((N_traj, T_traj + 1))
        dataset_z     = np.zeros((N_traj, T_traj + 1))

        # Empty trajectories - States (Neural Network)
        dataset_y_dot_NN = np.zeros((N_traj, T_traj + 1))
        dataset_p1_NN    = np.zeros((N_traj, T_traj + 1))
        dataset_p2_NN    = np.zeros((N_traj, T_traj + 1))
        dataset_z_NN     = np.zeros((N_traj, T_traj + 1))

        # Other empty trajectories
        dataset_ref = np.zeros((N_traj, T_traj))
        dataset_Fd  = np.zeros((N_traj, T_traj))
        dataset_u   = np.zeros((N_traj, T_traj))

        # Empty optimization trajectory
        dataset_iter     = np.zeros((N_traj, T_traj))
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
                x0_NN = np.array([x0[1],x0[2],x0[3],x0[4]])

                # Affect initial state to trajectory array (Simulator)
                dataset_y[idx,0]     = y_init
                dataset_y_dot[idx,0] = y_dot_init
                dataset_p1[idx,0]    = p1_init
                dataset_p2[idx,0]    = p2_init
                dataset_z[idx,0]     = z_init

                # Affect initial state to trajectory array (Neural Network)
                dataset_y_dot_NN[idx,0] = y_dot_init
                dataset_p1_NN[idx,0]    = p1_init
                dataset_p2_NN[idx,0]    = p2_init
                dataset_z_NN[idx,0]     = z_init

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

                    # Simulator - MPC
                    for _ in range(inner_loop):
                        x0 = simulator.make_step(u0, v0 = v0, w0 = w0).squeeze()

                    # Generate LSTM input
                    if t == 0:
                        x0_NN = np.concatenate((x0_NN, u0[0]))
                        x0_NN = np.expand_dims(x0_NN, axis=0)
                        x0_NN = scalers['input'].transform(x0_NN)
                        x0_NN = np.repeat(x0_NN, lookback, axis=0)
                        input_simulator = np.expand_dims(x0_NN, axis=0)
                    else:
                        x_next_NN = np.concatenate((x_next_NN, u0[0]))
                        x_next_NN = np.expand_dims(x_next_NN, axis=0)
                        x_next_NN = scalers['input'].transform(x_next_NN)
                        x_next_NN = np.expand_dims(x_next_NN, axis=0)
                        input_simulator = input_simulator[:,1:lookback,:]
                        input_simulator = np.concatenate((input_simulator, x_next_NN), axis=1)

                    # Simulator - NN
                    x_next_NN = NeuralNetwork.NN_make_step(input_simulator, model, scalers).squeeze()

                    # Add states to trajectory (Simulator)
                    dataset_y[idx,t+1]     = x0[0]
                    dataset_y_dot[idx,t+1] = x0[1]
                    dataset_p1[idx,t+1]    = x0[2]
                    dataset_p2[idx,t+1]    = x0[3]
                    dataset_z[idx,t+1]     = x0[4]

                    # Add states to trajectory (Neural Network)
                    dataset_y_dot_NN[idx,t+1] = x_next_NN[0]
                    dataset_p1_NN[idx,t+1]    = x_next_NN[1]
                    dataset_p2_NN[idx,t+1]    = x_next_NN[2]
                    dataset_z_NN[idx,t+1]     = x_next_NN[3]

                    # Retrieve reference trajectory       
                    dataset_ref[idx,t] = controller.data['_tvp'][-1,0]
                    dataset_Fd[idx,t]  = controller.data['_aux'][-1,1]
                
                    # Add commmands to trajectory
                    dataset_u[idx,t] = u0

                    # Add optimization variables to trajectory
                    dataset_iter[idx,t]     = controller.solver_stats['iter_count']
                    dataset_mu[idx,t]       = controller.solver_stats['iterations']['mu'][-1]
                    dataset_obj[idx,t]      = controller.solver_stats['iterations']['obj'][-1]
                    dataset_reg_size[idx,t] = controller.solver_stats['iterations']['regularization_size'][-1]
                    dataset_d_norm[idx,t]   = controller.solver_stats['iterations']['d_norm'][-1]
                    dataset_inf_du[idx,t]   = controller.solver_stats['iterations']['inf_du'][-1]
                    dataset_inf_pr[idx,t]   = controller.solver_stats['iterations']['inf_pr'][-1]
                
                # Set the initial command
                controller.u0 = 0.0  # MPC 
                simulator.u0 = 0.0   # Simulator 
                
                # Update progress bar
                bar()

        MPC_results = {'y' : dataset_y , 'y_dot': dataset_y_dot, 
                       'p1': dataset_p1, 'p2'   : dataset_p2   ,
                       'z' : dataset_z , 'ref'  : dataset_ref  , 
                       'u' : dataset_u , 'F_d'  : dataset_Fd   }
        
        NN_results = {'y_dot': dataset_y_dot_NN, 'p1': dataset_p1_NN, 
                      'p2'   : dataset_p2_NN   , 'z' : dataset_z_NN }
        
        opt_results = {'iter'   : dataset_iter  , 'mu'      : dataset_mu, 
                       'obj'    : dataset_obj   , 'reg_size': dataset_reg_size,
                       'd_norm' : dataset_d_norm, 'inf_du'  : dataset_inf_du, 
                       'inf_pr' : dataset_inf_pr }
        

        return controller, simulator, MPC_results, NN_results, opt_results, timer
    
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

    def plot(data_series, axis_options, title='', tab_title = '', rows=1, cols=1, 
             subplot_titles=None, show = True, save_fig = False, slider_info = None, 
             add_hover = True, **kwargs):
        """
        Plots multiple graphs with different plot types on subplots using Plotly.

        Parameters
        ----------
        data_series : list of dicts
            List of dictionaries, where each dict represents a dataset with keys `x`, `y`, `type`,
            `name`, `row`, `col`, and optionally `kwargs`.                   
        axis_options : list of dicts
            List of dictionaries, where each dict represents the axis options of a subplot with keys 
            `x_label`, `y_label`, `col`, `rol`, and optionnaly `x_kwargs` and `y_kwargs`.
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


