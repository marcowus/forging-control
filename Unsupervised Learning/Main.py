# ---------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------

# Standard Library Imports
import pickle         

# Third-Party Library Imports
import numpy  as np                                   # Arrays
import pandas as pd                                   # DataFrames
from sklearn.model_selection import train_test_split  # Split Data                    

# PyTorch 
import torch                             # Tensors
import torch.nn as nn                    # Neural Network
from torch.utils.data import DataLoader  # DataLoaders

# Local Imports
from Functions import (Data, NeuralNetwork, MPC, FNNModel, LSTMModel, MPCLoss, FeasibilityRecovery, Graphics)

# Notification
from notifypy import Notify

# do-mpc    
import do_mpc
from do_mpc.tools       import Timer           
from template_model     import template_model          
from template_mpc       import template_mpc       
from template_simulator import template_simulator  

# Casadi - Optimization
from casadi import *

# Print information
import logging

# Get the device in which the network is trained
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0)} ({device})")

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
# USER SETTINGS
# ----------------------------------------------------------------
# Debug
enable_debug = False

# MPC flags
enable_mpc     = True    # MPC flag
silence_solver = True    # Silence IPOPT solver
enable_noise   = False    # Noise flag

# Neural Network 
enable_training    = True   # Training flag
enable_feasibility = False   # Feasibility Recovery flag

# Graphics 
show_plots      = False   # Graphics flag
show_comparison = True    # Comparison flag

# ---------------------------------------------------------------
# SIMULATION PARAMETERS
# ----------------------------------------------------------------

T_TRAJ = 300   # Trajectory period
N_TRAJ = 15    # Number of trajectories (15)
N_SIM  = 1     # Number of closed-loop simulations

# Hyperparameters
TOTAL_BATCH_SIZE = 150          # Total batch size
N_EPOCHS         = 150          # Number of epochs
LEARNING_RATE    = 0.0001       # Learning rate

# Process noise std per state
if enable_noise:
    process_std = np.array([
        5e-1,     # Displacement (y)
        2e-0,     # Speed (y_dot) 
        5e7,      # Pressure (p1) 
        5e7,      # Pressure (p2) 
        2e-0      # Spool valve displacement (z) 
    ])
else:
    process_std = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# File names
if enable_noise:
    add_to_file = f'_noise'
else:
    add_to_file = f''

if enable_feasibility:
    add_feas_to_file = f'_feasible'
else:
    add_feas_to_file = f''

# No measurement noise
meas_std = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Command constraints
LB_U, UB_U = -0.2, 0.2 

# State constraints
LB_P1   , UB_P1    = 0.0, 32*10**6
LB_P2   , UB_P2    = 0.0, 32*10**6

# Declare empty lists
test_results = {'MAE': [], 'RMSE': [], 'R2': [], 'Train_time': []}

NN_results = {'MAE': [], 'RMSE': [], 'R2': [], 'Command': [], 'Mean_time': [], 'Std_time': [], 'Median_time': [], '25_perc': [], '75_perc': []}

MPC_results = {'MAE': [], 'RMSE': [], 'R2': [], 'Command': [], 'Mean_time': [], 'Std_time': [], 'Median_time': [], '25_perc': [], '75_perc': []}

for sim in range(N_SIM):

    # ---------------------------------------------------------------
    # CONTROLLER - MPC
    # ----------------------------------------------------------------
    model_MPC      = template_model()                         
    controller_MPC = template_mpc(model_MPC, silence_solver, enable_feasibility) 
    sim_MPC        = template_simulator(model_MPC)           

    # ---------------------------------------------------------------
    # MODEL - LSTM
    # ----------------------------------------------------------------
    with open(f"Model_NN/results/NN_model_data.bin", "rb") as f: 
        NN_data = pickle.load(f)     

    # Model Definition
    simulator_LSTM = LSTMModel(NN_data.input_dim, NN_data.hidden_dim, NN_data.output_dim, NN_data.width_dim)

    model_scalers = {
        'input':     pickle.load(open(f'Model_NN/results/scaler_model_input' + add_to_file + '.pkl' , 'rb')),
        'output':    pickle.load(open(f'Model_NN/results/scaler_model_output' + add_to_file + '.pkl', 'rb'))
    }

    # Get parameters (weights and bias) from a trained network
    simulator_LSTM.load_state_dict(torch.load(f'Model_NN/results/model_NN' + add_to_file + '.pt'))

    # ---------------------------------------------------------------
    # CONTROLLER - NN
    # ----------------------------------------------------------------

    # Inputs and outputs - Controller
    features = ['y_dot','z','ref'] 
    target   = ['u']               

    # Inputs - Model
    recurrent_features = ['y_dot', 'p1', 'p2', 'z', 'u']

    # NN Parameters
    INPUT_DIM  = len(features)    # Number of inputs 
    WIDTH_DIM  = 1                # Number of hidden layers
    HIDDEN_DIM = 50               # Number of neurons in the hidden layer
    OUTPUT_DIM = len(target)      # Number of outputs

    # Model Definition
    controller_NN = FNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, WIDTH_DIM, nn.ReLU, bias = True)  

    # Loss Function
    N = controller_MPC.settings.n_horizon # Prediction horizon
    loss_function = MPCLoss(prediction_horizon = N, alpha = 20.0)

    # Optimizer
    optimizer = torch.optim.AdamW(controller_NN.parameters(), lr = LEARNING_RATE)

    " ----------------------------------------------------------- "
    "                      DATA PREPARATION                       "
    " ----------------------------------------------------------- "
    with open(f'../Data/forging_mult_traj_process_noise_N_{N}.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # States 
    y_dot = results['mpc']._x[:,1]    # Deformation rate [m/s]
    ref   = results['mpc']._tvp[:,0]  # Reference [m/s]
    p1    = results['mpc']._x[:,2]    # Pressure in working cylinders [Pa]
    p2    = results['mpc']._x[:,3]    # Pressure in return cylinders [Pa]
    z     = results['mpc']._x[:,4]    # Opening of the servo-valve [m]
    u     = results['mpc']._u[:,0]    # Input action [ad]

    # Time
    mpc_time = results['mpc']._time.squeeze()

    # Pandas Dataframe
    df = pd.DataFrame({'y_dot': y_dot, 'p1': p1, 'p2': p2, 'z': z, 'ref': ref, 'u': u})

    # ----------------------------------------------------------------
    # SPLIT DATA
    # ----------------------------------------------------------------

    # Ratio: 60% for training, 20% for validation and 20% for test
    TRAIN_PROPORTION = 0.6 
    VAL_PROPORTION   = 0.2   

    # Split data
    df_train, temp  = train_test_split(df, test_size = 1-TRAIN_PROPORTION, shuffle = False)
    df_val, df_test = train_test_split(temp, test_size = VAL_PROPORTION/(1 - TRAIN_PROPORTION), shuffle=False)

    # ----------------------------------------------------------------
    # SCALE DATA
    # ----------------------------------------------------------------
    pd.options.mode.chained_assignment = 'warn'  # default='warn'

    # Make a dictionary of scalers
    scalers = {
        'dataframe': Data.get_scaler('maxabs'),
        'input':     Data.get_scaler('maxabs'),
        'output':    Data.get_scaler('maxabs'),
        'y_dot':     Data.get_scaler('maxabs'),
    }

    # Fit data
    scalers['dataframe'].fit(df_train)
    scalers['dataframe'].set_output(transform="pandas")

    # Standardize data to 0 mean and 1 std
    df_train_scaled = scalers['dataframe'].transform(df_train)
    df_val_scaled   = scalers['dataframe'].transform(df_val)
    df_test_scaled  = scalers['dataframe'].transform(df_test)

    # Fit 'input' and 'output' scalers
    scalers['input'].fit(df_train[features].values)
    scalers['output'].fit(df_train[target].values)

    # Get "speed" scaler
    scalers['y_dot'].fit(df_train[['y_dot']].values)

    # Scale the 'Reference' using the 'y_dot' scaler
    df_train_scaled[['ref']] = scalers['y_dot'].transform(df_train[['ref']].values)
    df_val_scaled[['ref']]   = scalers['y_dot'].transform(df_val[['ref']].values)
    df_test_scaled[['ref']]  = scalers['y_dot'].transform(df_test[['ref']].values)

    # print(scalers['dataframe'].transform(np.array([[0.2, 32*10**6, 32*10**6, 0.2, 0.2, 0.2]])))

    # ----------------------------------------------------------------
    # DATASETS
    # ----------------------------------------------------------------
    lookback = 10 # Lookback for recurrent networks

    # Get list with individual datasets for feed-forward and recurrent networks
    dataset_train, _ = Data.get_individual_dataset(df_train_scaled, target, features, recurrent_features, T_TRAJ, lookback)
    dataset_val  , _ = Data.get_individual_dataset(df_val_scaled  , target, features, recurrent_features, T_TRAJ, lookback)
    dataset_test , _ = Data.get_individual_dataset(df_test_scaled , target, features, recurrent_features, T_TRAJ, lookback)

    # Concatenate the individual datasets
    datasets = {
        'train': torch.utils.data.ConcatDataset(dataset_train),
        'val':   torch.utils.data.ConcatDataset(dataset_val),
        'test':  torch.utils.data.ConcatDataset(dataset_test)
    }

    # Resample the dataset
    index_list_train = list(range(0, len(datasets['train']), N))
    index_list_val   = list(range(0, len(datasets['val'])  , N))
    index_list_test  = list(range(0, len(datasets['test']) , N))

    # Resample the dataset
    datasets_resempled = {
        'train': torch.utils.data.Subset(datasets['train'], index_list_train),
        'val':   torch.utils.data.Subset(datasets['val']  , index_list_val),
        'test':  torch.utils.data.Subset(datasets['test'] , index_list_test)
    }

    # ----------------------------------------------------------------
    # DATALOADERS
    # ----------------------------------------------------------------
    BATCH_SIZE = TOTAL_BATCH_SIZE//N # Batch size for training and validation
    
    loaders = {
        'train': DataLoader(datasets['train'], batch_size = BATCH_SIZE, shuffle = True),
        'val':   DataLoader(datasets['val']  , batch_size = BATCH_SIZE, shuffle = False),
        'test':  DataLoader(datasets['test'] , batch_size = BATCH_SIZE, shuffle = False)
    }

    loaders_resampled = {
        'train': DataLoader(datasets_resempled['train'], batch_size = BATCH_SIZE, shuffle = True),
        'val':   DataLoader(datasets_resempled['val']  , batch_size = BATCH_SIZE, shuffle = False),
        'test':  DataLoader(datasets_resempled['test'] , batch_size = BATCH_SIZE, shuffle = False)
    }

    # Loader length
    logger.info("\n-------- \nDATA \n--------")
    logger.info(f"Length of train set: {len(datasets_resempled['train'])}")
    logger.info(f"Length of validation set: {len(df_val)}")
    logger.info(f"Length of test set: {len(df_test)}")

    # Data shape
    X, y, z = next(iter(loaders['train']))
    logger.info(f"\nFeatures shape: {X.shape}")
    logger.info(f"Recurrent features shape: {z.shape}")
    logger.info(f"Target shape: {y.shape}")

    " ----------------------------------------------------------- "
    "                        TRAINING LOOP                        "
    " ----------------------------------------------------------- "
    if enable_training:
        # Send Model to GPU if available
        controller_NN  = controller_NN.to(device) 
        simulator_LSTM = simulator_LSTM.to(device) 

        # Train Loop
        controller_NN, vec_t_loss, vec_v_loss, comp_time, loss_features = NeuralNetwork.train_loop(controller_NN, simulator_LSTM, loaders_resampled['train'], loaders['val'], loss_function, optimizer, N_EPOCHS, device, enable_noise)

        # Get the loss features
        loss_matrix     = loss_features['loss']
        command_loss    = loss_features['command']
        error_loss      = loss_features['error']
        prediction_loss = loss_features['prediction']

        # Save network weights
        torch.save(controller_NN.state_dict(), f'results/NN_controller_N_{N}_{sim}' + add_to_file + '.pt')

        # Get training time
        test_results['Train_time'].append(comp_time)

        # Send Model back to CPU
        controller_NN = controller_NN.cpu()
        simulator_LSTM = simulator_LSTM.cpu()
    else:
        # Get parameters (weights and bias) from a trained network
        controller_NN.load_state_dict(torch.load(f'results/NN_controller_N_{N}_{sim}' + add_to_file + '.pt'))

        # Get training time
        test_results['Train_time'].append(0.0)

    # ----------------------------------------------------------------
    # TESTING NETWORK
    # ----------------------------------------------------------------
    df_train, df_train_scaled = NeuralNetwork.test(loaders['train'], df_train, df_train_scaled, controller_NN, target, scalers)
    df_val  , df_val_scaled   = NeuralNetwork.test(loaders['val']  , df_val  , df_val_scaled  , controller_NN, target, scalers)
    df_test , df_test_scaled  = NeuralNetwork.test(loaders['test'] , df_test , df_test_scaled , controller_NN, target, scalers)

    # ----------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------
    logger.info("\n-------- \nTEST RESULTS \n--------")

    # Loop to concatenate the scaled prediction to the DataFrame
    prediction_col = []
    for _, col in enumerate(target):
        prediction_col.append(f"NN({col})")

    # Extract the reference and predictions from the DataFrame
    y_true     = df_test_scaled[target].to_numpy()
    y_forecast = df_test_scaled[prediction_col].to_numpy()

    # Metrics 
    NeuralNetwork.metrics(y_true, y_forecast, test_results)

    # ----------------------------------------------------------------
    # PLOT - LOSS
    # ----------------------------------------------------------------
    if enable_training and show_plots:

        # Time definition
        train_time = np.linspace(0, N_EPOCHS, N_EPOCHS)*LEARNING_RATE
        min_loss_idx = np.argmin(vec_v_loss)*LEARNING_RATE

        # Create empty list to store the data 
        data = []

        # Training loss
        data.append({'x': train_time, 'y': vec_t_loss, 'row': 1, 'col': 1, 'name': 'Training', 'kwargs': {'line': dict(color='green', width=2)}})
        
        # Validation loss
        data.append({'x': train_time, 'y': vec_v_loss,'row': 1, 'col': 1,  'name': 'Validation', 'kwargs': {'line': dict(color='red', width=2)}})
        
        # Minimum of validation loss
        data.append({'x': min_loss_idx, 'y': None, 'row': 1, 'col': 1, 'type': 'v_line'})
        
        # Training loss (log)
        data.append({'x': train_time, 'y': vec_t_loss, 'row': 2, 'col': 1, 'name': 'Log training', 'kwargs': {'line': dict(color='green', width=2), 'showlegend': False}})
        
        # Validation loss (log)
        data.append({'x': train_time, 'y': vec_v_loss, 'row': 2, 'col': 1, 'name': 'Log validation', 'kwargs': {'line': dict(color='red', width=2), 'showlegend': False}})
        
        # Minimum of validation loss (log)
        data.append({'x': min_loss_idx, 'y': None , 'row': 2, 'col': 1, 'type': 'v_line'})

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Loss'    , 'row': 1, 'col': 1},
                {'x_label': 'Time [s]', 'y_label': 'Log Loss', 'row': 2, 'col': 1, 'y_kwargs': {'type': 'log'}}]

        # Graphic
        Graphics.plot(data, axis, rows = 2, cols = 1, tab_title = 'Loss', height = 1000, subplot_titles = ["Loss", "Logarithmic Loss"])

        
    # ----------------------------------------------------------------
    # PLOT - TEST
    # ----------------------------------------------------------------
    if enable_training and show_plots:
        # Create empty list to store the data 
        data = []

        # MPC command
        data.append({'x': mpc_time, 'y': df_test.u, 'name': 'u<sub>MPC</sub>', 'kwargs': {'line': dict(color='blue', width=2), 'line_shape': 'hvh'}})

        # NN command
        data.append({'x': mpc_time, 'y': df_test['NN(u)'], 'name': 'u<sub>NN</sub>', 'kwargs': {'line': dict(color='red' , width=2), 'line_shape': 'hvh'}})

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Command'}]

        # Graphic
        Graphics.plot(data, axis, title = f"Test (N = {N})", tab_title = 'Test', height = 1100)

    " ----------------------------------------------------------- "
    "                       MPC SIMULATION                        "
    " ----------------------------------------------------------- "
    # Set a timer
    timer = Timer()

    # MPC Time Step
    TS = controller_MPC.settings.t_step

    # Set initial states
    y_init     = 0.0
    y_dot_init = 0.0      
    p1_init    = 2156275.6006012624
    p2_init    = 2961363.827545376
    z_init     = 0.0 

    init_state = {'y' : y_init, 'y_dot': y_dot_init, 'p1': p1_init , 'p2': p2_init, 'z': z_init}

    # MPC simulation
    if enable_mpc:
        logger.info("\n-------- \nMPC SIMULATION \n--------")

        # Run the MPC loop to generate the dataset
        controller_MPC, sim_MPC, MPC_data, opt, timer = MPC.loop(N_traj = N_TRAJ, T_traj = T_TRAJ, 
                                                                 controller  = controller_MPC, 
                                                                 simulator   = sim_MPC, 
                                                                 init_state  = init_state,
                                                                 timer       = timer,
                                                                 bar_title   = 'MPC',
                                                                 process_std = process_std,
                                                                 meas_std    = meas_std)

        # Store results
        file       = f'forging_MPC_N_{N}' + add_feas_to_file + add_to_file
        do_mpc.data.save_results([sim_MPC], file, overwrite = True)

        # ----------------------------------------------------------------
        # RESULTS
        # ----------------------------------------------------------------
        logger.info("\n-------- \nMPC RESULTS \n--------")

        # Retrieve results
        sim_MPC = sim_MPC.data

        # Compute metrics
        NeuralNetwork.metrics(sim_MPC['_tvp'][:,0], sim_MPC['_y'][:,1], MPC_results)
        NeuralNetwork.other_metrics(sim_MPC['_u'][:,0], timer.t_list, MPC_results)

    # ----------------------------------------------------------------
    # PLOT - MPC
    # ----------------------------------------------------------------
    if enable_mpc and show_plots:

        # Retrieve time variable
        sim_time = sim_MPC['_time'][:,0]

        # Create empty list to store the data 
        data = []

        # Position
        data.append({'x': sim_time, 'y': sim_MPC['_y'][:,0], 'row': 1, 'col': 1, 'name': 'x', 'kwargs': {'line': dict(color='burlywood', width=2)}})
        
        # Speed reference
        data.append({'x': sim_time, 'y': sim_MPC['_tvp'][:,0], 'row': 1, 'col': 2, 'name': 'Ref', 'kwargs': {'line': dict(color='green', width=2), 'line_shape': 'hvh'}})
        
        # Speed
        data.append({'x': sim_time, 'y': sim_MPC['_y'][:,1], 'row': 1, 'col': 2, 'name': 'v', 'kwargs': {'line': dict(color='orange', width=2)}})
        
        # Pressure - p1
        data.append({'x': sim_time, 'y': sim_MPC['_y'][:,2], 'row': 2, 'col': 1, 'name': 'p1', 'kwargs': {'line': dict(color='red', width=2)}})
        
        # Pressure - p2
        data.append({'x': sim_time, 'y': sim_MPC['_y'][:,3], 'row': 2, 'col': 2, 'name': 'p2', 'kwargs': {'line': dict(color='blue', width=2)}})
        
        # Valve displacement
        data.append({'x': sim_time, 'y': sim_MPC['_y'][:,4], 'row': 3, 'col': 1, 'name': 'z', 'kwargs': {'line': dict(color='chocolate', width=2)}})
        
        # Command
        data.append({'x': sim_time, 'y': sim_MPC['_u'][:,0], 'row': 3, 'col': 2, 'name': 'u', 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh'}})

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Position [m]'    , 'row': 1, 'col': 1},
                {'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'     , 'row': 1, 'col': 2},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row': 2, 'col': 1},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row': 2, 'col': 2},
                {'x_label': 'Time [s]', 'y_label': 'Displacement [m]', 'row': 3, 'col': 1},
                {'x_label': 'Time [s]', 'y_label': 'Command [ad]'    , 'row': 3, 'col': 2}]

        # Graphic
        Graphics.plot(data, axis, rows = 3, cols = 2, tab_title = 'MPC', height = 1100,
                      subplot_titles = ["Deformation", "Deformation Speed", "Pressure (p1)", "Pressure (p2)", "Valve Displacement", "Command"])

    # ----------------------------------------------------------------
    # PLOT - MPC + SLIDER
    # ----------------------------------------------------------------
    mpc_time = controller_MPC.data['_time'][:,0]

    if enable_mpc and show_plots:
        # Create empty list to store the data 
        data = []

        # Loop over the references 
        for idx in range (N_TRAJ): 
            
            # Position
            data.append({'x': mpc_time, 'y': MPC_data['y'][idx], 'row': 1, 'col': 1, 'name': 'x', 'kwargs': {'line': dict(color='burlywood', width=2), 'visible': False}})
            
            # Speed
            data.append({'x': mpc_time, 'y': MPC_data['y_dot'][idx], 'row': 1, 'col': 2, 'name': 'v', 'kwargs': {'line': dict(color='orange', width=2), 'visible': False}})
            
            # Reference
            data.append({'x': mpc_time, 'y': MPC_data['ref'][idx], 'row': 1, 'col': 2, 'name': 'v<sub>ref</sub>', 'kwargs': {'line': dict(color='green', width=2), 'line_shape': 'hvh', 'visible': False}})
            
            # Pressure (p1)
            data.append({'x': mpc_time, 'y': MPC_data['p1'][idx], 'row': 2, 'col': 1, 'name': 'p<sub>1</sub>', 'kwargs': {'line': dict(color='red', width=2), 'visible': False}})
            
            # Pressure (p2)
            data.append({'x': mpc_time, 'y': MPC_data['p2'][idx], 'row': 2, 'col': 2, 'name': 'p<sub>2</sub>', 'kwargs': {'line': dict(color='blue', width=2), 'visible': False}})
            
            # Displacement (z)
            data.append({'x': mpc_time, 'y': MPC_data['z'][idx], 'row': 3, 'col': 1, 'name': 'z', 'kwargs': {'line': dict(color='chocolate', width=2), 'visible': False}})
            
            # Plot Command
            data.append({'x': mpc_time, 'y': MPC_data['u'][idx], 'row': 3, 'col': 2, 'name': 'u<sub>NN</sub>', 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh','visible': False}})
        
        # Slider information    
        slider_info = {'N_traj': N_TRAJ}

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Position [m]'    , 'row' : 1, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'     , 'row' : 1, 'col' : 2},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row' : 2, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row' : 2, 'col' : 2},
                {'x_label': 'Time [s]', 'y_label': 'Displacement [m]', 'row' : 3, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Command [ad]'    , 'row' : 3, 'col' : 2}]

        # Graphic
        Graphics.plot(data, axis, tab_title = "MPC + Slider", rows = 3, cols = 2, slider_info = slider_info, height = 1100, width = 2000,
                      subplot_titles = ["Deformation", "Deformation Speed", "Pressure (p1)", "Pressure (p2)", "Valve Displacement", "Command"])

    " ----------------------------------------------------------- "
    "                        NN SIMULATION                        "
    " ----------------------------------------------------------- "

    ############# Feasibility Recovery #############

    # Declare optimizer
    opti = Opti()

    # Declare optimization variables - u
    u = opti.variable()

    # Slack variable
    s = opti.variable(2)
    s_upper = s[0]
    s_lower = s[1]
    opti.subject_to([s_upper >= 0, s_lower >= 0])

    # Declare model state - x = [y, y_dot, p1, p2, z]
    x = opti.parameter(5)

    # Declare auxiliary parameters
    network_command = opti.parameter()  
    initial_state   = opti.parameter(5)

    # Scaling factors
    scaling_factors = {
        'u': 1/0.2,            
        'y': 1/0.02,
        'y_dot': 1/0.4,
        'p1': 1/32e6,
        'p2': 1/32e6,
        'z': 1/0.15
    }

    # Scaled variables
    u_scaled = scaling_factors['u'] * u
    x_scaled = vertcat(
        scaling_factors['y']     * x[0],
        scaling_factors['y_dot'] * x[1],
        scaling_factors['p1']    * x[2],
        scaling_factors['p2']    * x[3],
        scaling_factors['z']     * x[4]
    )

    # Cost function
    penalty_weight = 1e4
    opti.minimize(scaling_factors['u']*(network_command - u)**2 + penalty_weight *(s_upper**2 + s_lower**2))

    # Xdot expression (unscaled values)
    xdot = FeasibilityRecovery.forging_model(x, u)

    # Model Function
    f = Function('f', [x, u], [xdot])
    f = f.expand() # Transform MX into SX function

    # Build the integrator function (Runge Kutta method)
    F = FeasibilityRecovery.Ruge_Kuta(TS, f)

    # Call the integrator
    X = F(x0 = initial_state, u = u)
    X_NEW = F(x0 = X["xf"], u = u)

    # Constraint over pressure (p1) 
    opti.subject_to(opti.bounded(LB_P1*scaling_factors['p1'], X["xf"][2]*scaling_factors['p1'], UB_P1*scaling_factors['p1']))
    opti.subject_to(opti.bounded(LB_P1*scaling_factors['p1'], X_NEW["xf"][2]*scaling_factors['p1'], UB_P1*scaling_factors['p1']))

    # Constraint over pressure (p2) 
    opti.subject_to(opti.bounded(LB_P2*scaling_factors['p2'], X["xf"][3]*scaling_factors['p2'], UB_P2*scaling_factors['p2']))
    opti.subject_to(opti.bounded(LB_P2*scaling_factors['p2'], X_NEW["xf"][3]*scaling_factors['p2'], UB_P2*scaling_factors['p2']))

    # Solver
    opts = {'ipopt.warm_start_init_point': 'yes', 
            'ipopt.tol': 1e-5, 
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes', 
            'ipopt.linear_solver': 'ma27', 
            'ipopt.hsllib': '/home/martinxavier/coinhsl/.libs/libcoinhsl.so',
            }
    opti.solver('ipopt', opts)


    # Get Feasibility Recovery options
    if enable_feasibility:
        feasibility_variables = {'optimization': opti, 
                                 'opti_var': u,
                                 'slack_variables': s,
                                 'u_param': network_command, 
                                 'x_param': initial_state}
    else:
        feasibility_variables = None

    logger.info("\n-------- \nNN SIMULATION \n--------")

    # Get configured do-mpc modules
    sim_NN = template_simulator(model_MPC) 

    # Run the NN loop to generate the dataset
    sim_NN, data_NN, data_LSTM, timer_NN, feas_results = NeuralNetwork.loop(N_traj = N_TRAJ, T_traj = T_TRAJ, Ts = TS,
                                                                            controller     = controller_NN, 
                                                                            simulator      = sim_NN, 
                                                                            simulator_LSTM = simulator_LSTM,
                                                                            init_state     = init_state,
                                                                            scalers        = scalers,
                                                                            model_scalers  = model_scalers,
                                                                            bias_work      = 300, 
                                                                            bias_return    = 20**6,
                                                                            lookback       = lookback,
                                                                            feasibility    = feasibility_variables,
                                                                            bar_title      = 'NN',
                                                                            process_std    = process_std,
                                                                            meas_std       = meas_std)

    # Store results
    file       = f'forging_unsupervised_N_{N}' + add_feas_to_file + add_to_file
    
    if enable_feasibility:
        logger.info("\n-------- \nNN RESULTS + FEASIBILITY RECOVERY \n--------")
    else:
        logger.info("\n-------- \nNN RESULTS \n--------")

    do_mpc.data.save_results([sim_NN], file, overwrite = True)

    # ----------------------------------------------------------------
    # RESULTS
    # ----------------------------------------------------------------
    # Retrieve results
    sim_NN = sim_NN.data
    
    # Compute metrics
    NeuralNetwork.metrics(sim_NN['_tvp'][:,0], sim_NN['_y'][:,1], NN_results)
    NeuralNetwork.other_metrics(sim_NN['_u'][:,0], timer_NN.t_list, NN_results)

    # ----------------------------------------------------------------
    # PLOT - NN
    # ----------------------------------------------------------------
    sim_time = sim_NN['_time'][:,0]

    if show_plots:
        # Create empty list to store the data 
        data = []

        # Position (y)
        data.append({'x': sim_time, 'y': sim_NN['_y'][:,0], 'row': 1, 'col': 1, 'name': f'x', 'kwargs': {'line': dict(color='burlywood', width=2)}})

        # Speed reference (v_ref)
        data.append({'x': sim_time, 'y': sim_NN['_tvp'][:,0], 'row': 1, 'col': 2, 'name': f'v<sub>ref</sub>', 'kwargs': {'line': dict(color='green', width=2), 'line_shape': 'hvh'}})

        # Speed (v)
        data.append({'x': sim_time, 'y': sim_NN['_y'][:,1], 'row': 1, 'col': 2, 'name': f'v', 'kwargs': {'line': dict(color='orange', width=2)}})

        # Pressure (p1)
        data.append({'x': sim_time, 'y': sim_NN['_y'][:,2], 'row': 2, 'col': 1, 'name': f'p<sub>1</sub>', 'kwargs': {'line': dict(color='red', width=2)}})

        # Pressure (p2)
        data.append({'x': sim_time, 'y': sim_NN['_y'][:,3], 'row': 2, 'col': 2, 'name': f'p<sub>2</sub>', 'kwargs': {'line': dict(color='blue', width=2)}})

        # Valve displacement (z)
        data.append({'x': sim_time, 'y': sim_NN['_y'][:,4], 'row': 3, 'col': 1, 'name': f'z', 'kwargs': {'line': dict(color='chocolate', width=2)}})

        # Command (u)
        data.append({'x': sim_time, 'y': sim_NN['_u'][:,0], 'row': 3, 'col': 2, 'name': f'u', 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh'}})

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Position [m]'    , 'row': 1, 'col': 1},
                {'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'     , 'row': 1, 'col': 2},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row': 2, 'col': 1},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row': 2, 'col': 2},
                {'x_label': 'Time [s]', 'y_label': 'Displacement [m]', 'row': 3, 'col': 1},
                {'x_label': 'Time [s]', 'y_label': 'Command [ad]'    , 'row': 3, 'col': 2}]

        # Graphic
        Graphics.plot(data, axis, rows = 3, cols = 2, tab_title = 'NN', height = 1100,
                      subplot_titles = ["Deformation", "Deformation Speed", "Pressure (p1)", "Pressure (p2)", "Valve Displacement", "Command"])

    # ----------------------------------------------------------------
    # PLOT - NN + SLIDER
    # ----------------------------------------------------------------
    mpc_time = np.linspace(0, T_TRAJ + 1, T_TRAJ + 1, endpoint=False)*TS

    if show_plots:
        data = [] # Empty list

        # Loop over the references 
        for idx in range (N_TRAJ): 

            # Position
            data.append({'x': mpc_time, 'y': data_NN['y'][idx], 'row': 1, 'col': 1, 'name': 'x', 'kwargs': {'line': dict(color='burlywood', width=2), 'visible': False}})
            
            # Speed
            data.append({'x': mpc_time, 'y': data_NN['y_dot'][idx], 'row': 1, 'col': 2, 'name': 'v', 'kwargs': {'line': dict(color='orange', width=2), 'visible': False}})
            
            # Speed - NN
            data.append({'x': mpc_time, 'y': data_LSTM['y_dot'][idx], 'row': 1, 'col': 2, 'name': 'v<sub>NN</sub>', 'kwargs': {'line': dict(color='orangered', width=2), 'visible': False}})
            
            # Reference
            data.append({'x': mpc_time, 'y': data_NN['ref'][idx], 'row': 1, 'col': 2, 'name': 'v<sub>ref</sub>',  'kwargs': {'line': dict(color='green', width=2), 'line_shape': 'hvh', 'visible': False}})
            
            # Pressure (p1)
            data.append({'x': mpc_time, 'y': data_NN['p1'][idx], 'row': 2, 'col': 1, 'name': 'p<sub>1</sub>', 'kwargs': {'line': dict(color='red', width=2), 'visible': False}})
            
            # Pressure (p1) - NN
            data.append({'x': mpc_time, 'y': data_LSTM['p1'][idx], 'row': 2, 'col': 1, 'name': 'p<sub>1,NN</sub>', 'kwargs': {'line': dict(color='darksalmon', width=2), 'visible': False}})
            
            # Pressure (p2)
            data.append({'x': mpc_time, 'y': data_NN['p2'][idx], 'row': 2, 'col': 2, 'name': 'p<sub>2</sub>', 'kwargs': {'line': dict(color='blue', width=2), 'visible': False}})
            
            # Pressure (p2) - NN
            data.append({'x': mpc_time, 'y': data_LSTM['p2'][idx], 'row': 2, 'col': 2, 'name': 'p<sub>2,NN</sub>', 'kwargs': {'line': dict(color='skyblue', width=2), 'visible': False}})
            
            # Displacement (z)
            data.append({'x': mpc_time, 'y': data_NN['z'][idx], 'row': 3, 'col': 1, 'name': 'z', 'kwargs': {'line': dict(color='chocolate', width=2), 'visible': False}})
            
            # Displacement (z) - NN
            data.append({'x': mpc_time, 'y': data_LSTM['z'][idx], 'row': 3, 'col': 1, 'name': 'z<sub>NN</sub>', 'kwargs': {'line': dict(color='tan', width=2), 'visible': False}})
            
            # Plot Command
            data.append({'x': mpc_time, 'y': data_NN['u'][idx], 'row': 3, 'col': 2, 'name': 'u<sub>NN</sub>', 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh','visible': False}})

        # Slider information
        slider_info = {'N_traj': N_TRAJ}

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Position [m]'    , 'row' : 1, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'     , 'row' : 1, 'col' : 2},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row' : 2, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row' : 2, 'col' : 2},
                {'x_label': 'Time [s]', 'y_label': 'Displacement [m]', 'row' : 3, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Command [ad]'    , 'row' : 3, 'col' : 2}]

        # Graphic
        Graphics.plot(data, axis, rows = 3, cols = 2, tab_title = "NN + Slider", height = 1100, width = 2000, slider_info = slider_info,
                      subplot_titles = ["Deformation", "Deformation Speed", "Pressure (p1)", "Pressure (p2)", "Valve Displacement", "Command"])

    # ----------------------------------------------------------------
    # PLOT - FEASIBILITY RECOVERY STATISTICS
    # ----------------------------------------------------------------
    if show_plots and enable_feasibility:
        # Create empty list to store the data 
        data = []

        # Loop over the references 
        for idx in range (N_TRAJ): 

            # Iteration counter
            data.append({'x': mpc_time, 'y': feas_results['iter_count'][idx], 'row': 1, 'col': 1, 'name': 'iter_count', 'kwargs': {'line': dict(color='burlywood', width=2), 'visible': False}})
            
            # Dual step size
            data.append({'x': mpc_time, 'y': feas_results['alpha_du'][idx], 'row': 1, 'col': 2, 'name': 'alpha_du', 'kwargs': {'line': dict(color='orange', width=2), 'visible': False}})
            
            # Primal step size
            data.append({'x': mpc_time, 'y': feas_results['alpha_pr'][idx], 'row': 1, 'col': 2, 'name': 'alpha_pr', 'kwargs': {'line': dict(color='green', width=2), 'line_shape': 'hvh', 'visible': False}})
            
            # Norm of the primal search direction
            data.append({'x': mpc_time, 'y': feas_results['d_norm'][idx], 'row': 2, 'col': 1, 'name': 'd_norm', 'kwargs': {'line': dict(color='red', width=2), 'visible': False}})
            
            # Dual infeasibility
            data.append({'x': mpc_time, 'y': feas_results['inf_du'][idx], 'row': 2, 'col': 2, 'name': 'inf_du', 'kwargs': {'line': dict(color='blue', width=2), 'visible': False}})
            
            # Primal infeasibility
            data.append({'x': mpc_time, 'y': feas_results['inf_pr'][idx], 'row': 2, 'col': 2, 'name': 'inf_pr', 'kwargs': {'line': dict(color='chocolate', width=2), 'visible': False}})
            
            # Barrier parameter 
            data.append({'x': mpc_time, 'y': feas_results['mu'][idx], 'row': 3, 'col': 1, 'name': 'mu', 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh','visible': False}})
            
            # Objective Function
            data.append({'x': mpc_time, 'y': feas_results['obj'][idx], 'row': 3, 'col': 2, 'name': 'obj', 'kwargs': {'line': dict(color='pink', width=2), 'line_shape': 'hvh','visible': False}})

        # Slider information
        slider_info = {'N_traj': N_TRAJ}

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Iteration counter'    , 'row' : 1, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Step size'            , 'row' : 1, 'col' : 2},
                {'x_label': 'Time [s]', 'y_label': 'Search direction norm', 'row' : 2, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Infeasibility'        , 'row' : 2, 'col' : 2},
                {'x_label': 'Time [s]', 'y_label': 'Barrier parameter'    , 'row' : 3, 'col' : 1},
                {'x_label': 'Time [s]', 'y_label': 'Objective Function'   , 'row' : 3, 'col' : 2}]

        # Graphic
        Graphics.plot(data, axis, rows = 3, cols = 2, tab_title = "Feasibility", height = 1100, width = 2000, slider_info = slider_info,
                      subplot_titles = ["Iteration counter", "Step size", "Search direction norm", "Infeasibility", "Barrier parameter", "Objective Function"])


    # ----------------------------------------------------------------
    # PLOT - NN vs MPC
    # ----------------------------------------------------------------
    if show_comparison:
        # Empty list to store the data 
        data = []

        # MPC Speed
        if enable_mpc:
            data.append({'x': sim_time, 'y': sim_MPC['_y'][:,1], 'name': 'MPC', 'kwargs': {'line': dict(color='green', width=2)}})

        # NN Speed
        data.append({'x': sim_time, 'y': sim_NN['_y'][:,1], 'name': 'NN', 'kwargs': {'line': dict(color='blue', width=2)}})

        # Reference
        data.append({'x': sim_time, 'y': sim_NN['_tvp'][:,0], 'name': 'Reference', 'kwargs': {'line': dict(color='chocolate' , width=2, dash='dash'), 'line_shape': 'hvh'}})

        # Update axis
        axis = [{'x_label': 'Time [s]', 'y_label': 'Speed [m/]'}]

        # Graphic
        Graphics.plot(data, axis, tab_title = 'MPC vs NN', title = 'MPC vs NN', height = 1100)

    " ----------------------------------------------------------- "
    "              NOTIFICATION - END OF SIMULATION               "
    " ----------------------------------------------------------- "
        
    notification = Notify()
    notification.title   = f"Successful finished executing iteration !!! "
    notification.message = f"Iteration {sim} completed !!!"
    notification.audio   = "/home/martinxavier/Téléchargements/Notification Sound/sound2.wav"
    notification.send(block=False)

# Test Results
logger.info("\n-------- \nTEST STATS \n--------")
Data.show_tabulate(test_results, f'test_results_N_{N}' + add_to_file + '.csv', enable_feasibility, enable_training, enable_debug)

# NN Results - Closed Loop
logger.info("\n-------- \nNN STATS \n--------")
Data.show_tabulate(NN_results, f'NN_results_N_{N}' + add_to_file + '.csv', enable_feasibility, debug_flag = enable_debug)

# MPC Results - Closed Loop
logger.info("\n-------- \nMPC STATS \n--------")
Data.show_tabulate(MPC_results, f'MPC_results_N_{N}' + add_to_file + '.csv', enable_feasibility, enable_mpc, enable_debug)

# Creating a Dataframe
MPC_RESULTS = {
    'time' : sim_MPC['_time'][:,0],
    'ref'  : sim_MPC['_tvp'][:,0],
    'y'    : sim_MPC['_y'][:,0],
    'y_dot': sim_MPC['_y'][:,1],
    'p1'   : sim_MPC['_y'][:,2],
    'p2'   : sim_MPC['_y'][:,3],
    'z'    : sim_MPC['_y'][:,4],
    'u'    : sim_MPC['_u'][:,0],
}

MPC_dataframe = pd.DataFrame(MPC_RESULTS)
MPC_dataframe.to_csv('results/MPC_dataframe.txt', sep='\t', index=False, float_format="%.6f")

UNSUPERVISED_RESULTS = {
    'time' : sim_NN['_time'][:,0],
    'ref'  : sim_NN['_tvp'][:,0],
    'y'    : sim_NN['_y'][:,0],
    'y_dot': sim_NN['_y'][:,1],
    'p1'   : sim_NN['_y'][:,2],
    'p2'   : sim_NN['_y'][:,3],
    'z'    : sim_NN['_y'][:,4],
    'u'    : sim_NN['_u'][:,0],
}

Unsupervised_dataframe = pd.DataFrame(UNSUPERVISED_RESULTS)
Unsupervised_dataframe.to_csv('results/Unsupervised_dataframe.txt', sep='\t', index=False, float_format="%.6f")