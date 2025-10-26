# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------

# Standard Library Imports
import pickle   # Save data

# Third-Party Library Imports
import numpy  as np                                  # Arrays
import pandas as pd                                  # DataFrames
from sklearn.model_selection import train_test_split # Split Data                 

# PyTorch 
import torch                             # Tensors
import torch.nn as nn                    # Neural Network
from torch.utils.data import DataLoader  # DataLoaders

# do-mpc    
import do_mpc
from do_mpc.tools       import Timer           
from template_model     import template_model          
from template_mpc       import template_mpc       
from template_simulator import template_simulator  

# Local Imports
from Functions import (Data, 
                       MPC,
                       NeuralNetwork, 
                       LSTMModel,
                       Graphics,
                       Save_Network_Data)

# Notification
from notifypy import Notify

# Print information
import logging

# Get the device in which the nerwork is trained
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    print(f"Using device: {device}")
else:
    print(f"Using device: {torch.cuda.get_device_name(0)} ({device})")

# ----------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Configure logging messages
logging.basicConfig(level=logging.INFO, format='%(message)s', 
                    handlers=[logging.FileHandler(f"my_log.log", mode='w'), stream_handler])
logger = logging.getLogger()

# ----------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------
enable_mpc     = True   # MPC flag
silence_solver = True   # Silence IPOPT solver
enable_noise   = True   # Noise flag

# ---------------------------------------------------------------
# SIMULATION PARAMETERS
# ----------------------------------------------------------------

T_TRAJ = 300    # Trajectory period
N_TRAJ = 10     # Number of trajectories

# Hyperparameters 
BATCH_SIZE    = 256    # Batch size
LEARNING_RATE = 0.001  # Learning rate
N_EPOCHS      = 50     # Number of epochs

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

# No measurement noise
meas_std = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Make the simulation deterministic
# Data.random_simulation(seed = 600)

# ---------------------------------------------------------------
# DATA PREPARATION
# ----------------------------------------------------------------
data_files = ['../../Data/forging_mult_traj_process_noise_N_5.pkl',
              '../../Data/forging_mult_traj_process_noise_N_10.pkl',
              '../../Data/forging_mult_traj_process_noise_N_15.pkl',
              '../../Data/forging_mult_traj_process_noise_N_20.pkl',
              '../../Data/forging_mult_traj_process_noise_N_25.pkl']

# Create an Empty DataFrame object
df = pd.DataFrame()

for file in data_files:
    # Load MPC data from pickle file
    with open(file, 'rb') as f:
        results = pickle.load(f)

    # Extract States 
    y_dot = results['mpc']._x[:,1]   # Deformation rate [m/s]
    ref   = results['mpc']._tvp[:,0] # Reference [m/s]
    p1    = results['mpc']._x[:,2]   # Pressure working cylinder [Pa]
    p2    = results['mpc']._x[:,3]   # Pressure return cylinder [Pa]
    z     = results['mpc']._x[:,4]   # Opening of the servo-valve [m]
    u     = results['mpc']._u[:,0]   # Input action [ad]

    # Creating a Pandas Dataframe
    dataframe = pd.DataFrame({'y_dot': y_dot, 'p1': p1, 'p2': p2, 'z': z, 'ref': ref, 'u': u})
    df = pd.concat([df, dataframe])

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

# Inputs and outputs
features = ['y_dot', 'p1', 'p2', 'z', 'u']
target   = ['y_dot', 'p1', 'p2', 'z']

scalers = {
    'dataframe': Data.get_scaler('maxabs'),
    'input':     Data.get_scaler('maxabs'),
    'output':    Data.get_scaler('maxabs')
}

# Standardize Dataframe
scalers['dataframe'].fit(df_train)
scalers['dataframe'].set_output(transform="pandas")

# Standardize data to 0 mean and 1 std
df_train_scaled = scalers['dataframe'].transform(df_train)
df_val_scaled   = scalers['dataframe'].transform(df_val)
df_test_scaled  = scalers['dataframe'].transform(df_test)

# Only standardize features and targets
scalers['input'].fit(df_train[features].values)
scalers['output'].fit(df_train[target].values)

# Save the scaler
pickle.dump(scalers['input'] , open(f'results/scaler_model_input'  + add_to_file + '.pkl', 'wb'))
pickle.dump(scalers['output'], open(f'results/scaler_model_output' + add_to_file + '.pkl', 'wb'))

# Silence pandas warnings
pd.options.mode.chained_assignment = 'warn'  # default='warn'

# ----------------------------------------------------------------
# DATASETS
# ----------------------------------------------------------------
lookback = 10 # Lookback for reccurent networks

# Get list with individual datasets for reccurent networks
_, dataset_train_recurrent = Data.get_individual_dataset(df_train_scaled, target, features, T_TRAJ, lookback = lookback)
_, dataset_val_recurrent   = Data.get_individual_dataset(df_val_scaled  , target, features, T_TRAJ, lookback = lookback)
_, dataset_test_recurrent  = Data.get_individual_dataset(df_test_scaled , target, features, T_TRAJ, lookback = lookback)

# Concatenate the individual datasets - Recurrent
datasets_recurrent = {
    'train': torch.utils.data.ConcatDataset(dataset_train_recurrent),
    'val':   torch.utils.data.ConcatDataset(dataset_val_recurrent),
    'test':  torch.utils.data.ConcatDataset(dataset_test_recurrent)
}

# ----------------------------------------------------------------
# DATALOADERS
# ----------------------------------------------------------------
loaders_recurrent = {
    'train': DataLoader(datasets_recurrent['train'], batch_size = BATCH_SIZE, shuffle = True),
    'val':   DataLoader(datasets_recurrent['val']  , batch_size = BATCH_SIZE, shuffle = False),
    'test':  DataLoader(datasets_recurrent['test'] , batch_size = BATCH_SIZE, shuffle = False)
}

# Loader lenght
logger.info("\n-------- \nDATA \n--------")
logger.info(f"\nLenght of train set: {len(df_train)}")
logger.info(f"Lenght of validation set: {len(df_val)}")
logger.info(f"Lenght of test set: {len(df_test)}")

# Data shape - recurrent
X, y = next(iter(loaders_recurrent['train']))
logger.info(f"\nRecurrent features shape: {X.shape}")
logger.info(f"Recurrent target shape: {y.shape}")

" ----------------------------------------------------------- "
"                         MODEL - NN                          "
" ----------------------------------------------------------- "

# NN Parameters
INPUT_DIM  = len(features)    # Number of inputs 
WIDTH_DIM  = 3                # Number of hidden layers
HIDDEN_DIM = 50               # Number of neurons in the hidden layers
OUTPUT_DIM = len(target)      # Number of outputs

# Model Definition
model = LSTMModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, WIDTH_DIM)

# Loss Function
loss_function = nn.MSELoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay=0.0)

# ----------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------
# Send Model to GPU if available
model = model.to(device) 

# Train Loop
model, vec_t_loss, vec_v_loss, comp_time = NeuralNetwork.train_loop(model, loaders_recurrent['train'], loaders_recurrent['val'], loss_function, optimizer, N_EPOCHS, device)

# Save network weights
torch.save(model.state_dict(), 'results/model_NN' + add_to_file + '.pt')

# Send Model back to CPU
model = model.cpu()

# ----------------------------------------------------------------
# TESTING NETWORK
# ----------------------------------------------------------------
df_train, df_train_scaled = NeuralNetwork.test(loaders_recurrent['train'], df_train, df_train_scaled, model, target, scalers)
df_val  , df_val_scaled   = NeuralNetwork.test(loaders_recurrent['val']  , df_val  , df_val_scaled  , model, target, scalers)
df_test , df_test_scaled  = NeuralNetwork.test(loaders_recurrent['test'] , df_test , df_test_scaled , model, target, scalers)

# ----------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------
logger.info("\n-------- \nTEST RESULTS - MODEL \n--------")

# Loop to concatenate the scaled prediction to the DataFrame
prediction_col = []
for _, col in enumerate(target):
    prediction_col.append(f"NN({col})")

# Extract the reference and predictions from the DataFrame
y_true = df_test_scaled[target].to_numpy()
y_forecast = df_test_scaled[prediction_col].to_numpy()

# Test metrics
NeuralNetwork.metrics(y_true, y_forecast)

print("Speed")
NeuralNetwork.metrics(y_true[:,0], y_forecast[:,0])

print("Pressure 1")
NeuralNetwork.metrics(y_true[:,1], y_forecast[:,1])

print("Pressure 2")
NeuralNetwork.metrics(y_true[:,2], y_forecast[:,2])

print("Displacement Spool valve")
NeuralNetwork.metrics(y_true[:,3], y_forecast[:,3])

# ----------------------------------------------------------------
# PLOT - LOSS
# ----------------------------------------------------------------

# Time definition
train_time   = np.linspace(0, N_EPOCHS, N_EPOCHS)*LEARNING_RATE
min_loss_idx = np.argmin(vec_v_loss)*LEARNING_RATE

# List of data to plot
data = [
    {'x': train_time  , 'y': vec_t_loss, 'name': 'Training'      , 'row': 1, 'col': 1, 'kwargs': {'line': dict(color='green', width=2)}},
    {'x': train_time  , 'y': vec_v_loss, 'name': 'Validation'    , 'row': 1, 'col': 1, 'kwargs': {'line': dict(color='red', width=2)}},
    {'x': min_loss_idx, 'y': None      , 'type': 'v_line'        , 'row': 1, 'col': 1},
    {'x': train_time  , 'y': vec_t_loss, 'name': 'Log training'  , 'row': 2, 'col': 1, 'kwargs': {'line': dict(color='green', width=2), 'showlegend': False}},
    {'x': train_time  , 'y': vec_v_loss, 'name': 'Log validation', 'row': 2, 'col': 1, 'kwargs': {'line': dict(color='red', width=2)  , 'showlegend': False}},
    {'x': min_loss_idx, 'y': None      , 'type': 'v_line'        , 'row': 2, 'col': 1},
]

# Update axis
axis = [{'x_label': 'Time [s]', 'y_label': 'Loss'    , 'row': 1, 'col': 1},
        {'x_label': 'Time [s]', 'y_label': 'Log Loss', 'row': 2, 'col': 1, 'y_kwargs': {'type': 'log'}}
]

# Graphic
Graphics.plot(data, axis, rows = 2, cols = 1, tab_title = 'Loss', height = 1100,
              subplot_titles = ["Loss", "Logarithmic Loss"]
)

# ----------------------------------------------------------------
# PLOT - TEST
# ----------------------------------------------------------------

# List of the data to plot
mpc_time = np.linspace(0, len(df_test[target[0]]), len(df_test[target[0]]), endpoint=False)*0.001

data = [
    {'x': mpc_time, 'y': df_test[target[0]]        , 'name': 'MPC', 'row': 1, 'col': 1, 'kwargs': {'line': dict(color='blue', width=2)}},
    {'x': mpc_time, 'y': df_test[prediction_col[0]], 'name': 'NN' , 'row': 1, 'col': 1, 'kwargs': {'line': dict(color='red' , width=2)}},
    {'x': mpc_time, 'y': df_test[target[1]]        , 'name': 'MPC', 'row': 1, 'col': 2, 'kwargs': {'line': dict(color='blue', width=2), 'showlegend': False}},
    {'x': mpc_time, 'y': df_test[prediction_col[1]], 'name': 'NN' , 'row': 1, 'col': 2, 'kwargs': {'line': dict(color='red' , width=2), 'showlegend': False}},
    {'x': mpc_time, 'y': df_test[target[2]]        , 'name': 'MPC', 'row': 2, 'col': 1, 'kwargs': {'line': dict(color='blue', width=2), 'showlegend': False}},
    {'x': mpc_time, 'y': df_test[prediction_col[2]], 'name': 'NN' , 'row': 2, 'col': 1, 'kwargs': {'line': dict(color='red' , width=2), 'showlegend': False}},
    {'x': mpc_time, 'y': df_test[target[3]]        , 'name': 'MPC', 'row': 2, 'col': 2, 'kwargs': {'line': dict(color='blue', width=2), 'showlegend': False}},
    {'x': mpc_time, 'y': df_test[prediction_col[3]], 'name': 'NN' , 'row': 2, 'col': 2, 'kwargs': {'line': dict(color='red' , width=2), 'showlegend': False}},
]

# Update axis
axis = [{'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'       , 'row': 1, 'col': 1},
        {'x_label': 'Time [s]', 'y_label': 'Pressure - p1 [Pa]', 'row': 1, 'col': 2},
        {'x_label': 'Time [s]', 'y_label': 'Pressure - p2 [Pa]', 'row': 2, 'col': 1},
        {'x_label': 'Time [s]', 'y_label': 'Displacement [m]'  , 'row': 2, 'col': 1}
]

# Graphic
Graphics.plot(data, axis, rows = 2, cols = 2, tab_title = 'Test',  height = 1100,
              subplot_titles = ["Speed", "Pressure - p1", "Pressure - p2", "Spool Valve Stroke"],
   )

# ----------------------------------------------------------------
# SAVE DATA
# ----------------------------------------------------------------

network = Save_Network_Data(BATCH_SIZE, N_EPOCHS  , INPUT_DIM    , HIDDEN_DIM, 
                            WIDTH_DIM , OUTPUT_DIM, LEARNING_RATE, features, 
                            target    , optimizer)
    
with open("results/NN_model_data.bin", "wb") as f: # "wb" because we want to write in binary mode
    pickle.dump(network, f)

" ----------------------------------------------------------- "
"                       MPC SIMULATION                        "
" ----------------------------------------------------------- "

# Get configured do-mpc modules
model_MPC      = template_model()                         
controller_MPC = template_mpc(model_MPC, silence_solver)  
sim_MPC        = template_simulator(model_MPC)            

# Set a timer
timer = Timer()

# Set initial states
y_init     = 0.0
y_dot_init = 0.0      
p1_init    = 2156275.6006012624
p2_init    = 2961363.827545376
z_init     = 0.0 

init_state = {'y': y_init, 'y_dot': y_dot_init, 'p1': p1_init, 'p2': p2_init, 'z': z_init}

# MPC siulation
if enable_mpc:
    logger.info("\n-------- \nMPC SIMULATION \n--------")

    # Run the MPC loop to generate the dataset
    controller_MPC, sim_MPC, MPC_data, NN_data, opt, timer = MPC.loop(N_traj = N_TRAJ, T_traj = T_TRAJ, 
                                                                      controller  = controller_MPC, 
                                                                      simulator   = sim_MPC, 
                                                                      init_state  = init_state,
                                                                      timer       = timer,
                                                                      bar_title   = 'MPC',
                                                                      process_std = process_std,
                                                                      meas_std    = meas_std,
                                                                      model       = model,
                                                                      scalers     = scalers,
                                                                      lookback    = lookback)
    
    # Store results
    file = f'MPC_simulation' + add_to_file
    do_mpc.data.save_results([sim_MPC], file, overwrite = True)

    # MPC Info
    # timer.info()
    # timer.hist()

    # ----------------------------------------------------------------
    # RESULTS
    # ----------------------------------------------------------------
    # Retrieve results
    sim_MPC = sim_MPC.data
    
    # Compute metrics
    NeuralNetwork.metrics(sim_MPC['_tvp'][:,0], sim_MPC['_x'][:,1])
    NeuralNetwork.other_metrics(sim_MPC['_u'][:,0], timer.t_list)

    # print(NN_data['y_dot'])
    MPC_array = np.column_stack([v.flatten() for v in list(MPC_data.values())[1:5]])
    NN_array  = np.column_stack([v.flatten() for v in NN_data.values()])
    scaling   = np.max(NN_array, axis=0)

    logger.info("\n-------- \nNN - CLOSED LOOP \n--------")
    NeuralNetwork.metrics(MPC_array/scaling, NN_array/scaling)

# ----------------------------------------------------------------
# PLOT - MPC
# ----------------------------------------------------------------
sim_time = sim_MPC['_time'][:,0]

################ Figure 1 - MPC Simulation ################
if enable_mpc:
    # List of data to plot
    data = [
        {'x': sim_time, 'y': sim_MPC['_x'][:,0]  , 'row': 1, 'col': 1, 'name': 'x'               , 'kwargs': {'line': dict(color='burlywood', width=2)}},
        {'x': sim_time, 'y': sim_MPC['_tvp'][:,0], 'row': 1, 'col': 2, 'name': f'v<sub>ref</sub>', 'kwargs': {'line': dict(color='green', width=2), 'line_shape': 'hvh'}},
        {'x': sim_time, 'y': sim_MPC['_x'][:,1]  , 'row': 1, 'col': 2, 'name': 'v'               , 'kwargs': {'line': dict(color='orange', width=2)}},
        {'x': sim_time, 'y': sim_MPC['_x'][:,2]  , 'row': 2, 'col': 1, 'name': f'p<sub>1</sub>'  , 'kwargs': {'line': dict(color='red', width=2)}},
        {'x': sim_time, 'y': sim_MPC['_x'][:,3]  , 'row': 2, 'col': 2, 'name': f'p<sub>2</sub>'  , 'kwargs': {'line': dict(color='blue', width=2)}},
        {'x': sim_time, 'y': sim_MPC['_x'][:,4]  , 'row': 3, 'col': 1, 'name': 'z'               , 'kwargs': {'line': dict(color='chocolate', width=2)}},
        {'x': sim_time, 'y': sim_MPC['_u'][:,0]  , 'row': 3, 'col': 2, 'name': 'u'               , 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh'}},
    ]

    # Update axis
    axis = [{'x_label': 'Time [s]', 'y_label': 'Position [m]'    , 'row': 1, 'col': 1},
            {'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'     , 'row': 1, 'col': 2},
            {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row': 2, 'col': 1},
            {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row': 2, 'col': 2},
            {'x_label': 'Time [s]', 'y_label': 'Displacement [m]', 'row': 3, 'col': 1},
            {'x_label': 'Time [s]', 'y_label': 'Command [ad]'    , 'row': 3, 'col': 2}
    ]

    # Graphic
    Graphics.plot(data, axis, rows = 3, cols = 2, tab_title = 'MPC', height = 1100,
                  subplot_titles = ["Deformation"       , "Deformation Speed", 
                                    "Pressure (p1)"     , "Pressure (p2)"    , 
                                    "Valve Displacement", "Command"         ],
    )

# ----------------------------------------------------------------
# PLOT - MPC + SLIDER
# ----------------------------------------------------------------
mpc_time = controller_MPC.data['_time'][:,0]

if enable_mpc:
    # Create empty list to store the data 
    data = []

    # Loop over the references 
    for idx in range (N_TRAJ): 

        # Position
        data.append({'x': mpc_time, 'y': MPC_data['y'][idx]    , 'row': 1, 'col': 1, 'name': 'x', 
                     'kwargs': {'line': dict(color='chocolate', width=2), 'visible': False}})
        
        # Speed
        data.append({'x': mpc_time, 'y': MPC_data['y_dot'][idx], 'row': 1, 'col': 2, 'name': 'v', 
                     'kwargs': {'line': dict(color='blue', width=2), 'visible': False}})
        
        # Speed - NN
        data.append({'x': mpc_time, 'y': NN_data['y_dot'][idx] , 'row': 1, 'col': 2, 'name': 'v<sub>NN</sub>', 
                     'kwargs': {'line': dict(color='goldenrod', width=2), 'visible': False}})
        
        # Reference
        data.append({'x': mpc_time, 'y': MPC_data['ref'][idx]  , 'row': 1, 'col': 2, 'name': 'v<sub>ref</sub>', 
                     'kwargs': {'line': dict(color='red', width=2), 'line_shape': 'hvh', 'visible': False}})
        
        # Pressure (p1)
        data.append({'x': mpc_time, 'y': MPC_data['p1'][idx]   , 'row': 2, 'col': 1, 'name': 'p<sub>1</sub>', 
                     'kwargs': {'line': dict(color='green', width=2), 'visible': False}})
        
        # Pressure (p1) - NN
        data.append({'x': mpc_time, 'y': NN_data['p1'][idx]    , 'row': 2, 'col': 1, 'name': 'p<sub>1,NN</sub>', 
                     'kwargs': {'line': dict(color='lightsalmon', width=2), 'visible': False}})
        
        # Pressure (p2)
        data.append({'x': mpc_time, 'y': MPC_data['p2'][idx]   , 'row': 2, 'col': 2, 'name': 'p<sub>2</sub>', 
                     'kwargs': {'line': dict(color='orange', width=2), 'visible': False}}) 
        
        # Pressure (p2) - NN
        data.append({'x': mpc_time, 'y': NN_data['p2'][idx]    , 'row': 2, 'col': 2, 'name': 'p<sub>2,NN</sub>', 
                     'kwargs': {'line': dict(color='darkred', width=2), 'visible': False}})
        
        # Displacement (z)
        data.append({'x': mpc_time, 'y': MPC_data['z'][idx]    , 'row': 3, 'col': 1, 'name': 'z', 
                     'kwargs': {'line': dict(color='burlywood', width=2), 'visible': False}})
        
        # Displacement (z) - NN
        data.append({'x': mpc_time, 'y': NN_data['z'][idx]     , 'row': 3, 'col': 1, 'name': 'z<sub>NN</sub>', 
                     'kwargs': {'line': dict(color='coral', width=2), 'visible': False}})
        
        # Command
        data.append({'x': mpc_time, 'y': MPC_data['u'][idx]    , 'row': 3, 'col': 2, 'name': 'u<sub>MPC</sub>', 
                     'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh', 'visible': False}})
        
    # Slider information
    slider_info = {'N_traj': N_TRAJ}

    # Update axis
    axis = [{'x_label': 'Time [s]', 'y_label': 'Position [m]'       , 'row' : 1, 'col' : 1, 'y_kwargs': {'range': [-1.0, 1.0]}},
            {'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'        , 'row' : 1, 'col' : 2, 'y_kwargs': {'range': [-0.9, 0.9]}},
            {'x_label': 'Time [s]', 'y_label': 'Pressure - p1 [Pa]' , 'row' : 2, 'col' : 1, 'y_kwargs': {'range': [-0.5*10**7, 3.2*10**7]}},
            {'x_label': 'Time [s]', 'y_label': 'Pressure - p2'      , 'row' : 2, 'col' : 2, 'y_kwargs': {'range': [-0.5*10**7, 3.2*10**7]}},
            {'x_label': 'Time [s]', 'y_label': 'Displacement [m]'   , 'row' : 3, 'col' : 1, 'y_kwargs': {'range': [-0.5, 0.5]}},
            {'x_label': 'Time [s]', 'y_label': 'Command [ad]'       , 'row' : 3, 'col' : 2, 'y_kwargs': {'range': [-0.5, 0.5]}},
    ]

    # Graphic
    Graphics.plot(data, axis, tab_title = "MPC + Slider", rows = 3, cols = 2, slider_info = slider_info,
                  subplot_titles = ["Deformation"       , "Deformation Speed",
                                        "Pressure (p1)"     , "Pressure (p2)", 
                                        "Valve Displacement", "Command"],
                  height = 1100, width = 2000,
    )

# ----------------------------------------------------------------
# NOTIFICATION - END OF CODE
# ----------------------------------------------------------------
    
notification = Notify()
notification.title = "Script Execution Completed"
notification.message = "Successful finished executing !!!"
notification.send(block=False)