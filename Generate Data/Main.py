# Numpy
import numpy as np 

# MPC 
import do_mpc                           
from do_mpc.tools       import Timer  
from template_model     import template_model          
from template_mpc       import template_mpc              
from template_simulator import template_simulator  

# Local Imports
from Functions import Graphics, MPC

# Notification
from notifypy import Notify

# Display information
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
# USER SETTINGS
# ----------------------------------------------------------------
store_results  = True    # Store results in memory
silence_solver = True    # Silence IPOPT solver

# ----------------------------------------------------------------
# INITIALIZATION
# ----------------------------------------------------------------
model_MPC      = template_model()                         
controller_MPC = template_mpc(model_MPC, silence_solver)
sim_MPC        = template_simulator(model_MPC)    

N_TRAJ = 80    # Number of trajectories
T_TRAJ = 300   # Trajectory period

# Process noise std per state
process_std = np.array([
    5e-1,     # Displacement (y)
    2e-0,     # Speed (y_dot) 
    5e7,      # Pressure (p1) 
    5e7,      # Pressure (p2) 
    2e-0      # Spool valve displacement (z) 
])

# No measurement noise
meas_std = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# ----------------------------------------------------------------
# MPC LOOP
# ----------------------------------------------------------------

# Set a timer
timer = Timer()

# Initial states
y_init     = 0.0
y_dot_init = 0.0      
p1_init    = 2156275.6006012624
p2_init    = 2961363.827545376
z_init     = 0.0 

init_state = {'y': y_init, 'y_dot': y_dot_init, 'p1': p1_init, 'p2': p2_init, 'z': z_init}

# Run the MPC loop to generate the dataset
controller_MPC, sim_MPC, results, opt, timer = MPC.loop(N_traj = N_TRAJ, 
                                                        T_traj = T_TRAJ, 
                                                        controller = controller_MPC, 
                                                        simulator = sim_MPC, 
                                                        init_state = init_state,
                                                        timer = timer,
                                                        bar_title = 'MPC',
                                                        process_std = process_std,
                                                        meas_std = meas_std)

# MPC Info
# timer.info()
# timer.hist()

# Store results:
file_name = f'forging_mult_traj_process_noise_N_{controller_MPC.settings.n_horizon}' 
if store_results:
    do_mpc.data.save_results([controller_MPC, sim_MPC], file_name, overwrite = True)

# Number of trajectories generated
logger.info(f'\nThis code generated {N_TRAJ} trajectories, which accounts {N_TRAJ*T_TRAJ} data points')

# ----------------------------------------------------------------
# POST TREATEMENT
# ----------------------------------------------------------------

# Compute MPC score and wall time
logger.info("\n-------- \nMPC RESULTS \n--------")
score = MPC.metrics(controller_MPC.data._tvp[:,0],controller_MPC.data._x[:,1])
logger.info(f"- Average Command = {np.mean(np.abs(sim_MPC.data['_u'][:,0])):.4f}.")

# Display MPC results
t_wall = controller_MPC.data.t_wall_total
logger.info(f"- Runtime = {np.mean(timer.t_list)*1000:.4f}+-{np.std(timer.t_list)*1000:.4f}ms. Fastest run {np.min(timer.t_list)*1000:.2f}ms, slowest run {np.max(timer.t_list)*1000:.2f}ms")

if not(silence_solver):
    logger.info(f"\nMean compilation time {1000*np.ndarray.mean(t_wall)} +- {1000*np.std(t_wall)} ms")
    logger.info(f"\nFastest run {1000*np.ndarray.min(t_wall)} ms, slowest run {1000*np.ndarray.max(t_wall)} ms")
    logger.info(f"\nMean run {1000*np.ndarray.mean(t_wall)} ms")

# ----------------------------------------------------------------
# GRAPHICS
# ----------------------------------------------------------------
time = controller_MPC.data._time.squeeze()

################ MPC Simulation ################

# List of data to plot
data = [
    {'x': time, 'y': np.ravel(results['y'])     , 'name': 'x'  , 'row': 1, 'col': 1, 'kwargs': {'line': dict(color='burlywood', width=2)}},
    {'x': time, 'y': sim_MPC.data._x[:,0]       , 'name': 'x'  , 'row': 1, 'col': 1, 'kwargs': {'line': dict(color='pink', width=2)}},
    {'x': time, 'y': controller_MPC.data._x[:,0], 'name': 'x'  , 'row': 1, 'col': 1, 'kwargs': {'line': dict(color='black', width=2)}},
    {'x': time, 'y': np.ravel(results['ref'])  , 'name': 'Ref', 'row': 1, 'col': 2, 'kwargs': {'line': dict(color='green', width=2), 'line_shape': 'hvh'}},
    {'x': time, 'y': np.ravel(results['y_dot']), 'name': 'v'  , 'row': 1, 'col': 2, 'kwargs': {'line': dict(color='orange', width=2)}},
    {'x': time, 'y': np.ravel(results['p1'])   , 'name': 'p1' , 'row': 2, 'col': 1, 'kwargs': {'line': dict(color='red', width=2)}},
    {'x': time, 'y': np.ravel(results['p2'])   , 'name': 'p2' , 'row': 2, 'col': 2, 'kwargs': {'line': dict(color='blue', width=2)}},
    {'x': time, 'y': np.ravel(results['z'])    , 'name': 'z'  , 'row': 3, 'col': 1, 'kwargs': {'line': dict(color='chocolate', width=2)}},
    {'x': time, 'y': np.ravel(results['u'])    , 'name': 'u'  , 'row': 3, 'col': 2, 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh'}},
]

# Update axis
axis = [{'x_label': 'Time [s]', 'y_label': 'Position [m]'    , 'row': 1, 'col': 1},
        {'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'     , 'row': 1, 'col': 2},
        {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'  , 'row': 2, 'col': 1},
        {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'  , 'row': 2, 'col': 2},
        {'x_label': 'Time [s]', 'y_label': 'Displacement [m]', 'row': 3, 'col': 1},
        {'x_label': 'Time [s]', 'y_label': 'Input [ad]'    , 'row': 3, 'col': 2}
]

# Graphic
Graphics.plot(data, axis, rows = 3, cols = 2, tab_title = 'MPC', height = 1000,
              subplot_titles = ["Deformation"       , "Deformation Speed", 
                                "Pressure (p1)"     , "Pressure (p2)"    , 
                                "Valve Displacement","Command"          ],  
)

################ Reference vs Speed ################

# Create empty list to store the data 
data = []

# Loop over the references 
for idx in range (N_TRAJ): 
    
    # Speed
    data.append({'x': time, 'y': results['y_dot'][idx], 'row': 1, 'col': 1, 'name': 'Speed', 
                 'kwargs': {'line': dict(color='blue', width=2), 'showlegend': False, 'visible': False}})
    
    # Reference
    data.append({'x': time, 'y': results['ref'][idx]  , 'row': 1, 'col': 1, 'name': 'Reference', 
                 'kwargs': {'line': dict(color='red', width=2), 'line_shape': 'hvh', 'showlegend': False, 'visible': False}})
    
    # Position
    data.append({'x': time, 'y': results['y'][idx]    , 'row': 1, 'col': 2, 'name': 'Position',
                 'kwargs': {'line': dict(color='chocolate', width=2), 'showlegend': False, 'visible': False}})
    
    # Pressure (p1)
    data.append({'x': time, 'y': results['p1'][idx]   , 'row': 2, 'col': 1, 'name': 'Pressure - p1',
                 'kwargs': {'line': dict(color='green', width=2), 'showlegend': False, 'visible': False}})
    
    # Pressure (p2)
    data.append({'x': time, 'y': results['p2'][idx]   , 'row': 2, 'col': 1, 'name': 'Pressure - p2', 
                 'kwargs': {'line': dict(color='orange', width=2), 'showlegend': False, 'visible': False}})
    
    # Displacement (z)
    data.append({'x': time, 'y': results['z'][idx]    , 'row': 2, 'col': 2, 'name': 'Displacement (z)',
                 'kwargs': {'line': dict(color='burlywood', width=2), 'showlegend': False, 'visible': False}})
    
    # Command
    data.append({'x': time, 'y': results['u'][idx]    , 'row': 3, 'col': 1, 'name': 'Command', 
                 'kwargs': {'line': dict(color='darkcyan', width=2), 'line_shape': 'hvh', 'showlegend': False, 'visible': False}})
    
    # Deformation force
    data.append({'x': time, 'y': results['F_d'][idx]  , 'row': 3, 'col': 2, 'name': 'Fd', 
                 'kwargs': {'line': dict(color='darkolivegreen', width=2), 'showlegend': False, 'visible': False}})
 
# Slider information
slider_info = {'N_traj': N_TRAJ}

# Update axis
axis = [{'x_label': 'Time [s]', 'y_label': 'Speed [m/s]'     , 'row' : 1, 'col' : 1, 'y_kwargs': {'range': [-1.0, 1.0]}},
        {'x_label': 'Time [s]', 'y_label': 'Pressure [Pa]'   , 'row' : 2, 'col' : 1, 'y_kwargs': {'range': [-0.5*10**7, 3.2*10**7]}},
        {'x_label': 'Time [s]', 'y_label': 'Position [m]'    , 'row' : 1, 'col' : 2, 'y_kwargs': {'range': [-0.5, 0.5]}},
        {'x_label': 'Time [s]', 'y_label': 'Displacement [m]', 'row' : 2, 'col' : 2, 'y_kwargs': {'range': [-0.5, 0.5]}},
        {'x_label': 'Time [s]', 'y_label': 'Command [ad]'    , 'row' : 3, 'col' : 1, 'y_kwargs': {'range': [-0.5, 0.5]}},
        {'x_label': 'Time [s]', 'y_label': 'Deformation Force [N] Count', 'row' : 3, 'col' : 2}
]

# Graphic
Graphics.plot(data, axis, tab_title = "State Space", rows = 3, cols = 2, slider_info = slider_info,
    subplot_titles = ["Speed", "Position", "Pressure", "Valve Displacement", "Command", "Deformation Force"],
    height = 1000, width = 1700
)

################ Optimization ################
# Create empty list to store the data 
data = []

# Loop over the references 
for idx in range (N_TRAJ): 
    # Mu
    data.append({'x': time, 'y': opt['mu'][idx]      , 'row': 1, 'col': 1, 'name': 'Mu', 
                 'kwargs': {'line': dict(color='blue', width=2), 'showlegend': False, 'visible': False}})
    
    # Plot obj
    data.append({'x': time, 'y': opt['obj'][idx]     , 'row': 1, 'col': 2, 'name': 'Obj',
                 'kwargs': {'line': dict(color='red', width=2), 'showlegend': False, 'visible': False}})
    
    # reg_size
    data.append({'x': time, 'y': opt['reg_size'][idx], 'row': 2, 'col': 1, 'name': 'reg_size',
                 'kwargs': {'line': dict(color='chocolate', width=2), 'showlegend': False, 'visible': False}})
    
    # iter_count
    data.append({'x': time, 'y': opt['iter'][idx]    , 'row': 2, 'col': 2, 'name': 'iter_count',
                 'kwargs': {'line': dict(color='green', width=2), 'showlegend': False, 'visible': False}})
    
    # d_norm
    data.append({'x': time, 'y': opt['d_norm'][idx]  , 'row': 3, 'col': 1, 'name': 'd_norm',
                 'kwargs': {'line': dict(color='orange', width=2), 'showlegend': False, 'visible': False}})
    
    # inf_du
    data.append({'x': time, 'y': opt['inf_du'][idx]  , 'row': 3, 'col': 2, 'name': 'inf_du',
                 'kwargs': {'line': dict(color='burlywood', width=2), 'showlegend': False, 'visible': False}})
    
    # inf_pr
    data.append({'x': time, 'y': opt['inf_pr'][idx]  , 'row': 3, 'col': 1, 'name': 'inf_pr', 
                 'kwargs': {'line': dict(color='darkcyan', width=2), 'showlegend': False, 'visible': False}})
    
# Slider information
slider_info = {'N_traj': N_TRAJ}

# Update axis
axis = [{'x_label': 'Time [s]', 'y_label': 'Barrier parameter', 'row': 1, 'col': 1},
        {'x_label': 'Time [s]', 'y_label': 'Objective'        , 'row' : 2, 'col' : 1},
        {'x_label': 'Time [s]', 'y_label': 'Regularization'   , 'row' : 1, 'col' : 2},
        {'x_label': 'Time [s]', 'y_label': 'Iteration Count'  , 'row' : 2, 'col' : 2},
        {'x_label': 'Time [s]', 'y_label': 'Norm'             , 'row' : 3, 'col' : 1},
        {'x_label': 'Time [s]', 'y_label': 'Infeasibility'    , 'row' : 3, 'col' : 2}
]

# Graphic
Graphics.plot(data, axis, tab_title = "Optimization", rows = 3, cols = 2, slider_info = slider_info,
    subplot_titles = ["Barrier parameter", "Objective", "Regularization",  
                      "Iteration Count"  , "Norm"     , "Infeasibility"],
    height = 1000, width = 1700
)

# ----------------------------------------------------------------
# NOTIFICATION - END OF CODE
# ----------------------------------------------------------------
notification = Notify()
notification.title = "Script Execution Completed!!!"
notification.icon = "/home/martinxavier/Téléchargements/pythoned.png"
notification.send(block=False)