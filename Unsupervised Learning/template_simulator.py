import numpy as np              # To perform operations
from casadi import *            # To compute gradients, Jacobians and Hessians 
from casadi.tools import *
import do_mpc                   # To import the "do_mpc" package

# To create random numbers
from random import seed
from random import random

# Simulator
def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)  

    params_simulator = {                          
        'integration_tool': 'cvodes',
        'abstol': 1e-5,
        'reltol': 1e-6,
        't_step': 0.001,
    }


    simulator.set_param(**params_simulator)     

    # Realization of uncertain parameters
    p_num   = simulator.get_p_template()
    tvp_num = simulator.get_tvp_template()

    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)

    # Trajectory period
    T_TRAJ = 300

    # MPC Time step
    TS = 0.001

    # Duration of the reference:
    T_REF = TS*T_TRAJ

    # To avoid precision errors
    epsilon = 10**(-7)
    
    def tvp_fun(t_now):
        # Reference
        if ((np.round(np.squeeze(t_now),3)+epsilon)%T_REF) < T_REF/2: 
            # Select seed
            seed((np.round(np.squeeze(t_now),3)+epsilon)//T_REF + 300) 

            # Get the reference
            tvp_num ['ref'] = 0.8*random() + 0.1
        else: 
            # Select seed
            seed((np.round(np.squeeze(t_now),3)+epsilon)//T_REF + 20**6) 

            # Get the reference
            tvp_num ['ref'] = -0.8*random() - 0.1
        return tvp_num

    simulator.set_tvp_fun(tvp_fun)

    # Build simulation
    simulator.setup()

    return simulator