# ----------------------------------------------------------------
#   Example: Forging Problem (MPC controller)
# ----------------------------------------------------------------

import numpy as np              # To perform operations
from casadi import *            # To compute gradients, Jacobians and Hessians 
from casadi.tools import *
import do_mpc                   # To import the "do_mpc" package

# To create random numbers
from random import seed
from random import random

# MPC
def template_mpc(model, silence_solver = False, do_feasibility = False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    # Declaration of MPC parameters
    mpc.settings.n_horizon = 10
    mpc.settings.n_robust  = 0
    mpc.settings.open_loop = 0
    mpc.settings.t_step    = 0.001
    mpc.settings.store_full_solution =  True
    mpc.settings.nlpsol_opts = {
        'ipopt.linear_solver': 'MA27', 
        'ipopt.hsllib': '/home/martinxavier/coinhsl/.libs/libcoinhsl.so'
    }

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # Definition of the scaling factors
    mpc.scaling['_x', 'y']     = 0.1
    mpc.scaling['_x', 'y_dot'] = 0.1
    mpc.scaling['_x', 'p1']    = 10**7
    mpc.scaling['_x', 'p2']    = 10**7
    mpc.scaling['_x', 'z']     = 0.01
    mpc.scaling['_u', 'u']     = 0.01

    # Import state variables from model
    _x = model.x
    _tvp = model.tvp

    # Objective function
    ref = _tvp['ref']               # Desired speed
    mterm = (_x['y_dot'] - ref)**2  # Terminal cost 
    lterm = (_x['y_dot'] - ref)**2  # Stage cots 

    # Set objective function
    mpc.set_objective(mterm = mterm, lterm = lterm) 
    mpc.set_rterm(u = 0.02) # The penalization is in its quadratic form 

    if do_feasibility:
        # Constraint over pressure (p1) 
        mpc.bounds['lower', '_x', 'p1'] = 0.0
        mpc.bounds['upper', '_x', 'p1'] = 32*10**6

        # Constraint over pressure (p2) 
        mpc.bounds['lower', '_x', 'p2'] = 0.0 
        mpc.bounds['upper', '_x', 'p2'] = 32*10**6

    # Get template for time-varying variables 
    tvp_template = mpc.get_tvp_template()

    # Trajectory period
    T_TRAJ = 300

    # Duration of the reference:
    T_REF = mpc.settings.t_step*T_TRAJ

    # To avoid precision errors
    epsilon = 10**(-7)
    
    def tvp_fun(t_now):
        # Reference
        if ((np.round(np.squeeze(t_now),3)+epsilon)%T_REF) < T_REF/2: 
            # Select seed
            seed((np.round(np.squeeze(t_now),3)+epsilon)//T_REF + 300) 

            # Get the reference
            tvp_template['_tvp'] = 0.8*random() + 0.1
        else: 
            # Select seed
            seed((np.round(np.squeeze(t_now),3)+epsilon)//T_REF + 20**6) 

            # Get the reference
            tvp_template['_tvp'] = -0.8*random() - 0.1
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    # Build mpc
    mpc.setup()
    
    return mpc
