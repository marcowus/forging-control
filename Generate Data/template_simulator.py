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
    simulator = do_mpc.simulator.Simulator(model)  # Declaration of variable 'simulator' as a "do_mpc.simulator.Simulator" type

    params_simulator = {                           # Declaration of a 'params_simulator' array defining the simulator parameters 
        'integration_tool': 'cvodes',
        'abstol': 1e-5,
        'reltol': 1e-6,
        't_step': 0.001,
    }

    # simulator.settings.integration_opts = {'gather_stats':True, 'print_stats': False, 'verbose':False}

    simulator.set_param(**params_simulator)        # Setting the parameters of the simulation using the 'params_simulator' array

    # Realization of uncertain parameters
    p_num   = simulator.get_p_template()
    tvp_num = simulator.get_tvp_template()

    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)
    
    def tvp_fun(t_now):
        return tvp_num

    simulator.set_tvp_fun(tvp_fun)

    # Build simulation
    simulator.setup()

    return simulator