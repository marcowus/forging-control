import numpy as np              # To perform operations
from casadi import *            # To compute gradients, Jacobians and Hessians 
from casadi.tools import *
import do_mpc                   # To import the "do_mpc" package

# To create random numbers
from random import seed
from random import random

def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous'                            # Either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)  # Declaration of variable 'model' as a "do_mpc.model.Model" type

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
    KL_2 = 14*10**(-14) # Coefficient of external leakage flow in the return cylinders

    # Flow rate parameters
    CD  = 0.63           # Valve discharge coefficient 
    RHO = 858            # Density of the oil [kg/m^3]
    D   = 0.006          # Diameter of the control valve hole [m]

    ################# Equations #################
      
    # Pressure parameters of the hydraulic press
    PS = 32*10**6                                   # Supply pressure [Pa]
    PT = 101325                                     # Return pressure [Pa] (1 atm)

    # Geometric parameters
    MU = 0.3            # Coefficient of friction stress
    K  = 1.115          # Deformation strengthening indicator
    W0 = 0.2            # Original width [m]
    H0 = 0.5            # Original height [m]
    B0 = 0.1            # Original bite length [m]
    H1 = 0.35           # Deformed height [m]

    # Geometric equations
    A  = 0.14 + 0.36*(B0/W0) - 0.054*(B0/W0)**2     # Spreading coefficient (Tomlinson ans Stringer)

    # Characteristics of the part
    T = 900                                         # Deformation temperature [K]     

    # Servo valve parameters
    T1 = 0.005         # Time constant of the servo valve 

    ############ States struct (optimization variables) ############
    y     = model.set_variable('_x', 'y')           # Displacement of the upper die (deformation)
    y_dot = model.set_variable('_x', 'y_dot')       # Velocity of the upper die (deformation speed)
    p1    = model.set_variable('_x', 'p1')          # Pressure in the working cylinder 
    p2    = model.set_variable('_x', 'p2')          # Pressure in the return cylinder  
    z     = model.set_variable('_x', 'z')           # Displacement of the spool valve 

    # Input struct (optimization variables):
    u = model.set_variable('_u',  'u')              # Tension applied to the servo valve  

    h1 = H0 - y
    w1 = W0*(H0/h1)**A                              # Deformed width [m]
    b1 = B0*(1 + 0.67*(H0/h1*W0/w1 - 1))            # Deformed bite length [m]

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

    # Time vaying variable - Reference
    ref = model.set_variable(var_type='_tvp', var_name = 'ref')

    # Auxiliary variables
    model.set_expression(expr_name='Fd_article', expr = Fd_article)  # Fd Article
    
    ############ Differential equations ############
    Ft = if_else(fabs(y_dot) <= 0.5, FT*y_dot/0.5, FT)

    # Select the right expression according to return and working motions
    model.set_rhs('y'    , y_dot, process_noise=True)
    model.set_rhs('y_dot', (3*np.pi*D1**2*p1/4 - np.pi*D2**2*p2/2 - B*y_dot - Ft - Fd_article)/M + G, process_noise=True) # Global
    model.set_rhs('p1'   , (KB/V1*(qvPB/3 - A1*y_dot - KL_1*p1)), process_noise=True)
    model.set_rhs('p2'   , (KB/V2*(-qvAT/2 + A2*y_dot - KL_2*p2)), process_noise=True)
    model.set_rhs('z'    , (-z/T1 + u/T1), process_noise=True)

    # State Measurements
    model.set_meas('y_meas'    , y    , meas_noise=True)
    model.set_meas('y_dot_meas', y_dot, meas_noise=True)
    model.set_meas('p1_meas'   , p1   , meas_noise=True)
    model.set_meas('p2_meas'   , p2   , meas_noise=True)
    model.set_meas('z_meas'    , z    , meas_noise=True)

    # Build the model
    model.setup()

    return model
