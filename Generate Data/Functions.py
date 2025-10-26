# ---------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------

# Numpy
import numpy as np  

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Metrics
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Custom plotly identifier
from plotly.io._base_renderers import BrowserRenderer, open_html_in_browser
from plotly.io._renderers import renderers

# MPC tools
import do_mpc 

# Progress bar
from alive_progress import alive_bar

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
# GRAPHICS
# ---------------------------------------------------------------- 
class Graphics:
    """Class to deal with graphics and plotting."""

    def plot(data_series, axis_options, title='', tab_title = '', rows=1, cols=1, 
             subplot_titles=None, show = True, save_fig = False, slider_info = None, 
             add_hover = True, **kwargs):
        """
        Plots multiple graphs with different plot types on subplots using Plotly.

        Parameters:
            data_series (list of dicts): 
                List of dictionaries, where each dict represents a dataset with keys 'x', 'y', 'type',
                'name', 'row', 'col', and optionally 'kwargs'.             
            axis_options (list of dicts): 
                List of dictionaries, where each dict represents the axis options of a subplot with keys 
                'x_label', 'y_label', 'col', 'rol', and optionnaly 'x_kwargs' and 'y_kwargs'.
            title (str): 
                Title of the entire figure.
            tab_title (str): 
                Title of the HTML tab open in the navigator.
            rows (int): 
                Number of rows in the subplot grid.
            cols (int): 
                Number of columns in the subplot grid.
            subplot_titles (list): 
                List of titles for the individual subplots.
            show (bool): 
                Whether to show the plot immediately.
            save_fig (bool): 
                Whether to save the figure locally.
            **kwargs: 
                Additional layout parameters passed to Plotly.

        Returns:
            fig (Figure)
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
                fig.add_vline(x = x_data, line_width=2, line_dash="dash", line_color="black", row = row, col = col, **trace_kwargs)
            # H-line
            elif plot_type == 'h_line':
                fig.add_hline(y = y_data, line_width=2, line_dash="dash", line_color="black", row = row, col = col, **trace_kwargs)
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
# MODEL PREDICTIVE CONTROL
# ----------------------------------------------------------------   
class MPC:
    """Class to manage MPC functions."""

    @staticmethod
    def metrics(out: np.ndarray, ref:np.ndarray):
        """
        Compute performance metrics for the MPC controller.

        Parameters:
            out (np.ndarray): 
                MPC output values.
            ref (np.ndarray): 
                Desired output values (reference).

        Returns:
            results (dict)
                Dictionary containing the keys:
                    - *MAE*: Mean Average Error, 
                    - *RMSE*: Root Mean Squared Error,
                    - *R2 score*: R2 Coefficient.
        """
        results = {'mae': mean_absolute_error(out, ref),
                   'mse': root_mean_squared_error(out, ref),
                   'r2': r2_score(out, ref)
        }

        # Print results
        logger.info(f"- Mean Average Error (MAE) = {results['mae']:.4f}.")
        logger.info(f"- Root Mean Squared Error (RMSE) = {results['mse']:.4f}.")
        logger.info(f"- R2 Coefficient = {results['r2']:.4f}.")

        return results
    
    @staticmethod
    def loop(N_traj:int, T_traj:float, controller:do_mpc.controller._mpc.MPC, simulator:do_mpc.simulator.Simulator, init_state:dict, timer:do_mpc.tools._timer.Timer, bar_title:str, process_std: np.ndarray, meas_std: np.ndarray):
        """
        Function which implements the closed-loop simulation using a MPC controller.
        
        Parameters:
            N_traj (int): 
                Number of complete trajectories (working and return) in the simulation.
            T_traj (float): 
                Duration of the trajectory.
            controller (do_mpc.controller._mpc.MPC):
                MPC controller used in the closed-loop simulation.
            simulator (do_mpc.simulator.Simulator):
                Simulator of *do_mpc* acting as the physical system.
            init_state (dict):
                Dictionary with the initial states of the system.
            timer (do_mpc.tools._timer.Timer)
                Timer that will contain the computational cost of the controller.
            bar_title (str):
                Title displayed in the alive bar.
            process_std (np.ndarray):
                Standard deviation of the process noise for each state.
            meas_std (np.ndarray):
                Standard deviation of the measurement noise for each state.
        
        Returns: 
            controller (do_mpc.controller._mpc.MPC):
                MPC controller containing the controller's data after the simulation.
            simulator (do_mpc.simulator.Simulator)
                Simulator containing the system's data after the simulation.
            MPC_results (dict)
                Closed-loop trajectories from the *do_mpc* simulator.
            opt_results (dict)
                Optmization indicators from the IPOPT solver.
            timer (do_mpc.tools._timer.Timer)
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
        dataset_iter     = np.zeros((N_traj, T_traj))
        dataset_mu       = np.zeros((N_traj, T_traj))
        dataset_obj      = np.zeros((N_traj, T_traj))
        dataset_reg_size = np.zeros((N_traj, T_traj))
        dataset_d_norm   = np.zeros((N_traj, T_traj))
        dataset_inf_du   = np.zeros((N_traj, T_traj))
        dataset_inf_pr   = np.zeros((N_traj, T_traj))

        # Empty noise trajectory
        dataset_noise = np.zeros((N_traj, T_traj + 1, 5))

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
                dataset_noise[idx,0] = np.zeros((5))

                # Set the initial state
                controller.x0 = x0  # MPC
                simulator.x0  = x0  # Simulator 

                # Set initial guess
                controller.set_initial_guess()  

                # Main loop
                for t in range(T_traj):

                    # Compute the MPC command
                    timer.tic() # Set timer
                    u0 = controller.make_step(x0) 
                    timer.toc() # Stop timer

                    # Process noise
                    w0 = np.random.normal(loc=0.0, scale=process_std).reshape(-1, 1)
                    dataset_noise[idx,t+1] = w0.squeeze()

                    # Measurement noise
                    v0 = np.random.normal(loc=0.0, scale=meas_std).reshape(-1, 1)
                    # dataset_noise[idx,t+1] = v0.squeeze()

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
                
                    # Add commmands to trajectory
                    dataset_u[idx,t] = u0

                    # Add optmization variables to trajectory
                    dataset_iter[idx,t]     = controller.solver_stats['iter_count']
                    dataset_mu[idx,t]       = controller.solver_stats['iterations']['mu'][-1]
                    dataset_obj[idx,t]      = controller.solver_stats['iterations']['obj'][-1]
                    dataset_reg_size[idx,t] = controller.solver_stats['iterations']['regularization_size'][-1]
                    dataset_d_norm[idx,t]   = controller.solver_stats['iterations']['d_norm'][-1]
                    dataset_inf_du[idx,t]   = controller.solver_stats['iterations']['inf_du'][-1]
                    dataset_inf_pr[idx,t]   = controller.solver_stats['iterations']['inf_pr'][-1]

                # Set the initial command
                controller.u0 = 0.0  # MPC 
                simulator.u0  = 0.0  # Simulator 
                
                # Update progress bar
                bar()

        MPC_results = {'y' : dataset_y , 'y_dot': dataset_y_dot, 
                       'p1': dataset_p1, 'p2'   : dataset_p2   ,
                       'z' : dataset_z , 'ref'  : dataset_ref  , 
                       'u' : dataset_u , 'F_d'  : dataset_Fd   ,
                       'w' : dataset_noise}
        
        opt_results = {'iter'   : dataset_iter  , 'mu'      : dataset_mu, 
                       'obj'    : dataset_obj   , 'reg_size': dataset_reg_size,
                       'd_norm' : dataset_d_norm, 'inf_du'  : dataset_inf_du, 
                       'inf_pr' : dataset_inf_pr }
        

        return controller, simulator, MPC_results, opt_results, timer
    
# ----------------------------------------------------------------
# CUSTOM PLOTLY IDENTIFIER
# ---------------------------------------------------------------- 
class TitleBrowserRenderer(BrowserRenderer):
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