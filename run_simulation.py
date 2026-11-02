import numpy as np
import yaml

from models.plant_block import plant
from controllers.controller_block import PD_FF_Controller
#from controllers.control_allocation_clock import allocator
#from trajectories.reference_trajectory_block import ref_trajectory

def run_simulation(config):

    # Object Initialization
    payload = plant(config)
    controller = PD_FF_Controller(config)
    #allocation = allocator(config)
    #trajectory = reference_trajectory(config)

    # SIM Parameter Setup
    sim = config['SIM']
    dt = sim["dt"]
    duration = sim["duration"]
    num_steps = int(round(duration/dt))
    t_vec = np.linspace(0, duration, num_steps + 1)

    # DATA logging setup
    logged_state = np.zeros((num_steps + 1, payload.state.size))
    logged_state[0] = payload.state.copy()

    # Simulation Loop
    for i in range(num_steps):

        # get time
        t = t_vec[i]

        # get reference trajectory and reference accel at time t
        #x_r, xddot_r = trajectory.get_trajectory(t)

        # calculate control input
        tau_d = controller.control_cmd(payload.state, x_r, xddot_r)

        # control allocation
        # u_optimal = allocation.allocate() -- will fill in later

        # integrate plant dynamics
        q_next = payload.integrate(u_optimal, dt)

        # store states for plotting and analysis
        logged_state[i + 1] = q_next



def main():

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    sim_results = run_simulation(config)


if __name__ == "__main__":
    main()