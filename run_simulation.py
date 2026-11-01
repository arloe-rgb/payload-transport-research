import numpy as np
import yaml

from models.plant_block import plant
from controllers.controller_block import PD_FF_Controller
#from controllers.control_allocation_clock import allocator
#from trajectories.reference_trajectory_block import ref_trajectory

def run_simulation(config):

    payload = plant(config)
    controller = PD_FF_Controller(config)
    #allocation = allocator(config)
    #x_ref = ref_trajectory(config)

    # SIM Setup
    sim = config['SIM']
    dt = sim["dt"]
    duration = sim["duration"]




def main():

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    sim_results = run_simulation(config)


if __name__ == "__main__":
    main()