# Import Project Folder
from utils.Utils import FOLDER

import os, copy, time, json
import numpy as np
import gym

# Import QP Utilities
from scipy import sparse
import osqp

def AdamBA():

    pass

def AdamBA_SC():

    pass

def store_heatmap(env:gym.Env, cnt_store_heatmap_trigger, trigger_by_pre_execute, safe_index_now, threshold, n, k, sigma, pre_execute):

    # HeatMap Logger
    heat_map_logger_trigger = dict()
    heat_map_logger_trigger["distance_delta"]     = list()
    heat_map_logger_trigger["n_2_k_0.10_delta"]   = list()
    heat_map_logger_trigger["n_2_k_0.25_delta"]   = list()
    heat_map_logger_trigger["n_2_k_0.50_delta"]   = list()
    heat_map_logger_trigger["n_2_k_0.75_delta"]   = list()
    heat_map_logger_trigger["n_2_k_1.00_delta"]   = list()
    heat_map_logger_trigger["n_2_k_2.00_delta"]   = list()
    heat_map_logger_trigger["n_2_k_5.00_delta"]   = list()
    heat_map_logger_trigger["n_2_k_10.00_delta"]  = list()
    heat_map_logger_trigger["n_2_k_100.00_delta"] = list()

    # Initialize u(0)
    u_0 = 10.05

    for _ in range(200):

        # Initialize u(0), u(1)
        u_0, u_1 -= 0.1, -10.05

        for _ in range(200):

            # Initialize u(1)
            u_1 += 0.1

            # Copy Env State
            stored_state = copy.deepcopy(env.sim.get_state())

            # Get Safety Index
            safe_index_u_0_u_1_distance_before     = env.closest_distance_cost_n(n=1)
            safe_index_u_0_u_1_n_2_k_0_10_before   = env.adaptive_safety_index(k=0.10,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_0_25_before   = env.adaptive_safety_index(k=0.25,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_0_50_before   = env.adaptive_safety_index(k=0.50,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_0_75_before   = env.adaptive_safety_index(k=0.75,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_1_00_before   = env.adaptive_safety_index(k=1.00,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_2_00_before   = env.adaptive_safety_index(k=2.00,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_5_00_before   = env.adaptive_safety_index(k=5.00,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_10_00_before  = env.adaptive_safety_index(k=10.00,  sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_100_00_before = env.adaptive_safety_index(k=100.00, sigma=sigma, n=2)

            # Simulate the Action
            s_new = env.step(np.array([[u_0, u_1]]), simulate_in_adamba=True)

            # Get Closest Distance Cost
            safe_index_u_0_u_1_distance_future = env.closest_distance_cost_n(n=1)

            # Get Safety Index
            safe_index_u_0_u_1_n_2_k_0_10_future   = env.adaptive_safety_index(k=0.10,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_0_25_future   = env.adaptive_safety_index(k=0.25,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_0_50_future   = env.adaptive_safety_index(k=0.50,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_0_75_future   = env.adaptive_safety_index(k=0.75,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_1_00_future   = env.adaptive_safety_index(k=1.00,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_2_00_future   = env.adaptive_safety_index(k=2.00,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_5_00_future   = env.adaptive_safety_index(k=5.00,   sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_10_00_future  = env.adaptive_safety_index(k=10.00,  sigma=sigma, n=2)
            safe_index_u_0_u_1_n_2_k_100_00_future = env.adaptive_safety_index(k=100.00, sigma=sigma, n=2)

            # Compute Delta Safety Index
            safe_index_u_0_u_1_distance_delta = safe_index_u_0_u_1_distance_future - safe_index_u_0_u_1_distance_before

            if safe_index_u_0_u_1_distance_delta < 0:

                # Set Environment State
                env.sim.set_state(stored_state)

                # Environment Forward
                env.sim.forward()

                return env, cnt_store_heatmap_trigger

            # Compute Safe Index Delta
            safe_index_u_0_u_1_n_2_k_0_10_delta   = safe_index_u_0_u_1_n_2_k_0_10_future   - safe_index_u_0_u_1_n_2_k_0_10_before
            safe_index_u_0_u_1_n_2_k_0_25_delta   = safe_index_u_0_u_1_n_2_k_0_25_future   - safe_index_u_0_u_1_n_2_k_0_25_before
            safe_index_u_0_u_1_n_2_k_0_50_delta   = safe_index_u_0_u_1_n_2_k_0_50_future   - safe_index_u_0_u_1_n_2_k_0_50_before
            safe_index_u_0_u_1_n_2_k_0_75_delta   = safe_index_u_0_u_1_n_2_k_0_75_future   - safe_index_u_0_u_1_n_2_k_0_75_before
            safe_index_u_0_u_1_n_2_k_1_00_delta   = safe_index_u_0_u_1_n_2_k_1_00_future   - safe_index_u_0_u_1_n_2_k_1_00_before
            safe_index_u_0_u_1_n_2_k_2_00_delta   = safe_index_u_0_u_1_n_2_k_2_00_future   - safe_index_u_0_u_1_n_2_k_2_00_before
            safe_index_u_0_u_1_n_2_k_5_00_delta   = safe_index_u_0_u_1_n_2_k_5_00_future   - safe_index_u_0_u_1_n_2_k_5_00_before
            safe_index_u_0_u_1_n_2_k_10_00_delta  = safe_index_u_0_u_1_n_2_k_10_00_future  - safe_index_u_0_u_1_n_2_k_10_00_before
            safe_index_u_0_u_1_n_2_k_100_00_delta = safe_index_u_0_u_1_n_2_k_100_00_future - safe_index_u_0_u_1_n_2_k_100_00_before

            # Store Data in the HeatMap Logger
            heat_map_logger_trigger["distance_delta"].append(safe_index_u_0_u_1_distance_delta)
            heat_map_logger_trigger["n_2_k_0.10_delta"].append(safe_index_u_0_u_1_n_2_k_0_10_delta)
            heat_map_logger_trigger["n_2_k_0.25_delta"].append(safe_index_u_0_u_1_n_2_k_0_25_delta)
            heat_map_logger_trigger["n_2_k_0.50_delta"].append(safe_index_u_0_u_1_n_2_k_0_50_delta)
            heat_map_logger_trigger["n_2_k_0.75_delta"].append(safe_index_u_0_u_1_n_2_k_0_75_delta)
            heat_map_logger_trigger["n_2_k_1.00_delta"].append(safe_index_u_0_u_1_n_2_k_1_00_delta)
            heat_map_logger_trigger["n_2_k_2.00_delta"].append(safe_index_u_0_u_1_n_2_k_2_00_delta)
            heat_map_logger_trigger["n_2_k_5.00_delta"].append(safe_index_u_0_u_1_n_2_k_5_00_delta)
            heat_map_logger_trigger["n_2_k_10.00_delta"].append(safe_index_u_0_u_1_n_2_k_10_00_delta)
            heat_map_logger_trigger["n_2_k_100.00_delta"].append(safe_index_u_0_u_1_n_2_k_100_00_delta)

            # Reset Environment (Set q_pos and q_vel)
            env.sim.set_state(stored_state)

            # Environment Forward
            env.sim.forward()

            # AdamBA Status
            adamba_status = trigger_by_pre_execute or safe_index_now >= 0

    # Exit if Counter Exceed 100
    if cnt_store_heatmap_trigger >= 100: exit(0)

    else:

        import os, time, json

        # Increase Counter
        cnt_store_heatmap_trigger += 1

        # Debug Print
        print('Storing Data of This Trigger Moment for HeatMap')
        print('Storing %d/100 File' %(cnt_store_heatmap_trigger))

        # Create HeatMap
        json_data_heat_map = json.dumps(heat_map_logger_trigger, indent=1)

        # Write File
        time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

        # Check Directory Existence
        json_dir_path = os.path.join(FOLDER, 'data/heatmap/fixed_adaptive_hazard_%s_size_%s_threshold_%s_sigma_%s_n_%s_k_%s_pre_execute_%s_fixed_simulation/' % (str(env.hazards_num), str(env.hazards_size), str(threshold), str(sigma), str(n), str(k), str(pre_execute)))
        if not os.path.exists(json_dir_path): os.makedirs(json_dir_path)

        # Write JSON File
        json_file_path = json_dir_path + '%s+%s.json' % (str(cnt_store_heatmap_trigger), time_str)
        with open(json_file_path, 'w') as json_file: json_file.write(json_data_heat_map)

        print('Stored')

    return env, cnt_store_heatmap_trigger

def quadratic_programming(H, f, A=None, b=None, initvals=None, verbose=False):

    # QP Initialization
    qp_P = sparse.csc_matrix(H)
    qp_f = np.array(f)
    qp_l = -np.inf * np.ones(len(b))
    qp_A = sparse.csc_matrix(A)
    qp_u = np.array(b)

    # QP-Model Initialization
    model = osqp.OSQP()
    model.setup(P=qp_P, q=qp_f, A=qp_A, l=qp_l, u=qp_u, verbose=verbose)

    # Add Init Values
    if initvals is not None: model.warm_start(x=initvals)

    # Solve Model
    results = model.solve()

    # Return Results
    return results.x, results.info.status
