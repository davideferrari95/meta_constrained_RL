# Import Project Folder
from utils.Utils import FOLDER

import os, copy, time, json, gym
import numpy as np
from typing import Optional

# Import QP Utilities
from scipy import sparse
import osqp

def AdamBA(obs:np.ndarray, act:np.ndarray, env:gym.Env, threshold:float, dt_ratio:float=1.0, ctrlrange:float=10.0):

    # Reshape Action
    act, action_space_num = np.clip(act, -ctrlrange, ctrlrange), 2
    action = np.array(act).reshape(-1, action_space_num)
    limits = [[-ctrlrange, ctrlrange], [-ctrlrange, ctrlrange]]

    # AdamBA Parameters
    dt_adamba = 0.002 * env.frameskip_binom_n * dt_ratio
    bound = 0.0001

    # Generate Direction
    NP_vec_dir, NP_vec = [], []
    sigma_vec = [[1, 0], [0, 1]]
    vec_num = 100

    # Num of Actions Input, Default = 1
    for t in range(0, action.shape[0]):

        # Initialize Arrays
        vec_set, vec_dir_set = [], []

        for _ in range(0, vec_num):

            # Compute Action Vectors
            vec_dir = np.random.multivariate_normal(mean=[0, 0], cov=sigma_vec)
            vec_dir_set.append(vec_dir / np.linalg.norm(vec_dir))
            vec_set.append(action[t])

        # Append Action Vectors
        NP_vec_dir.append(vec_dir_set)
        NP_vec.append(vec_set)

    # Record how many Boundary Points Found
    collected_num, valid, cnt, out, yes = 0, 0, 0, 0, 0

    for n in range(0, action.shape[0]):

        # Extract Temp Vectors
        NP_vec_tmp, NP_vec_dir_tmp = NP_vec[n], NP_vec_dir[n]

        for v in range(0, vec_num):

            # Break After 2 Collections
            if collected_num >= 2: break

            # Increase Collected Num
            collected_num = collected_num + 1

            # Update NP_vec
            NP_vec_tmp_i, NP_vec_dir_tmp_i = NP_vec_tmp[v], NP_vec_dir_tmp[v]
            eta, decrease_flag = bound, False

            while eta >= bound:

                # Check Environment Safety
                flag, env = check_unsafe(obs, NP_vec_tmp_i, dt_ratio=dt_ratio, dt_adamba=dt_adamba, env=env, threshold=threshold)

                # Check Safety-Gym Env out-of-bound
                if out_of_bound(limits, NP_vec_tmp_i):

                    # Not Found -> Discard the Recorded Number
                    collected_num = collected_num - 1
                    break

                # AdamBA Procedure

                # Outreach
                if flag == True and decrease_flag == False:

                    NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                    eta = eta * 2
                    continue

                # Monitor for 1st Reaching Out Boundary
                if flag == False and decrease_flag == False:

                    decrease_flag = True
                    eta = eta * 0.25
                    continue

                # Decrease Eta
                if flag == True and decrease_flag == True:

                    NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                    eta = eta * 0.5
                    continue

                if flag == False and decrease_flag == True:

                    NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                    eta = eta * 0.5
                    continue

            # Update `NP_vec`
            NP_vec_tmp[v] = NP_vec_tmp_i

        NP_vec_tmp_new = []

        for v_num in range(0, len(NP_vec_tmp)):

            # Increase Counter
            cnt += 1

            # Check Out-Of-Bound
            if out_of_bound(limits, NP_vec_tmp[v_num]):

                out += 1
                continue

            if NP_vec_tmp[v_num][0] == act[0] and NP_vec_tmp[v_num][1] == act[1]:

                yes += 1
                continue

            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[v_num])

        NP_vec[n] = NP_vec_tmp_new

    # Start to Get the A and B for the Plane
    NP_vec_tmp = NP_vec[0]

    # AdamBA Status
    if valid == 2: valid_adamba = "adamba success"
    elif valid == 0 and yes == 100: valid_adamba = "itself satisfy"
    elif valid == 0 and out == 100: valid_adamba = "all out"
    elif valid == 1: valid_adamba = "one valid"
    else: valid_adamba = "exception"

    # Only Need 2 Points
    if len(NP_vec_tmp) == 2:

        # Compute the AdamBA Actions
        x1, y1 = NP_vec_tmp[0][0], NP_vec_tmp[0][1]
        x2, y2 = NP_vec_tmp[1][0], NP_vec_tmp[1][1]
        a = threshold * (y1 - y2) / (x2 * y1 - x1 * y2)
        b = threshold * (x1 - x2) / (y2 * x1 - y1 * x2)
        A = [a, b]

        # Return the Safe Action
        return [A, threshold], valid_adamba

    # Return None
    else: return [None, None], valid_adamba

def AdamBA_SC(obs:np.ndarray, act:np.ndarray, env:gym.Env, threshold:float=0, dt_ratio:float=1.0, ctrlrange:float=10.0,
              margin:float=0.4, adaptive_k:float=3, adaptive_n:float=1, adaptive_sigma:float=0.04, trigger_by_pre_execute:bool=False,
              pre_execute_coef:float=0.0, vec_num:Optional[int]=None, max_trial_num:int=1):

    # Reshape Action
    act, action_space_num = np.clip(act, -ctrlrange, ctrlrange), env.action_space.shape[0]
    action = np.array(act).reshape(-1, action_space_num)
    limits= [[-ctrlrange, ctrlrange]] * action_space_num

    # AdamBA Parameters
    dt_adamba = env.model.opt.timestep * env.frameskip_binom_n * dt_ratio
    loc, scale, bound = 0, 0.1, 0.0001

    # Check `dt_ratio`
    assert dt_ratio == 1

    # Generate Direction
    NP_vec_dir, NP_vec = [], []

    # Generate `vec_num`
    if   action_space_num ==  2: vec_num = 10 if vec_num == None else vec_num
    elif action_space_num == 12: vec_num = 20 if vec_num == None else vec_num
    else: raise NotImplementedError

    # Num of Actions Input, Default = 1
    for t in range(0, action.shape[0]):

        if action_space_num == 2:

            # Initialize Arrays
            vec_set, vec_dir_set = [], []

            for m in range(0, vec_num):

                # Compute Action Vectors
                theta_m = m * (2 * np.pi / vec_num)
                vec_dir = np.array([np.sin(theta_m), np.cos(theta_m)]) / 2
                vec_dir_set.append(vec_dir)
                vec_set.append(action[t])

            # Append Action Vectors
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)

        else:

            # Compute Action Vectors
            vec_dir_set = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
            vec_set = [action[t]] * vec_num

            # Append Action Vectors
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)

    # Record how many Boundary Points Found
    valid, cnt, out, yes = 0, 0, 0, 0

    for n in range(0, action.shape[0]):

        # Init Variables
        trial_num, at_least_1 = 0, False

        while trial_num < max_trial_num and not at_least_1:

            # Increase Counter
            trial_num, at_least_1 = trial_num + 1, False
            NP_vec_tmp = copy.deepcopy(NP_vec[n])

            # Compute NP_vec_dir
            if trial_num == 1: NP_vec_dir_tmp = NP_vec_dir[n]
            else: NP_vec_dir_tmp = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])

            for v in range(0, vec_num):

                # Update NP_vec
                NP_vec_tmp_i, NP_vec_dir_tmp_i = NP_vec_tmp[v], NP_vec_dir_tmp[v]
                eta, decrease_flag = bound, False

                while True:

                    # Check Environment Safety
                    flag, env = check_unsafe_sc(obs, NP_vec_tmp_i, dt_ratio, dt_adamba, env, threshold, margin,
                                                adaptive_k, adaptive_n, adaptive_sigma, trigger_by_pre_execute, pre_execute_coef)

                    # Check Safety-Gym Env out-of-bound
                    if out_of_bound(limits, NP_vec_tmp_i, sc=True): break

                    # Check Boundaries
                    if eta <= bound and flag == False:

                        at_least_1 = True
                        break

                    # AdamBA SC Procedure

                    # Outreach
                    if flag == True and decrease_flag == False:

                        NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                        eta = eta * 2
                        continue

                    # Monitor for 1st Reaching Out Boundary
                    if flag == False and decrease_flag == False:

                        decrease_flag = True
                        eta = eta * 0.25
                        continue

                    # Decrease Eta
                    if flag == True and decrease_flag == True:

                        NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                        eta = eta * 0.5
                        continue

                    if flag == False and decrease_flag == True:

                        NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                        eta = eta * 0.5
                        continue

                # Update `NP_vec`
                NP_vec_tmp[v] = NP_vec_tmp_i

        NP_vec_tmp_new = []

        for v_num in range(0, len(NP_vec_tmp)):

            # Increase Counter
            cnt += 1

            if out_of_bound(limits, NP_vec_tmp[v_num], sc=True):

                out += 1
                continue

            if NP_vec_tmp[v_num][0] == act[0] and NP_vec_tmp[v_num][1] == act[1]:

                yes += 1
                continue

            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[v_num])

        NP_vec[n] = NP_vec_tmp_new

    NP_vec_tmp = NP_vec[0]

    # AdamBA SC Status
    if valid > 0: valid_adamba_sc = "adamba_sc success"
    elif valid == 0 and yes == vec_num: valid_adamba_sc = "itself satisfy"
    elif valid == 0 and out==vec_num: valid_adamba_sc = "all out"
    else: valid_adamba_sc = "exception"

    # At Least 1 Sampled Action Satisfying the Safety Index
    if len(NP_vec_tmp) > 0:

        # Compute the AdamBA SC Actions
        norm_list = np.linalg.norm(NP_vec_tmp, axis=1)
        optimal_action_index = np.where(norm_list == np.amin(norm_list))[0][0]

        # Return the Safe Action
        return NP_vec_tmp[optimal_action_index], valid_adamba_sc, env, NP_vec_tmp

    # Return None
    else: return None, valid_adamba_sc, env, None

def store_heatmap(env:gym.Env, cnt_store_heatmap_trigger:int, trigger_by_pre_execute:bool, safe_index_now:float,
                  threshold:float, n:float, k:float, sigma:float, pre_execute:bool):

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
        u_0, u_1 = u_0 - 0.1, -10.05

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

def check_unsafe(s, point, dt_ratio:float, dt_adamba:float, env:gym.Env, threshold:float):

    # Create Action
    action = [point[0], point[1]]

    # Save State of Environment
    stored_state = copy.deepcopy(env.sim.get_state())

    # Get Robot Body Jacobian
    stored_robot_position = env.robot_pos
    mujoco_id = env.sim.model.body_name2id('robot')
    stored_robot_body_jacp = copy.deepcopy(env.sim.data.body_jacp[mujoco_id])

    # Compute Projection Cost
    cost_now = env.cost()['cost']
    projection_cost_now = env.projection_cost()

    # Simulate the Action
    s_new = env.step(action, dt_ratio, simulate_in_adamba=True)
    vel_after_tmp_action = env.sim.data.get_body_xvelp('robot')

    # Compute Future Projection Cost
    cost_future = env.cost()['cost']
    projection_cost_future = env.projection_cost()

    # Projection d_phi = cost_future - cost_now
    d_phi = projection_cost_future - projection_cost_now

    # Safe = False | Unsafe = True
    if d_phi <= threshold * dt_adamba: flag = False
    else: flag = True

    # Reset Env (Set q_pos and q_vel)
    env.sim.set_state(stored_state)
    env.sim.forward()

    return flag, env

def check_unsafe_sc(s, point, dt_ratio:float, dt_adamba:float, env:gym.Env, threshold:float,
                    margin:float, adaptive_k:float, adaptive_n:float, adaptive_sigma:float,
                    trigger_by_pre_execute:bool, pre_execute_coef:float):

    # Create Action
    action = point.tolist()

    # Save State of Environment
    stored_state = copy.deepcopy(env.sim.get_state())

    # Compute Safety Index
    safe_index_now = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

    # Simulate the Action
    s_new = env.step(action, simulate_in_adamba=True)

    # Compute Future Safety Index
    safe_index_future = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

    # Safety Index d_phi = safe_index_future - safe_index_now
    d_phi = safe_index_future - safe_index_now

    # Pre-Execute Trigger
    if trigger_by_pre_execute:

        # Safe = False | Unsafe = True
        if safe_index_future < pre_execute_coef: flag = False
        else: flag = True

    # Here dt_adamba = dt_env
    else:

        # Safe = False | Unsafe = True
        if d_phi <= threshold * dt_adamba: flag = False
        else: flag = True

    # Reset Env (Set q_pos and q_vel)
    env.sim.set_state(stored_state)
    env.sim.forward()

    return flag, env

def out_of_bound(limit, action, sc=False):

    # Assert Limit Length
    if not sc: assert len(limit[0]) == 2, 'WARNING: Length of Limit != 2'

    for i in range(len(limit)):

        # Check Limit Make Sense
        assert limit[i][1] > limit[i][0]

        # Limit Violation -> Out-Of-Bound -> Return True
        if action[i] < limit[i][0] or action[i] > limit[i][1]: return True

    # Return False if No-Limit-Violation
    return False
