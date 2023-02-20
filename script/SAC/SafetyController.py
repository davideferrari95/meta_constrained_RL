from SAC.Utils import print_float_array, is_between_180, get_index
import numpy as np

class Odometry():

    ''' Odometry Computation Class '''
        
    def __init__(self):
        
        # Initialize Odometry
        self.x, self.y, self.θ = 0,0,0
        self.dx, self.dy, self.dθ = 0,0,0
        
        # BUG: Mujoco Step-Time = 0.002 ? -> x10 Factor Somewhere 
        self.dt = 0.02
    
    def update_odometry(self, new_observation):
        
        # Get Sensors
        accelerometer = new_observation['sorted_obs']['accelerometer']
        velocimeter   = new_observation['sorted_obs']['velocimeter']
        gyroscope     = new_observation['sorted_obs']['gyro']
        
        # Robot Sensors
        acc, vel, gyro = accelerometer, velocimeter, gyroscope
        
        # Compute Acceleration
        # self.ddx = accel[0] * np.cos(self.θ) - accel[1] * np.sin(self.θ)
        # self.ddy = accel[0] * np.sin(self.θ) + accel[1] * np.cos(self.θ)
        
        # Compute Velocities
        # self.dx = self.dx + self.ddx * self.dt
        # self.dy = self.dy + self.ddy * self.dt
        self.dx = vel[0] * np.cos(self.θ) - vel[1] * np.sin(self.θ)
        self.dy = vel[0] * np.sin(self.θ) + vel[1] * np.cos(self.θ)
        self.dθ = gyro[2]
        
        # Compute Pose
        self.x = self.x + self.dx * self.dt
        self.y = self.y + self.dy * self.dt
        self.θ = self.θ + self.dθ * self.dt

        return self.x, self.y, self.θ % np.radians(360)
    
    def stay_in_position(self, x, y, θ): 
        
        # TODO:

        return False

    def odometry_print(self):
    
        print(f'X: {self.x} | Y: {self.y} | θ: {np.degrees(self.θ)}\n')

class SafetyController():
    
    ''' Safety Controller Class to Avoid Constraint Collision '''    

    def __init__(self, lidar_num_bins:int=16, lidar_max_dist:float=None, 
                 lidar_exp_gain:float=1.0, lidar_type:str='pseudo', 
                 debug_print=False):
        
        # Environment Parameters
        self.lidar_num_bins   = lidar_num_bins    # Number of Lidar Dots
        self.lidar_max_dist   = lidar_max_dist    # Maximum distance for lidar sensitivity (if None, exponential distance)
        self.lidar_exp_gain   = lidar_exp_gain    # Scaling factor for distance in exponential distance lidar
        self.lidar_type       = lidar_type        # 'pseudo', 'natural', see self.obs_lidar()
        self.debug_print      = debug_print       # Debug Terminal Print
        
        # Move Compute Lidar Bin Position in Env Initialization
        self.lidar_bin_position = self.compute_lidar_bin_position()
    
    def compute_lidar_bin_position(self):
        
        ''' Compute Lidar Bin Angles-Indexes Disposition '''
            
        lidar_bins = []
        
        # Compute Lidar Bin Position
        for idx in range(self.lidar_num_bins):
            
            # Offset to Center of Bin
            i = idx + 0.5 if self.lidar_type == 'pseudo' else idx 
            
            # Compute Bin Angle
            theta = 2 * np.pi * i / self.lidar_num_bins
            theta_prev = 2 * np.pi * (i-1) / self.lidar_num_bins
            theta_subs = 2 * np.pi * (i+1) / self.lidar_num_bins
            
            # Convert (-θ) and (360 + θ) to θ
            if theta_prev < 0: theta_prev += np.radians(360)
            if theta_subs > np.radians(360): theta_subs -= np.radians(360)
            
            # Append to Bin List
            lidar_bins.append({'Index': idx, 'Angle': np.degrees(theta), 
                            'Area': [np.degrees(theta_prev), np.degrees(theta_subs)]})
            
        # print('\n\nLidar Bins Position:\n')
        # for bin in lidar_bins: print(f'\tIndex: {bin["Index"]} \t|\t Angle: {bin["Angle"]:.2f} \t|\t Area: [{bin["Area"][0]:.2f}, {bin["Area"][1]:.2f}]')
        
        return lidar_bins

    def get_unsafe_lidar_bins(self, obs, threshold=0.5):
        
        ''' Get Unsafe Lidar Bins '''
    
        # Get Hazard, Vases... Lidar
        hazard_lidar   = obs['hazards_lidar']  if 'hazards_lidar'  in obs.keys() else np.zeros(self.lidar_num_bins)
        vases_lidar    = obs['vases_lidar']    if 'vases_lidar'    in obs.keys() else np.zeros(self.lidar_num_bins)
        pillars_lidar  = obs['pillars_lidar']  if 'pillars_lidar'  in obs.keys() else np.zeros(self.lidar_num_bins)
        gremlins_lidar = obs['gremlins_lidar'] if 'gremlins_lidar' in obs.keys() else np.zeros(self.lidar_num_bins)
        
        # Sum all the Lidar
        lidar_max = np.maximum(np.maximum(hazard_lidar, vases_lidar), np.maximum(pillars_lidar, gremlins_lidar))
        
        # Obstacle Real Distance || hazard_lidar = np.exp(-self.lidar_exp_gain * dist)
        # real_dist = [(- np.log(dist) if dist > 0.0001 else -1) for dist in hazard_lidar]
        
        # Get Unsafe Lidar (Value-Index) Tuple
        unsafe_lidar = [{'Value':lidar, 'Index':idx[0]} for idx, lidar 
                        in np.ndenumerate(lidar_max) if lidar > threshold]
                
        for bin in unsafe_lidar:
            
            # Get Unsafe Lidar Index
            idx = bin['Index']
        
            # Get Bin Angle and Areas
            bin['Angle'] = self.lidar_bin_position[idx]['Angle']
            bin['Area']  = self.lidar_bin_position[idx]['Area']
        
        # Print Unsafe Lidar Bins
        if self.debug_print and unsafe_lidar != []:
            print('\n\nUnsafe Lidar:\n')
            for bin in unsafe_lidar: print(f'\tIndex: {bin["Index"]} \t|\t Value: {bin["Value"]:.8f} \t|\t Angle: {bin["Angle"]:.2f} \t|\t Area: [{bin["Area"][0]:.2f}, {bin["Area"][1]:.2f}]')
            # print_float_array('\nHazard Lidar Real:', real_dist, decimal_number=3)
        
        return unsafe_lidar

    def check_safe_action(self, action, obs, threshold=0.5):
    
        ''' Check if Action is Safe and Eventually Return a Safe One Instead '''
        
        # Get Unsafe Lidar Bins Vector
        unsafe_lidar = self.get_unsafe_lidar_bins(obs, threshold)
        
        # All Lidar are Safe -> All Actions are Safe -> Return Input Action
        if unsafe_lidar == []: return unsafe_lidar, action
        
        # Get Vector of Unsafe Angles
        obstacle_angles = np.array([(element['Angle'] if element['Angle'] <= 180 else element['Angle'] - 360) for element in unsafe_lidar])
        if self.debug_print: print (f'\nObstacle Angles: {obstacle_angles}')
        
        # Check Rotation Input
        rot = self.check_rotation_action(action, linear_vel = obs['velocimeter'][0], 
                                         obstacle_angles = obstacle_angles)
        
        # Check Linear Input
        lin = self.check_linear_action(action, linear_vel = obs['velocimeter'][0],
                                       distance = np.max([value['Value'] for value in unsafe_lidar]),
                                       obstacle_angles = obstacle_angles)
            
        return unsafe_lidar, np.array([lin, rot], dtype=np.float32)
    
    def check_rotation_action(self, action, linear_vel, obstacle_angles):
        
        ''' Check Rotation Component of the Action '''
        
        θ_action = action[1]
        min_angle_sx, max_angle_dx = 90, -90
        
        # Negative Velocity -> Invert Angles
        if linear_vel < 0: obstacle_angles = - obstacle_angles

        for angle in obstacle_angles:
            
            # Get Min Positive Angle (SX) and Max Negative Angle (DX)
            min_angle_sx = angle if angle >= 0 and angle < min_angle_sx else min_angle_sx
            max_angle_dx = angle if angle < 0  and angle > max_angle_dx else max_angle_dx
        
        # TODO: Add an α (10°) Threshold to Safe Area
        # min_angle_sx, max_angle_dx = (min_angle_sx - 10), (max_angle_dx + 10)
        
        # Safe Area | Normalized Between +-1
        safe_area = [min_angle_sx / 90, max_angle_dx / 90]
        
        # If Action not in Safe Area -> Sample New Action
        if θ_action > safe_area[0] or θ_action < safe_area[1]:
            
            if self.debug_print: print(f'\nAction (x,{θ_action}): Outside Safe Area ({min_angle_sx},{max_angle_dx}) -> ({safe_area[0]},{safe_area[1]})')
            
            # Clip Action in Safe Area
            θ_action = np.clip(θ_action, safe_area[1], safe_area[0])
            
            # OR: Sample a New Action from a Gaussian Function of Safe Area
            # sample = np.random.normal(loc=(safe_area[0] + safe_area[1]) / 2, scale=1/6)
            # θ_action = np.clip(sample, safe_area[1], safe_area[0])
            
            if self.debug_print: print(f'New Safe Rotational Action: (x,{θ_action})')
            
        return θ_action
    
    def check_linear_action(self, action, linear_vel, distance, obstacle_angles):
        
        ''' Check Linear Component of the Action '''
        
        # Negative Velocity -> Change Angles ([145,-145] -> [35,-35])
        if linear_vel < 0: obstacle_angles = np.sign(obstacle_angles) * (180 - np.abs(obstacle_angles))
        
        # 60 Degree Front / Back Area
        self.front_area = [35, -35]
        
        # Check if an Obstacle is in Front -> if Obstacle Angle is in Front Area
        obstacle = np.array([is_between_180(angle, self.front_area[1], self.front_area[0])
                            for angle in obstacle_angles]).any()

        # No Obstacle In Front Area -> Return The Input Action
        if not obstacle: return action[0]
        
        # Else -> Obstacle Found in Front Area        
        if self.debug_print: print (f'\nObstacle in Front / Back Area')
        
        # Normalized Exponential Distance from Lidar (1 = In Collision)
        safe_distance = distance # + 0.2
        
        # Normalized Velocity
        vel_limit = 1.5
        vel_norm = np.interp(linear_vel, (-vel_limit, vel_limit), (-1,1))
        
        # If V-Max and D-Max -> F = - 1/2 * (1 + 1) = -1
        Kp = Kv = 0.5
        
        # F = - sign(v) * (Kv * |v_norm| + Kp * |safe_dist_norm|)
        F_action = - np.sign(linear_vel) * (Kv * np.abs(linear_vel) + Kp * np.abs(safe_distance))
        
        if self.debug_print: print(f'Safe Distance = {safe_distance} | Normalized Velocity = {vel_norm}')
        if self.debug_print: print(f'New Action: ({F_action}, {action[1]})')
                
        return F_action
    
    def simulate_unsafe_action(self, obs, sorted_obs, unsafe_lidar, unsafe_action):
        
        # Hard-Coded Reward and Cost
        unsafe_reward, unsafe_cost, done = -0.1, 0.1, False
        
        # Initialize Observations
        lidar = np.copy(sorted_obs['hazards_lidar']).astype(np.float32)
        unsafe_obs, new_lidar = np.copy(obs), np.copy(lidar)

        for bin in unsafe_lidar:
            
            # Convert [0;360] in [-180;+180]
            bin['Angle'] = bin['Angle'] if bin['Angle'] <= 180 else bin['Angle'] - 360 
            
            # If Front Action and 
            if (unsafe_action[0] >= 0 and 
                    
                    # Front Obstacle or Turn-Left Obstacle or Turn-Right Obstacle
                    (  is_between_180(bin['Angle'], self.front_area[1], self.front_area[0])
                    or (unsafe_action[1] > 0 and is_between_180(bin['Angle'], 0, 90))
                    or (unsafe_action[1] < 0 and is_between_180(bin['Angle'], -90, 0)))
            
            # If Back Action and 
            ) or (unsafe_action[0] < 0  and 
                  
                    # Back Obstacle or Back-Turn-Left Obstacle or Back-Turn-Right Obstacle
                    ( is_between_180(bin['Angle'], - 180 - self.front_area[1], 180 - self.front_area[0], extern_angle=True)
                    or (unsafe_action[1] < 0 and is_between_180(bin['Angle'], 0, 90))
                    or (unsafe_action[1] > 0 and is_between_180(bin['Angle'], -90, 0)))
            ):
                
                # Increase Value of Lidar Bin in Unsafe Action Direction
                new_lidar[bin['Index']] = np.clip(new_lidar[bin['Index']] + 0.05, 0, 1)

        # Print Lidar Changes
        # if self.debug_print: print(f'\nLidar: {lidar}')
        # if self.debug_print: print(f'\nNew Lidar: {new_lidar}')
        if self.debug_print: print_float_array(f'\nLidar Changes: ', new_lidar - lidar)
        
        # Map Sorted Lidar with Obs Vector
        lidar_index = get_index(np.round(lidar, 4), np.round(obs, 4))
        assert lidar_index is not None, f'Lidar Index is None'
        unsafe_obs[lidar_index:lidar_index+len(new_lidar)] = new_lidar
        
        # Print Mapping
        # if self.debug_print: print_float_array(f'\nSorted_obs: ', lidar, decimal_number=4)
        # if self.debug_print: print_float_array(f'Obs: ', obs, decimal_number=4)
        # if self.debug_print: print(f'Lidar Index: {lidar_index}')
        
        # Print Old and New Observation
        # if self.debug_print: print_float_array(f'\nObservation: ', obs)
        # if self.debug_print: print_float_array(f'New Observation: ', unsafe_obs)
        
        return (obs, unsafe_action, unsafe_reward, unsafe_cost, float(done), unsafe_obs)
    
    def observation_print(self, action, reward, done, truncated, info):
        
        # Print Observation
        print(f'\n\nObservation Space: \n')
        
        # Print Sensors Vector
        for element in info['sorted_obs'].keys():
            print_float_array(f'{element.capitalize()}:', info['sorted_obs'][element])
            
        # Print Action
        print(f'\n\nAction: ({action[0]}, {action[1]})')
        
        # Print Info
        cost = info.get('cost', 0)
        print(f'Reward: {reward}')
        print(f'Done, Trunc: {done}, {truncated}')
        print(f'Cost: {cost}\n')
        print('-'*100)
