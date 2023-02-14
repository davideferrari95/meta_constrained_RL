import numpy as np

class Odometry():

    ''' Odometry Computation Class '''
        
    def __init__(self):
        
        # Initialize Odometry
        self.x, self.y, self.θ = 0,0,0
        self.dx, self.dy, self.dθ = 0,0,0
        
        # BUG: Mujoco Step-Time = 0.002 ? -> x10 Factor Somewhere 
        self.dt = 0.02
    
    def update_odometry(self, accelerometer, velocimeter, gyroscope):
        
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

