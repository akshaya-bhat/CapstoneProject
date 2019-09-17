import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # Controller paramters are from "DBW Walkthrough"
        
        tau = 0.5
        ts = 0.02 
        self.current_velocity_lpf = LowPassFilter(tau, ts)
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        throttle_min = 0.0
        throttle_max = 0.4
        self.throttle_pid = PID(kp, ki, kd, throttle_min, throttle_max)
        
        self.steering_yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        self.vehicle_mass = vehicle_mass
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        # For PID
        self.last_time = rospy.get_time()

    def control(self, dbw_enabled, cmd_linear_velocity, cmd_angular_velocity, current_velocity):
        if not dbw_enabled:
            self.throttle_pid.reset()
            return 0.0, 0.0, 0.0
        
        current_velocity_filtered = self.current_velocity_lpf.filt(current_velocity)

        velocity_error = cmd_linear_velocity - current_velocity_filtered
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_pid.step(velocity_error, sample_time)
        brake = 0
        
        # Close enough to being stationary
        if cmd_linear_velocity == 0.0 and current_velocity_filtered < 0.1:
            throttle = 0
            brake = 700 # To prevent Carla from moving requires about 700 Nm of torque.
        
        # Want to slow down the car, but just letting off the throttle is not enough
        if velocity_error < 0 and throttle < 0.1:
            throttle = 0
            decel = abs(velocity_error / 1.0) # a = dv/dt is in m/s^2, so divide dv (m/s) by 1 second
            decel_limit = abs(self.decel_limit)
            # torque = radius * force = radius * mass * acceleration
            if decel < decel_limit:
              brake = self.wheel_radius * self.vehicle_mass * decel
            else:
              brake = self.wheel_radius * self.vehicle_mass * decel_limit  
        
        steering = self.steering_yaw_controller.get_steering(cmd_linear_velocity, cmd_angular_velocity, current_velocity)
        
        return throttle, brake, steering
        #return 1., 0., 0.
