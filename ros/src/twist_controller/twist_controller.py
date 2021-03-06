import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, 
                 vehicle_mass, 
                 fuel_capacity, 
                 brake_deadband, 
                 decel_limit,               
                 accel_limit, 
                 wheel_radius, 
                 wheel_base, 
                 steer_ratio,                                                                                         
                 max_lat_accel,
                 max_steer_angle):
        self.yaw = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.fp = open("/home/student/Desktop/autonomous-driving-system/ros/twist_log.txt", "w")
        kp = 0.15
        ki = 0.0003
        kd = 3
        min_throttle = 0.0
        max_throttle = 0.5
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        
        # cut-off frequency for LPF
        tau_v = 0.1
        tau_err = 0.3
        # time frame
        ts = 0.02

        self.vel_lpf = LowPassFilter(tau_v,ts)
        self.err_lpf = LowPassFilter(tau_err,ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.last_vel = 0

    def control(self, current_vel, angular_vel, linear_vel, dbw_enabled):
        #self.fp.write("en:{}\n".format(dbw_enabled))
        if not dbw_enabled:
            #self.fp.write("not enabled!!\n")
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        current_vel = self.vel_lpf.filt(current_vel)
        
        steering = self.yaw.get_steering(linear_vel, angular_vel, current_vel)
        current_time = rospy.get_time()
        time_interval = current_time - self.last_time
        self.last_time = current_time
        
        vel_error = linear_vel - current_vel
        vel_error = self.err_lpf.filt(vel_error)
        self.last_vel = current_vel
        throttle = self.throttle_controller.step(vel_error, time_interval)
        brake = 0
        
        #stop the car
        if linear_vel == 0.0 and vel_error < 0.1:
            throttle = 0
            brake = 700
        elif vel_error < -0.08:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            #Torque - Neuton * meter
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        elif throttle < 0.05:
            throttle = 0.0
            brake = 0.0
        # rospy.logwarn("T:{}, B:{}, S:{}".format(throttle,brake,steering))
        return throttle, brake, steering