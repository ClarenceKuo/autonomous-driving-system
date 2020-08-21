import rospy
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
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        min_throttle = 0.0
        max_throttle = 0.2
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        
        # cut-off frequency for LPF
        tau = 0.5
        # time frame
        ts = 0.02

        self.vel_lpf = LowPassFilter(tau,ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.throttle = 0
        self.brake = 0
        self.steering = 0

        self.last_time = rospy.get_time()
        self.last_vel = 0

    def control(self, current_vel, angular_vel, linear_vel, dbw_enabled):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        current_vel = self.vel_lpf.filt(current_vel)

        self.steering = self.yaw.get_steering(linear_vel, angular_vel, current_vel)
        current_time = rospy.get_time()
        time_interval = current_time - self.last_time
        self.last_time = current_time
        
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        self.throttle = self.throttle_controller.step(vel_error, time_interval)
        self.brake = 0
        
        #stop the car
        if linear_vel == 0 and current_vel < 0.1:
            self.throttle = 0
            self.brake = 400
        elif self.throttle < 0.1 and vel_error < 0:
            self.throttle = 0
            decel = max(vel_error, self.decel_limit)

            #Torque - Neuton * meter
            self.brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return self.throttle, self.brake, self.steering
