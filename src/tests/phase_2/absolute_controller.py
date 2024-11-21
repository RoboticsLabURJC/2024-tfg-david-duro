import math
from typing import Tuple, Union

import numpy as np

from smarts.core.chassis import AckermannChassis, BoxChassis
from smarts.core.coordinates import Pose
from smarts.core.utils.core_math import fast_quaternion_from_angle, radians_to_vec

from simple_pid import PID


class AbsoluteController:
    """A controller that sets a vehicle's linear and angular velocities using PID controllers
    for smoother transitions and reduced abrupt changes."""

    @classmethod
    def perform_action(
        cls,
        dt: float,
        vehicle,
        action: Union[float, Tuple[float, float]],
    ):
        """Performs an action adapting to the underlying chassis.
        Args:
            dt (float): Delta time value.
            vehicle: The vehicle to control.
            action (Union[float, Tuple[float, float]]): (speed) XOR (linear_velocity, angular_velocity).
        """
        chassis = vehicle.chassis

        if isinstance(action, (int, float)):
            # Special case: setting the initial speed
            if isinstance(chassis, BoxChassis):
                vehicle.control(vehicle.pose, action, dt)
            elif isinstance(chassis, AckermannChassis):
                chassis.speed = action  # Hack that calls control internally
            return
        assert isinstance(action, (list, tuple)) and len(action) == 2

        # PID controller parameters
        kp_linear, ki_linear, kd_linear = 1.0, 0.1, 0.01

        # Create and configure PID controllers
        linear_pid = PID(kp_linear, ki_linear, kd_linear, setpoint=action[0])

        # Extract target linear and angular velocities
        target_linear_velocity, target_angular_velocity = action

        # Get current linear velocity
        current_linear_velocity = vehicle.speed

        # Compute linear PID adjustment
        linear_pid.setpoint = target_linear_velocity
        linear_adjustment = linear_pid(current_linear_velocity)
        new_linear_velocity = current_linear_velocity + linear_adjustment * dt

        # Update the heading based on angular velocity
        target_heading = (vehicle.heading + target_angular_velocity * dt) % (2 * math.pi)

        if isinstance(chassis, BoxChassis):
            heading_vec = radians_to_vec(vehicle.heading)
            dpos = heading_vec * new_linear_velocity * dt
            new_pose = Pose(
                position=vehicle.position + np.append(dpos, 0.0),
                orientation=fast_quaternion_from_angle(target_heading),
            )
            vehicle.control(new_pose, new_linear_velocity, dt)

        elif isinstance(chassis, AckermannChassis):
            mass = chassis.mass_and_inertia[0]  # in kg
            wheel_radius = chassis.wheel_radius
            # XXX: should also take wheel inertia into account here
            # XXX: ... or else we should apply this force directly to the main link point of the chassis.
            if acceleration >= 0:
                # necessary torque is N*m = kg*m*acceleration
                torque_ratio = mass / (4 * wheel_radius * chassis.max_torque)
                throttle = np.clip(acceleration * torque_ratio, 0, 1)
                brake = 0
            else:
                throttle = 0 #np.clip(acceleration * torque_ratio, -1, 0)
                # necessary torque is N*m = kg*m*acceleration
                torque_ratio = mass / (4 * wheel_radius * chassis.max_btorque)
                brake = np.clip(acceleration * torque_ratio, 0, 1)

            steering = np.clip(dt * -angular_velocity * chassis.steering_ratio, -1, 1)
            vehicle.control(throttle=throttle, brake=brake, steering=steering)

        else:
            raise Exception("unsupported chassis type")
