#!/usr/bin/env python3
#
#   demo_gazebo_ikintracker.py
#
#   Demonstrate the IKin Tracking in Gazebo.  This uses the PD
#   position controllers, so the model *has* velocity and the impact
#   is as expected.  This assume the seveneffort.urdf!
#
#   Create a motion by continually sending joint values.  Also listen
#   to the point input.
#
#   Subscribe: /point                               geometry_msgs/PointStamped
#   Publish:   /joint_states                        sensor_msgs/JointState
#   Publish:   /sevenbot/joint1_position_controller/command    std_msgs/Float64
#   Publish:   /sevenbot/joint2_position_controller/command    std_msgs/Float64
#   Publish:   /sevenbot/joint3_position_controller/command    std_msgs/Float64
#   Publish:   /sevenbot/joint4_position_controller/command    std_msgs/Float64
#   Publish:   /sevenbot/joint5_position_controller/command    std_msgs/Float64
#   Publish:   /sevenbot/joint6_position_controller/command    std_msgs/Float64
#   Publish:   /sevenbot/joint7_position_controller/command    std_msgs/Float64
#
import rospy
import numpy as np
import matplotlib.pyplot as plt

from gazebodemos.kinematics2 import Kinematics
from std_msgs.msg            import Float64
from sensor_msgs.msg         import JointState
from geometry_msgs.msg       import PointStamped
from geometry_msgs.msg import Vector3

from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import GetPhysicsProperties
from gazebo_msgs.srv import SetPhysicsProperties
from gazebo_msgs.msg import LinkState
from gazebo_msgs.msg import ODEPhysics

from pyquaternion import Quaternion


#
#  Joint Command Publisher
#
#  Publish the commands on /joint_states (so RVIZ can see) as well as
#  on the /sevenbot/jointX_position/command topics (for Gazebo).
#
class JointCommandPublisher:
    def __init__(self, urdfnames, controlnames, topic_root):
        # Make sure the name lists have equal length.
        assert len(urdfnames) == len(controlnames), "Unequal lengths"

        # Save the dofs = number of names/channels.
        self.n = len(urdfnames)

        # Create a publisher to /joint_states and pre-populate the message.
        self.pubjs = rospy.Publisher("/joint_states", JointState, queue_size=100)
        self.msgjs = JointState()
        for name in urdfnames:
            self.msgjs.name.append(name)
            self.msgjs.position.append(0.0)

        # Prepare a list of publishers for each joint commands.
        self.pubX  = []
        for name in controlnames:
            topic = "/" + topic_root + "/" + name + "/command"
            self.pubX.append(rospy.Publisher(topic, Float64, queue_size=100))

        # Wait until connected.  You don't have to wait, but the first
        # messages might go out before the connection and hence be lost.
        # rospy.sleep(0.5)

        # Report.
        rospy.loginfo("Ready to publish command for %d DOFs", self.n)

    def dofs(self):
        # Return the number of DOFs.
        return self.n

    def send(self, q):
        # Send each individual command and populate the joint_states.
        for i in range(self.n):
            self.pubX[i].publish(Float64(q[i]))
            self.msgjs.position[i] = q[i]

        # Send the command (with specified time).
        self.msgjs.header.stamp = rospy.Time.now()
        self.pubjs.publish(self.msgjs)


#
#  Basic Rotation Matrices
#
#  Note the angle is specified in radians.
#
def Rx(phi):
    return np.array([[ 1, 0          , 0          ],
                     [ 0, np.cos(phi),-np.sin(phi)],
                     [ 0, np.sin(phi), np.cos(phi)]])

def Ry(phi):
    return np.array([[ np.cos(phi), 0, np.sin(phi)],
                     [ 0          , 1, 0          ],
                     [-np.sin(phi), 0, np.cos(phi)]])

def Rz(phi):
    return np.array([[ np.cos(phi),-np.sin(phi), 0],
                     [ np.sin(phi), np.cos(phi), 0],
                     [ 0          , 0          , 1]])

#
#  Simple Vector Utilities
#
#  Just collect a 3x1 column vector, perform a dot product, or a cross product.
#
def vec(x,y,z):
    return np.array([[x], [y], [z]])

def dot(a,b):
    return a.T @ b

def cross(a,b):
    return np.cross(a, b, axis=0)


#
#  6x1 Error Computation
#
#  Note the 3x1 translation is on TOP of the 3x1 rotation error!
#
#  Also note, the cross product does not return a column vector but
#  just a one dimensional array.  So we need to convert into a 2
#  dimensional matrix, and transpose into the column form.  And then
#  we use vstack to stack vertically...
#
def etip(p, pd, R, Rd):
    ep = pd - p
    eR = 0.5 * (cross(R[:,[0]], Rd[:,[0]]) +
                cross(R[:,[1]], Rd[:,[1]]) +
                cross(R[:,[2]], Rd[:,[2]]))
    #eR = 0.5 * (cross(R[:,[1]], Rd[:, [1]]))
    return np.vstack((ep,eR))


#
#  Calculate the Desired
#
#  This computes the desired position and orientation, as well as the
#  desired translational and angular velocities for a given time.
#
def desired(t, spline_z, spline_x=None, spline_y=None):
    # spline_x, spline_y are made to be optional because
    # they are only used in the forward pass. 
    # TODO: Clean this up.

    # The point is simply taken from the subscriber.
    pd = np.array([[0], [0.90], [0.6]])
    pd[2] = call_spline_pos(spline_z, t)
    if spline_x is not None and spline_y is not None:
        pd[0] = call_spline_pos(spline_x, t)
        pd[1] = call_spline_pos(spline_y, t)
        
    vd = np.zeros((3,1))
    vd[2] = call_spline_vel(spline_z, t)
    if spline_x is not None and spline_y is not None:
        vd[0] = call_spline_vel(spline_x, t)
        vd[1] = call_spline_vel(spline_y, t)

    # The orientation is constant (at the zero orientation).
    Rd = np.array([[ -1,  0,  0],
                   [  0,  0,  1],
                   [  0,  1,  0]])
    wd = np.zeros((3,1))

    # Return the data.
    return (pd, Rd, vd, wd)


def compute_spline(tf, control_points):
    # Calculate the coefficients of the desired
    # spline.

    # Control points is array of [p0, v0, pf, vf]
    Y = np.array([[1, 0, 0, 0], 
                 [0, 1, 0, 0],
                 [1, 1*tf, 1*tf**2, 1*tf**3],
                 [0, 1, 2*tf, 3*tf**2]])
    spline = np.linalg.inv(Y) @ control_points
    return spline

def call_spline_pos(spline, t):
    # Return the position for the given spline at
    # the given time
    return spline.T @ np.array([1, t, t**2, t**3])

def call_spline_vel(spline, t):
    # Return the velocity for the given spline at
    # the given time
    return spline.T @ np.array([0, 1, 2*t, 3*t**2])

def get_desired_paddle_velocity(initial_vel, final_vel, paddle_mass, ball_mass, e):
    '''
    Parameters
    ----------
    initial_velocity : 3x1 numpy array
        velocity in the world frame of the ball before the collision
    final_velocity : 3x1 numpy array
        velocity in the world frame of the ball after the collision
    paddel_mass : double
        Mass of the paddle
    ball_mass : double
        Mass of the ball
    e : double
        Coefficient of restitution between 0 and 1
    Returns
    -------
    3x1 numpy array representing the velocity of the paddle
    3x1 numpy array unit vector representing the orientation of the z axis
    '''
    
    J = ball_mass * (final_vel - initial_vel)
    J_norm = np.linalg.norm(J)
    J_unit = J / J_norm
    paddle_vel_norm = J_norm * (1/paddle_mass + 1/ball_mass) / (1 + e) + initial_vel.T @ J_unit
    paddle_vel = paddle_vel_norm * J_unit
    return paddle_vel, J_unit

def change_gravity(x, y, z):
    gravity = Vector3()
    gravity.x = x
    gravity.y = y
    gravity.z = z
    set_gravity = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
    set_gravity(physics().time_step, physics().max_update_rate, gravity, ODEPhysics())

# Get the current paddle velocity from Gazebo
# Account for tip transformations
def get_paddle_velocity(robot_name):
    paddle_state = model_state(robot_name, "link7")
    paddle_vel_transf = np.array([paddle_state.twist.linear.x, paddle_state.twist.linear.y,
                                  paddle_state.twist.linear.z])
    R = Rx(np.pi)
    paddle_vel = paddle_vel_transf.T @ R
    return paddle_vel.T.flatten()

# Get the current paddle velocity from Gazebo
# Account for tip transformations
def get_paddle_pos(robot_name):
    paddle_state = model_state(robot_name, "link7")
    paddle_pos_transf = np.array([paddle_state.pose.position.x, paddle_state.pose.position.y,
                                  paddle_state.pose.position.z])
    R = Rx(np.pi)
    paddle_pos = paddle_pos_transf.T @ R
    return paddle_pos.T

def compute_transform_quaternion(vec_s, vec_t):
    vec_s /= np.linalg.norm(vec_s)
    vec_t /= np.linalg.norm(vec_t)
    axis = np.cross(vec_s, vec_t)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(vec_s, vec_t))
    quat = Quaternion(axis=axis, angle=angle)
    return axis.reshape((-1, 1)), quat

def compute_final_roation(paddle_normal, ball_xf, ball_yf):
    vec_y = paddle_normal
    z_z = -(vec_y[0][0] * ball_xf + vec_y[1][0] * ball_yf) / vec_y[2][0]
    vec_z = np.array([[ball_xf], [ball_yf], [z_z]])
    vec_z = vec_z / np.linalg.norm(vec_z)
    vec_x = cross(vec_y, vec_z)
    return np.hstack((vec_x, vec_y, vec_z))


#####
# 7DOF robot class used to encapsulate desired robot and its parameters.
# Specifically useful for creating identical robots.
####
class SevenDOFRobot:

    def __init__(self, robot_description, robot_name):
        self.urdf = rospy.get_param(robot_description)
        self.kin  = Kinematics(self.urdf, 'world', 'tip')
        self.N    = self.kin.dofs()
        rospy.loginfo("Loaded URDF for %d joints" % self.N)
        self.name = robot_name
        print("Initializing Robot: ", robot_name)

        self.pub = JointCommandPublisher(('theta1', 'theta2', 'theta3', 'theta4',
                                 'theta5', 'theta6', 'theta7'),
                                ('joint1_position_controller',
                                 'joint2_position_controller',
                                 'joint3_position_controller',
                                 'joint4_position_controller',
                                 'joint5_position_controller',
                                 'joint6_position_controller',
                                 'joint7_position_controller'),
                                topic_root=robot_name)
        # Make sure the URDF and publisher agree in dimensions.
        if not self.pub.dofs() == self.kin.dofs():
            rospy.logerr("FIX Publisher to agree with URDF!")

    def compute_projected_ball_xy(self, intercept_height):
        # Get information on the state of the ball. Use this to compute the
        # time it will take the ball to reach the z = 0.6 point so that we can
        # plan the trajectory of the robot to hit it at the right time.
        ball_state = model_state("ball", "link")
        ball_vel = np.array([ball_state.twist.linear.x, ball_state.twist.linear.y, ball_state.twist.linear.z])
        ball_z = ball_state.pose.position.z
        ball_solns = np.roots(np.array([1/2*grav, ball_vel[2], ball_z - intercept_height]))
        print("Ball solns:", ball_solns)
        tf_ball = max(ball_solns)
        if not np.isreal(tf_ball):
            # default to tf if failed to find a root
            print('failed to find root')
            tf_ball = tf_default
        print('projected tf:', tf_ball)
        
        # Compute the projected final x, y positions of the ball
        ball_xf = ball_state.pose.position.x + ball_vel[0] * tf_ball 
        ball_yf = ball_state.pose.position.y + ball_vel[1] * tf_ball
        print(f"ball projected (x,y): {ball_xf, ball_yf}")

        print("ball landing radius:" ,np.sqrt(ball_xf **2 + ball_yf **2))

        max_height_t = -ball_vel[2] / grav
        max_height_actual = 1/2 * grav * max_height_t ** 2 + ball_vel[2] * max_height_t + ball_z
        print("max_height_prediction:", max_height_actual)

        return ball_xf, ball_yf, tf_ball, ball_vel

    def compute_intermediate_quaternions(self, theta, paddle_hit_rot, tf_ball):
        p0, R = self.kin.fkin(theta)
        print("R:\n", R)
        # Compute the Orientation interpolation using quaternions
        quat_axis, transform_quat = compute_transform_quaternion(R[:, 1], paddle_hit_rot.flatten())
        print("Transform quat rotation matrix:\n", transform_quat.rotation_matrix)
        current_quat = Quaternion(matrix=R)
        target_R = transform_quat.rotation_matrix @ current_quat.rotation_matrix
        target_quat = Quaternion(matrix=target_R)
        num_intermediates = int(tf_ball/(2*dt)) + 1
        intermediate_quats = Quaternion.intermediates(current_quat, target_quat, num_intermediates, 
                                                      include_endpoints=False)
        return intermediate_quats, target_quat

    def execute_motion(self, t, theta, desired_rot, splines):
        # Forward pass: Move Z of tip from 0.6 to 0.4 in the desired amount of time.
        #for t in np.arange(0, tf_ball/2+1e-6, dt):
        (p, R) = self.kin.fkin(theta)
        J      = self.kin.Jac(theta)

        (pd, Rd, vd, wd) = desired(t, splines[0], splines[1], splines[2])
        Rd = desired_rot
        # Determine the residual error.
        e = etip(p, pd, R, Rd)
        # Build the reference velocity.
        vr = np.vstack((vd,wd)) + lam * e
        # Compute the Jacbian inverse (pseudo inverse)
        weighted_inv = (np.linalg.inv(J.T@J + (gamma**2)*np.identity(N)))@J.T

        # Update the joint angles.
        thetadot = weighted_inv @ vr

        # Add Secondary Task
        #theta_center = np.array([[-0], [-0], [0], [-0], [0], [0], [0]])
        #heta_dot_secondary = -(theta - theta_center)
        theta_dot_secondary = np.zeros((N, 1))
        theta_dot_secondary[2] = -theta[2]
        #theta_dot_secondary[3] = -theta[3]
        thetadot += (np.identity(weighted_inv.shape[0])-weighted_inv@J)@theta_dot_secondary
        theta   += dt * thetadot

        return theta + dt*thetadot

#
#  Main Code
#

# def main():
if __name__ == "__main__":
    #
    #  LOGISTICAL SETUP
    #
    # Prepare the node.
    rospy.init_node('demo_gazebo_ikintracker')
    rospy.loginfo("Starting the demo code for the IKin tracking...")

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running with a loop dt of %f seconds (%fHz)" %
                  (dt, rate))

    # Set up the kinematics, from world to tip.
    robot1 = SevenDOFRobot("/robot_description", "sevenbot")
    robot2 = SevenDOFRobot("/2nd_robot_description", "sevenbot2")
    N = robot1.N

    # Set the numpy printing options (as not to be overly confusing).
    # This is entirely to make it look pretty (in my opinion).
    np.set_printoptions(suppress = True, precision = 6)


    #
    #  PICK AN INITIAL GUESS and INITIAL DESIRED
    #
    # Pick an initial joint position (pretty bad initial guess, but
    # away from the worst singularities).
    #theta = np.zeros((7, 1))
    theta1 = np.array([[0.0], [0.0], [0.0], [-0.05], [0.0], [0.0], [0.0]])
    theta2 = np.array([[0.0], [0.0], [0.0], [-0.05], [0.0], [0.0], [0.0]])

    intercept_height = .45

    # For the initial desired, head to the starting position (t=0).
    # Clear the velocities, just to be sure.
    (pd, Rd, vd, wd) = (np.zeros((3, 1)), np.zeros((3, 3)), np.zeros((3, 1)), np.zeros((3, 1)))
    vd = vd * 0.0
    wd = wd * 0.0
    pd = np.array([[0], [0.90], [intercept_height]])

    # Get a model state service proxy to be able to query state of the ball
    model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    # Get the gravity constant from gazebo
    physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
    grav = physics().gravity.z

    # Test changing gravity (if uncommented with these values then the ball shouldn't move
    # during the simulation)
    #change_gravity(0,0,0)
    r1_target_x = 4
    r1_target_y = 1.2

    r2_target_x = 0
    r2_target_y = 1.2

    # Function to return target (x,y) of ball at given time
    get_target_xy = lambda t: (1.2 * np.sin(t), 1.2 * np.cos(t))
    
    target_max_height = 1.5
    max_height_vel = np.sqrt(-2 * grav * (target_max_height - intercept_height))
    print("max_height_vel", max_height_vel)
    t_arc =  -2 *max_height_vel / grav
    #t_arc = -2 * max_height_vel / grav
    print("t_arc:", t_arc)

    #Masses
    ball_mass = 0.0002
    paddle_mass = 0.2

    #Coefficient of Restitution
    restitution = 1.0

    windup_ratio = 0.5
    gamma = 1e-3

    pullback_ratio = 0.16
    
    z_vel_array = []
    max_height_array = []
    ball_vel_desired = np.array([[0], [0], [0]])
    t_follow = 0.15

    correction_angle = -0/15 * np.pi / 180
    correction_matrix = Rx(correction_angle)


    #
    #  TIME LOOP
    #
    # I play one "trick": I start at (t=-1) and use the first second
    # to allow the poor initial guess to move to the starting point.
    #
    lam =  0.1/dt
    num_iters = 0
    while (not rospy.is_shutdown()):
        # Using the result theta(i-1) of the last cycle (i-1): Compute
        # the fkin(theta(i-1)) and the Jacobian J(theta(i-1)).
        num_iters += 1
        print()
        print("ITER #:", num_iters)
        
        tf_default = 0.8

        print("========================ROBOT1=====================")

        ball_xf, ball_yf, tf_ball, ball_vel = robot1.compute_projected_ball_xy(intercept_height)

        
       
        
        ball_vel_final = np.array([[ball_vel[0]], [ball_vel[1]], [ball_vel[2] + grav * tf_ball]])
        ball_vel_desired = np.array([[(r1_target_x - ball_xf)/t_arc], [(r1_target_y - ball_yf)/t_arc], [max_height_vel]])
        
        paddle_hit_vel, paddle_hit_rot = get_desired_paddle_velocity(ball_vel_final, ball_vel_desired, paddle_mass, ball_mass, restitution)

        p, R = robot1.kin.fkin(theta1)


        ball_posf = np.array([[ball_xf], [ball_yf], [intercept_height]])
        t_windup = tf_ball * windup_ratio
        t_hit = tf_ball * (1 - windup_ratio)
        windup_pos = ball_posf - pullback_ratio * paddle_hit_vel
        print("ball posf:\n", ball_posf)
        print("windup pos:\n", windup_pos)
        print("tf ball:\n", tf_ball)

        #Constant distance windup
        #windup_pos = ball_posf - 0.15 * paddle_hit_vel / np.linalg.norm(paddle_hit_vel)

        # Use the current paddle velocity as initial velocity to ensure smooth
        # trajectory
        paddle_vel = get_paddle_velocity(robot1.name)
        spline_x = compute_spline(t_windup, np.array([p[0], [paddle_vel[0]], windup_pos[0], paddle_hit_vel[0]]))
        spline_y = compute_spline(t_windup, np.array([p[1], [paddle_vel[1]], windup_pos[1], paddle_hit_vel[1]]))
        spline_z = compute_spline(t_windup, np.array([p[2], [paddle_vel[2]], windup_pos[2], paddle_hit_vel[2]]))
        
        # Compute rotation matrix trajectory
        intermediate_quats, target_quat = robot1.compute_intermediate_quaternions(theta1, paddle_hit_rot, tf_ball)

        # Forward (Down) pass. Move the ball from z = 0.6 to z = 0.4 and go to the desired ball_xf, ball_yf. Also orient
        # the paddle in the correct manner
        for t in np.arange(dt, t_windup, dt):
            desired_rot = next(intermediate_quats).rotation_matrix
            theta1 = robot1.execute_motion(t, theta1, desired_rot, [spline_z, spline_x, spline_y])
            robot1.pub.send(theta1)
            servo.sleep()

        # Compute the z-spline for the upwards motion. X and Y stay constant
        p, R = robot1.kin.fkin(theta1)
        paddle_vel = get_paddle_velocity(robot1.name)
        spline_x = compute_spline(t_hit, np.array([p[0], [paddle_vel[0]], ball_posf[0], paddle_hit_vel[0]]))
        spline_y = compute_spline(t_hit, np.array([p[1], [paddle_vel[1]], ball_posf[1], paddle_hit_vel[1]]))
        spline_z = compute_spline(t_hit, np.array([p[2], [paddle_vel[2]], ball_posf[2], paddle_hit_vel[2]]))

        # Windup
        for t in np.arange(dt, t_hit, dt):
            theta1 = robot1.execute_motion(t, theta1, target_quat.rotation_matrix, [spline_z, spline_x, spline_y])
            robot1.pub.send(theta1)
            servo.sleep()
        
        # Follow through pass
        print("doing follow through on robot 1")
        p, R = robot1.kin.fkin(theta1)
        paddle_vel = get_paddle_velocity(robot1.name)
        follow_pos = 1/2 * t_follow * paddle_hit_vel + p
        spline_x = compute_spline(t_follow, np.array([p[0], [paddle_vel[0]], follow_pos[0], [0]]))
        spline_y = compute_spline(t_follow, np.array([p[1], [paddle_vel[1]], follow_pos[1], [0]]))
        spline_z = compute_spline(t_follow, np.array([p[2], [paddle_vel[2]], follow_pos[2], [0]]))
        for t in np.arange(dt, t_follow, dt):
            theta1 = robot1.execute_motion(t, theta1, target_quat.rotation_matrix, [spline_z, spline_x, spline_y])
            robot1.pub.send(theta1)
            servo.sleep()


        ############################ Robot 2 ############################
        print("====================ROBOT2====================")
        
        ball_xf, ball_yf, tf_ball, ball_vel = robot2.compute_projected_ball_xy(intercept_height)

        ball_vel_final = np.array([[ball_vel[0]], [ball_vel[1]], [ball_vel[2] + grav * tf_ball]])
        ball_vel_desired = np.array([[(r2_target_x - ball_xf)/t_arc], [(r2_target_y - ball_yf)/t_arc], [np.sqrt(2 * (target_max_height - intercept_height))]])
        
        paddle_hit_vel, paddle_hit_rot = get_desired_paddle_velocity(ball_vel_final, ball_vel_desired, paddle_mass, ball_mass, restitution)
        p, R = robot2.kin.fkin(theta2)
        print("p:\n", p)
        # x0, y0, z0 = p0.flatten

        ball_posf = np.array([[ball_xf], [ball_yf], [intercept_height]])
        t_windup = tf_ball * windup_ratio
        t_hit = tf_ball * (1 - windup_ratio)
        windup_pos = ball_posf - pullback_ratio * paddle_hit_vel
        print("ball posf:\n", ball_posf)
        print("windup pos:\n", windup_pos)
        print("tf ball:\n", tf_ball)

        #Constant distance windup
        #windup_pos = ball_posf - 0.15 * paddle_hit_vel / np.linalg.norm(paddle_hit_vel)

        # Use the current paddle velocity as initial velocity to ensure smooth
        # trajectory
        paddle_vel = get_paddle_velocity(robot2.name)
        print("get paddle vel:\n", paddle_vel)
        print("paddle hit vel:\n", paddle_hit_vel)
        spline_x = compute_spline(t_windup, np.array([p[0], [paddle_vel[0]], windup_pos[0], paddle_hit_vel[0]]))
        spline_y = compute_spline(t_windup, np.array([p[1], [paddle_vel[1]], windup_pos[1], paddle_hit_vel[1]]))
        spline_z = compute_spline(t_windup, np.array([p[2], [paddle_vel[2]], windup_pos[2], paddle_hit_vel[2]]))
        print("mid x spline pos: ", call_spline_pos(spline_x, t_windup/2))
        print("mid y spline pos: ", call_spline_pos(spline_y, t_windup/2))
        print("mid z spline pos: ", call_spline_pos(spline_z, t_windup/2))
        
        # Compute rotation matrix trajectory
        intermediate_quats, target_quat = robot2.compute_intermediate_quaternions(theta2, paddle_hit_rot, tf_ball)

        # Forward (Down) pass. Move the ball from z = 0.6 to z = 0.4 and go to the desired ball_xf, ball_yf. Also orient
        # the paddle in the correct manner
        for t in np.arange(dt, t_windup, dt):
            desired_rot = next(intermediate_quats).rotation_matrix
            theta2 = robot2.execute_motion(t, theta2, desired_rot, [spline_z, spline_x, spline_y])
            robot2.pub.send(theta2)
            servo.sleep()

        # Compute the z-spline for the upwards motion. X and Y stay constant
        p, R = robot2.kin.fkin(theta2)
        paddle_vel = get_paddle_velocity(robot2.name)
        spline_x = compute_spline(t_hit, np.array([p[0], [paddle_vel[0]], ball_posf[0], paddle_hit_vel[0]]))
        spline_y = compute_spline(t_hit, np.array([p[1], [paddle_vel[1]], ball_posf[1], paddle_hit_vel[1]]))
        spline_z = compute_spline(t_hit, np.array([p[2], [paddle_vel[2]], ball_posf[2], paddle_hit_vel[2]]))

        # Backward (Up) pass: Move the paddle back up to original z, ending with a hit of the ball.
        # Don't change x or y since they should already be in the target orientation.
        for t in np.arange(dt, t_hit, dt):
            theta2 = robot2.execute_motion(t, theta2, target_quat.rotation_matrix, [spline_z, spline_x, spline_y])
            robot2.pub.send(theta2)
            servo.sleep()
        
        # Follow through pass
        p, R = robot2.kin.fkin(theta2)
        paddle_vel = get_paddle_velocity(robot2.name)
        follow_pos = 1/2 * t_follow * paddle_hit_vel + p
        spline_x = compute_spline(t_follow, np.array([p[0], [paddle_vel[0]], follow_pos[0], [0]]))
        spline_y = compute_spline(t_follow, np.array([p[1], [paddle_vel[1]], follow_pos[1], [0]]))
        spline_z = compute_spline(t_follow, np.array([p[2], [paddle_vel[2]], follow_pos[2], [0]]))
        for t in np.arange(dt, t_follow, dt):
            
            theta2 = robot2.execute_motion(t, theta2, target_quat.rotation_matrix, [spline_z, spline_x, spline_y])

            # Publish and sleep for the rest of the time.  You can choose
            # whether to show the initial "negative time convergence"....
            # if not t<0:

            #ball_state = model_state("ball", "link")
            #z_array.append(ball_state.pose.position.z)
            robot2.pub.send(theta2)
            servo.sleep()