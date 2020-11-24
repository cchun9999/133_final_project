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

from gazebodemos.kinematics2 import Kinematics
from std_msgs.msg            import Float64
from sensor_msgs.msg         import JointState
from geometry_msgs.msg       import PointStamped

from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import GetPhysicsProperties
from gazebo_msgs.msg import LinkState


#
#  Joint Command Publisher
#
#  Publish the commands on /joint_states (so RVIZ can see) as well as
#  on the /sevenbot/jointX_position/command topics (for Gazebo).
#
class JointCommandPublisher:
    def __init__(self, urdfnames, controlnames):
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
            topic = "/sevenbot/" + name + "/command"
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
    #eR = 0.5 * (cross(R[:,[0]], Rd[:,[0]]) +
    #            cross(R[:,[1]], Rd[:,[1]]) +
    #            cross(R[:,[2]], Rd[:,[2]]))
    # y axis points up, so we only need to correct that axis
    eR = cross(R[:,[1]], Rd)
    #eR = cross(R[:,[1]], np.array([[0], [0], [1]]))
    #eR = np.zeros((3,1)) for testing
    return np.vstack((ep,eR))


#
#  Calculate the Desired
#
#  This computes the desired position and orientation, as well as the
#  desired translational and angular velocities for a given time.
#
def desired(t, spline):
    # spline_x, spline_y are made to be optional because
    # they are only used in the forward pass. 
    # TODO: Clean this up.

    # The point is simply taken from the subscriber.
    pd = call_spline_pos(spline, t)
    vd = call_spline_vel(spline, t)

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

def call_spline_pos(t, spline):
    # Return the position for the given spline at
    # the given time
    return np.array([np.array([1, t, t**2, t**3]) @ spline]).T

def call_spline_vel(t, spline):
    # Return the velocity for the given spline at
    # the given time
    return np.array([np.array([0, 1, 2*t, 3*t**2]) @ spline]).T

def get_paddle_velocity(initial_vel, final_vel, paddle_mass, ball_mass, e):
    '''
    Parameters
    ----------
    initial_velocity : 3x1 numpy array
        velocity in the world frame of the ball before the collision
    final_velocity : 3x1 numpy array
        velocity in the world frame of the ball after the collision
    paddle_mass : double
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
    print("paddle_vel_norm:", paddle_vel_norm)
    paddle_vel = paddle_vel_norm * J_unit
    return paddle_vel, J_unit

#
#  Main Code
#
def main():
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
    urdf = rospy.get_param('/robot_description')
    kin  = Kinematics(urdf, 'world', 'tip')
    N    = kin.dofs()
    rospy.loginfo("Loaded URDF for %d joints" % N)

    # Set up the publisher, naming the joints!
    pub = JointCommandPublisher(('theta1', 'theta2', 'theta3', 'theta4',
                                 'theta5', 'theta6', 'theta7'),
                                ('joint1_position_controller',
                                 'joint2_position_controller',
                                 'joint3_position_controller',
                                 'joint4_position_controller',
                                 'joint5_position_controller',
                                 'joint6_position_controller',
                                 'joint7_position_controller'))

    # Make sure the URDF and publisher agree in dimensions.
    if not pub.dofs() == kin.dofs():
        rospy.logerr("FIX Publisher to agree with URDF!")

    # Set the numpy printing options (as not to be overly confusing).
    # This is entirely to make it look pretty (in my opinion).
    np.set_printoptions(suppress = True, precision = 6)

    # Get a model state service proxy to be able to query state of the ball
    model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
    # Get the gravity constant from gazebo
    physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
    grav = physics().gravity.z

    #
    #  PICK AN INITIAL GUESS and INITIAL DESIRED
    #
    # Pick an initial joint position (pretty bad initial guess, but
    # away from the worst singularities).
    #theta = np.zeros((7, 1))
    theta = np.array([[0.0], [0.0], [0.0], [-0.05], [0.0], [0.0], [0.0]])

    #Pick paddle intercept height
    intercept_height = 0.6

    #Masses
    ball_mass = 1
    paddle_mass = 100

    #Coefficient of Restitution
    restitution = 1

    windup_ratio = 0.05

    #Used to track the velocity between splines
    v_initial = np.zeros((3, 1))

    #Ball Targets
    target_x = 0
    target_y = 0.95
    target_max_height = 2
    t_arc = np.sqrt(-2 * target_max_height / grav)



    

    # Set fraction of time for path used to hit ball
    t_frac_hit = 0.3

    gamma_squared = 1e-6


    #
    #  TIME LOOP
    #
    # I play one "trick": I start at (t=-1) and use the first second
    # to allow the poor initial guess to move to the starting point.
    #
    tf_default = 0.8
    lam =  0.1/dt
    num_iters = 0
    while not rospy.is_shutdown():
        # Using the result theta(i-1) of the last cycle (i-1): Compute
        # the fkin(theta(i-1)) and the Jacobian J(theta(i-1)).
        num_iters += 1
        t = 0
        tf_ball = None

        # Get information on the state of the ball. Use this to compute the
        # time it will take the ball to reach the z = intercept_height point so that we can
        # plan the trajectory of the robot to hit it at the right time.
        ball_state = model_state("ball", "link")

        ball_vz = ball_state.twist.linear.z
        ball_z = ball_state.pose.position.z
        tf_ball = max(np.roots(np.array([1/2*grav, ball_vz, ball_z - intercept_height])))
        if not np.isreal(tf_ball): # default to tf_default if failed to find a root
            tf_ball = tf_default

        # Compute the projected final x, y positions of the ball
        ball_xf = ball_state.pose.position.x + ball_state.twist.linear.x * tf_ball 
        ball_yf = ball_state.pose.position.y + ball_state.twist.linear.y * tf_ball
        ball_posf = np.array([[ball_xf], [ball_yf], [intercept_height]])

        ball_vel_final = np.array([[ball_state.twist.linear.x], [ball_state.twist.linear.y], [ball_vz + grav * tf_ball]])
        ball_vel_desired = np.array([[(target_x - ball_xf)/t_arc], [(target_y - ball_yf)/t_arc], [np.sqrt(2 * (target_max_height - intercept_height))]])
        print("ball_vel_final", ball_vel_final)
        print("ball_vel_desired", ball_vel_desired)
        paddle_hit_vel, paddle_hit_rot = get_paddle_velocity(ball_vel_final, ball_vel_desired, paddle_mass, ball_mass, restitution)
        windup_pos = ball_posf - windup_ratio * paddle_hit_vel

        # Compute the desired splines for each dimension of the paddle tip.
        # Use the initial tip position of p0.
        # z is forced to go between 0.6 and 0.3
        if num_iters > 1:
            p0, R0 = kin.fkin(theta)
        else:
            p0 = np.array([[0], [0.95], [intercept_height]])

        print("windup pos:", windup_pos)

        spline_1 = compute_spline(tf_ball * (1 - t_frac_hit), np.array([p0.flatten(), v_initial.flatten(), windup_pos.flatten(), [0,0,0]]))

        # Windup
        for t in np.arange(0, tf_ball * (1 - t_frac_hit), dt):

            # TODO: The instability right at the beginning stems from here because the
            # actual original configuration is in a singularity but we set theta to not
            # be in one, can fix this with the negative time trick, but then have to 
            # account for the ball falling at the same time.
            (p, R) = kin.fkin(theta)
            J      = kin.Jac(theta)
            #Get desired values at given time
            pd = call_spline_pos(t, spline_1)
            vd = call_spline_vel(t, spline_1)

            Rd = paddle_hit_rot
            wd = np.zeros((3, 1))

            # Determine the residual error.
            e = etip(p, pd, R, Rd)

            # Build the reference velocity.
            vr = np.vstack((vd,wd)) + lam * e

            # Compute the Jacbian inverse (pseudo inverse)
            #Jpinv = np.linalg.pinv(J)
            weighted_inv = (np.linalg.inv(J.T@J + (gamma_squared)*np.identity(N)))@J.T

            # Update the joint angles.
            thetadot = weighted_inv @ vr
            theta   += dt * thetadot

            pub.send(theta)
            servo.sleep()

        spline_2 = compute_spline(tf_ball * t_frac_hit, np.array([windup_pos.flatten(), [0,0,0], ball_posf.flatten(), paddle_hit_vel.flatten()]))

        # Hit pass
        for t in np.arange(0, tf_ball * t_frac_hit, dt):
            (p, R) = kin.fkin(theta)
            J      = kin.Jac(theta)

            pd = call_spline_pos(t, spline_2)
            vd = call_spline_vel(t, spline_2)

            Rd = paddle_hit_rot
            wd = np.zeros((3, 1))

            # Determine the residual error.
            e = etip(p, pd, R, Rd)

            # Build the reference velocity.
            vr = np.vstack((vd,wd)) + lam * e
            weighted_inv = (np.linalg.inv(J.T@J + (gamma_squared)*np.identity(N)))@J.T

            # Update the joint angles.
            thetadot = weighted_inv @ vr
            theta   += dt * thetadot

            pub.send(theta)
            servo.sleep()

        #Follow through
        
        for t in np.arange(0, 2 * dt, dt):
            (p, R) = kin.fkin(theta)
            J      = kin.Jac(theta)

            pd = ball_posf + paddle_hit_vel * t
            vd = paddle_hit_vel

            Rd = paddle_hit_rot
            wd = np.zeros((3, 1))

            # Determine the residual error.
            e = etip(p, pd, R, Rd)

            # Build the reference velocity.
            vr = np.vstack((vd,wd)) + lam * e
            weighted_inv = (np.linalg.inv(J.T@J + (gamma_squared)*np.identity(N)))@J.T

            # Update the joint angles.
            thetadot = weighted_inv @ vr
            theta   += dt * thetadot

            pub.send(theta)
            servo.sleep()
        

        v_initial = paddle_hit_vel
        servo.sleep()
        servo.sleep()
        servo.sleep()

if __name__ == "__main__":
    main()
