#!/usr/bin/env python3
#
#   demoikintracker.py
#
#   Demonstrate/provide a structure for the ikin to track a point.
#   This assume the sixR.urdf!
#
#   Create a motion by continually sending joint values.  Also listen
#   to the point input.
#
#   Publish:   /joint_states      sensor_msgs/JointState
#   Subscribe: /point             geometry_msgs/PointStamped
#
import rospy
import numpy as np

from gazebodemos.kinematics2 import Kinematics
from sensor_msgs.msg         import JointState
from geometry_msgs.msg       import PointStamped


#
#  Point Subscriber
#
#  This listens for and saves the value of the last received point
#  message.  It isolate the ROS message subscriber to keep the main
#  code simpler.
#
class PointSubscriber:
    def __init__(self):
        # Instantiate the point p as a numpy vector.
        self.p = np.array([[0.0], [0.0], [0.0]])

        # Mark no arrived data.
        self.arrived = False

        # Create a subscriber to listen to point messages.
        rospy.Subscriber("point", PointStamped, self.process)

    def valid(self):
        return self.arrived

    def position(self):
        # Return the point.
        return self.p

    def process(self, msg):
        # Save the point contained in the message.
        self.p[0] = msg.point.x
        self.p[1] = msg.point.y
        self.p[2] = msg.point.z

        # Mark as valid.
        self.arrived = True


#
#  Joint States Publisher
#
#  Isolate the ROS message publisher to keep the main code simpler.
#
class JointStatePublisher:
    def __init__(self, names):
        # Save the dofs = number of names.
        self.n = len(names)

        # Create a publisher to send the joint values (joint_states).
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=100)

        # Wait until connected.  You don't have to wait, but the first
        # messages might go out before the connection and hence be lost.
        rospy.sleep(0.25)

        # Create a joint state message.
        self.msg = JointState()

        # You have to explicitly name each joint: Keep appending to
        # create a list.
        for i in range(self.n):
            self.msg.name.append(names[i])

        # We should also prepare the position list, initialize to zero.
        for i in range(self.n):
            self.msg.position.append(0.0)

        # Report.
        rospy.loginfo("Ready to publish /joint_states with %d DOFs", self.n)

    def dofs(self):
        # Return the number of DOFs.
        return self.n

    def send(self, q):
        # Set the positions.
        for i in range(self.n):
            self.msg.position[i] = q[i]

        # Send the command (with specified time).
        self.msg.header.stamp = rospy.Time.now()
        self.pub.publish(self.msg)
        

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
    return np.vstack((ep,eR))


#
#  Calculate the Desired
#
#  This computes the desired position and orientation, as well as the
#  desired translational and angular velocities for a given time.
#
def desired(t):
    # The point is simply taken from the subscriber.
    pd = point.position()
    vd = np.zeros((3,1))

    # The orientation is constant (at the zero orientation).
    Rd = np.array([[ -1,  0,  0],
                   [  0,  0,  1],
                   [  0,  1,  0]])
    wd = np.zeros((3,1))

    # Return the data.
    return (pd, Rd, vd, wd)



#
#  Main Code
#
if __name__ == "__main__":
    #
    #  LOGISTICAL SETUP
    #
    # Prepare the node.
    rospy.init_node('demoikintracker')
    rospy.loginfo("Starting the demo code for the ikin tracker...")

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
    pub = JointStatePublisher(('theta1', 'theta2', 'theta3',
                               'theta4', 'theta5', 'theta6'))

    # Make sure the URDF and publisher agree in dimensions.
    if not pub.dofs() == kin.dofs():
        rospy.logerr("FIX Publisher to agree with URDF!")

    # Set up the point subscriber.
    point = PointSubscriber()
    rospy.loginfo("Waiting for a point...")
    while not rospy.is_shutdown() and not point.valid():
        pass
    rospy.loginfo("Got a point.")

    # Set the numpy printing options (as not to be overly confusing).
    # This is entirely to make it look pretty (in my opinion).
    np.set_printoptions(suppress = True, precision = 6)


    #
    #  PICK AN INITIAL GUESS and INITIAL DESIRED
    #
    # Pick an initial joint position (pretty bad initial guess, but
    # away from the worst singularities).
    theta = np.array([[0.0], [0.0], [-1.5], [0.0], [0.0], [0.0]])

    # For the initial desired, head to the starting position (t=0).
    # Clear the velocities, just to be sure.
    (pd, Rd, vd, wd) = desired(0.0)
    vd = vd * 0.0
    wd = wd * 0.0


    #
    #  TIME LOOP
    #
    # I play one "trick": I start at (t=-1) and use the first second
    # to allow the poor initial guess to move to the starting point.
    #
    t   = -1.0
    tf  =  9.0
    lam =  0.1/dt
    while not rospy.is_shutdown():
        # Using the result theta(i-1) of the last cycle (i-1):
        # Compute the forward kinematics and the Jacobian.
        (p, R) = kin.fkin(theta)
        J      = kin.Jac(theta)

        # Use that data to compute the error (left after last cycle).
        e = etip(p, pd, R, Rd)

        
        # Advance to the new time step.
        t += dt

        # Compute the new desired.
        if t<0.0:
            # Note the negative time trick: Simply hold the desired at
            # the starting pose (t=0).  And enforce zero velocity.
            # This allows the initial guess to converge to the
            # starting position/orientation!
            (pd, Rd, vd, wd) = desired(0.0)
            vd = vd * 0.0
            wd = wd * 0.0
        else:
            (pd, Rd, vd, wd) = desired(t)

    
        # Build the reference velocity.
        vr = np.vstack((vd,wd)) + lam * e

        # Compute the Jacbian inverse (pseudo inverse)
        # Jpinv = np.linalg.pinv(J)
        Jinv = np.linalg.inv(J)

        # Update the joint angles.
        thetadot = Jinv @ vr
        theta   += dt * thetadot


        # Publish and sleep for the rest of the time.  You can choose
        # whether to show the initial "negative time convergence"....
        # if not t<0:
        pub.send(theta)
        servo.sleep()

        # Break the loop when we are finished.
        # if (t >= tf):
        #     break
