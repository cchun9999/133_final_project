#!/usr/bin/env python3
#
#   demoatlas.py
#
#   Demonstrate Atlas.  This assumes the atlas_v5.urdf!
#
#   Create a motion by continually sending joint values.  Also listen
#   to the point input.
#
#   Publish:   /joint_states      sensor_msgs/JointState
#   Subscribe: /point             geometry_msgs/PointStamped
#
import rospy
import numpy as np
import copy

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
    rospy.init_node('demoatlas')
    rospy.loginfo("Starting the demo code for the Atlas robot...")

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running with a loop dt of %f seconds (%fHz)" %
                  (dt, rate))

    # Set up any instances of the kinematics, for the segment (or
    # segments) that you wish.
    urdf = rospy.get_param('/robot_description')
    kinleftleg = Kinematics(urdf, 'l_foot', 'pelvis')
    Nleftleg   = kinleftleg.dofs()
    rospy.loginfo("Setup Atlas Left Leg with %d joints" % Nleftleg)

    # Report on the kinematic chains.
    kinleftleg.report(rospy.loginfo)
    print(kinleftleg.jointnames())

    # Other names
    otherjointnames = ['r_leg_akx',
                       'r_leg_aky',
                       'r_leg_kny',
                       'r_leg_hpy',
                       'r_leg_hpx',
                       'r_leg_hpz',
                       'back_bkx',
                       'back_bky',
                       'back_bkz',
                       'l_arm_shx',
                       'l_arm_shz',
                       'l_arm_elx',
                       'l_arm_ely',
                       'l_arm_wrx',
                       'l_arm_wry',
                       'l_arm_wry2',
                       'r_arm_shx',
                       'r_arm_shz',
                       'r_arm_elx',
                       'r_arm_ely',
                       'r_arm_wrx',
                       'r_arm_wry',
                       'r_arm_wry2',
                       'neck_ry']
    Nothers = len(otherjointnames)

    # Set up the publisher, for the left leg and other joints!
    pub = JointStatePublisher(kinleftleg.jointnames()
                              + otherjointnames)

    # Set up the subscriber and wait for the first point.
    point = PointSubscriber()
    rospy.loginfo("Waiting for a point...")
    while not rospy.is_shutdown() and not point.valid():
        pass
    rospy.loginfo("Got a point.")

    # Set the numpy printing options (as not to be overly confusing).
    # This is entirely to make it look pretty (in my opinion).
    np.set_printoptions(suppress = True, precision = 6)


    #
    #  PICK INITIAL JOINT AND TIP VALUES
    #
    # Pick the joint values.
    thetaleft0 = np.array([[0.0], [-0.6], [1.2], [-0.6], [0.0], [0.0]])
    thetaleft  = thetaleft0
    
    # Grab the tip values.
    (pleft, Rleft) = kinleftleg.fkin(thetaleft)

    # Grab the point.
    pd = point.position()


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
        # Using the result theta(i-1) of the last cycle (i-1): Compute
        # the position, orientation, and Jacobian.
        (pleft, Rleft) = kinleftleg.fkin(thetaleft)
        Jleft          = kinleftleg.Jac(thetaleft)

        # Do more stuff...

        # Advance to the new time step.
        t += dt

        # Grab a new point.
        pd = point.position()

        # And Update the theta values...
        thetaleft = thetaleft0 * np.cos(t)

        # Publish and sleep for the rest of the time.  You can choose
        # whether to show the initial "negative time convergence"....
        # if not t<0:
        pub.send(thetaleft[:,0].tolist() + [0.0]*Nothers)
        servo.sleep()

        # Break the loop when we are finished.
        # if (t >= tf):
        #     break
