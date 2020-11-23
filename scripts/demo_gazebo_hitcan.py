#!/usr/bin/env python3
#
#   demo_gazebo_hitcan.py
#
#   Demonstrate hitting the can.  Note this uses a direct position
#   command, so the model does not have velocity and the impact is not
#   particularly strong.  This assume the sevenposition.urdf!
#
#   Create a motion by continually sending joint values.  Also listen
#   to the point input.
#
#   Subscribe: /point                               geometry_msgs/PointStamped
#   Publish:   /joint_states                        sensor_msgs/JointState
#   Publish:   /sevenbot/joint1_position/command    std_msgs/Float64
#   Publish:   /sevenbot/joint2_position/command    std_msgs/Float64
#   Publish:   /sevenbot/joint3_position/command    std_msgs/Float64
#   Publish:   /sevenbot/joint4_position/command    std_msgs/Float64
#   Publish:   /sevenbot/joint5_position/command    std_msgs/Float64
#   Publish:   /sevenbot/joint6_position/command    std_msgs/Float64
#   Publish:   /sevenbot/joint7_position/command    std_msgs/Float64
#
import rospy
import numpy as np

from gazebodemos.kinematics2 import Kinematics
from std_msgs.msg            import Float64
from sensor_msgs.msg         import JointState
from geometry_msgs.msg       import PointStamped


#
#  Point Subscriber
#
#  Save any messages that come over the /point topic, containing the
#  XYZ slider data.
#
class PointSubscriber:
    def __init__(self):
        # Instantiate the point p as a numpy vector.
        self.p = np.array([[0.0], [0.0], [0.0]])

        # Create a subscriber to listen to point messages.
        rospy.Subscriber("point", PointStamped, self.process)

    def position(self):
        # Return the point.
        return self.p

    def process(self, msg):
        # Save the point contained in the message.
        self.p[0] = msg.point.x
        self.p[1] = msg.point.y
        self.p[2] = msg.point.z


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
        self.pubjs = rospy.Publisher("/joint_states",
                                     JointState, queue_size=100)
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
    eR = 0.5 * (cross(R[:,[0]], Rd[:,[0]]) +
                cross(R[:,[1]], Rd[:,[1]]) +
                cross(R[:,[2]], Rd[:,[2]]))
    return np.vstack((ep,eR))


#
#  Main Code
#
if __name__ == "__main__":
    #
    #  LOGISTICAL SETUP
    #
    # Prepare the node.
    rospy.init_node('demo_gazebo_hitcan')
    rospy.loginfo("Starting the demo code to hit the can...")

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
                                ('joint1_position',
                                 'joint2_position',
                                 'joint3_position',
                                 'joint4_position',
                                 'joint5_position',
                                 'joint6_position',
                                 'joint7_position'))

    # Make sure the URDF and publisher agree in dimensions.
    if not pub.dofs() == kin.dofs():
        rospy.logerr("FIX Publisher to agree with URDF!")

    # Set up the subscriber.
    point = PointSubscriber()

    # Set the numpy printing options (as not to be overly confusing).
    # This is entirely to make it look pretty (in my opinion).
    np.set_printoptions(suppress = True, precision = 6)


    #
    #  PREPARE THE MOTION
    #
    theta = np.zeros((7,1))


    #
    #  TIME LOOP
    #
    t0 = 0.0
    tf = 2.0
    t  = t0
    while not rospy.is_shutdown():
        # Advance to the new time step.
        t += dt

        # Compute the command.
        theta[3] = np.cos(np.pi*t) - 1.0
        
        # Publish and sleep for the rest of the time.
        pub.send(theta)
        servo.sleep()

        # Break the loop when we are finished.
        if (t >= tf):
            break
