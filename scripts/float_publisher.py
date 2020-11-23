#!/usr/bin/env python3
#
#   float_publisher.py
#
#   Publish an floating point number, as determined by a slider
#
#   This generates a point message - no visualization message.
#
#   Publish:    /float          std_msgs/Float64
#
import sys
import signal
import threading
import rospy
 
from PyQt5.QtCore    import (Qt, QTimer)
from PyQt5.QtWidgets import (QWidget, QSlider, QHBoxLayout, QVBoxLayout,
                             QLabel, QApplication)

from std_msgs.msg import Float64


#
#  Float Publisher
#
#  This continually publishes the float.
#
class FloatPublisher:
    def __init__(self, v):
        # Prepare the publisher (latching for new subscribers).
        self.pub = rospy.Publisher("/float", Float64,
                                   queue_size=1, latch=True)

        # Create the float message.
        self.float = Float64()
        self.float.data = v

    def update(self, v):
        # Update the value.
        self.float.data = v

    def publish(self):
        # Publish.
        self.pub.publish(self.float)

    def loop(self):
        # Prepare a servo loop at 10Hz.
        servo = rospy.Rate(10)
        rospy.loginfo("Float-Publisher publication thread running at 10Hz...")

        # Loop: Publish and sleep
        while not rospy.is_shutdown():
            self.publish()
            servo.sleep()

        # Report the cleanup.
        rospy.loginfo("Float-Publisher publication thread ending...")


#
#  GUI Slider Class
#
class VarGUI(QWidget):
    def __init__(self, val, callback):
        super().__init__()
        name   = 'Data'
        minval = -1.0
        maxval =  1.0
        self.value    = val
        self.offset   = (maxval + minval) / 2.0
        self.slope    = (maxval - minval) / 200.0
        self.callback = callback
        self.initUI(name)

    def initUI(self, name):
        # Top-Left: Name
        label = QLabel(name)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setMinimumWidth(40)

        # Top-Right: Number
        self.number = QLabel("%6.3f" % self.value)
        self.number.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.number.setMinimumWidth(100)
        self.number.setStyleSheet("border: 1px solid black;")

        # Bottom: Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-100, 100)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setPageStep(5)
        slider.valueChanged.connect(self.valueHandler)
        slider.setValue(int((self.value - self.offset)/self.slope))

        # Create the Layout
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addSpacing(10)
        hbox.addWidget(self.number)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(slider)

        self.setLayout(vbox)
        self.setWindowTitle('Data')
        self.show()

    def valueHandler(self, value):
        self.value = self.offset + self.slope * float(value)
        self.number.setText("%6.3f" % self.value)
        self.callback(self.value)

    def kill(self, signum, frame):
        self.close()


#
#  Main Code
#
if __name__ == "__main__":
    # Prepare the node.
    rospy.init_node('float_publish')

    # Pick an initial value.
    v0 = 0.0

    # Prepare the float publisher.
    fpub = FloatPublisher(v0)

    # Prepare Qt.
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Float Publisher")

    # Include a Qt Timer, setup up so every 500ms the python
    # interpreter runs (doing nothing).  This enables the ctrl-c
    # handler to be processed and close the window.
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    # Then setup the GUI window.  And declare the ctrl-c handler to
    # close that window.
    gui = VarGUI(v0, fpub.update)
    signal.signal(signal.SIGINT, gui.kill)


    # Start the publisher in a separate thread.
    fpubthread = threading.Thread(target=fpub.loop)
    fpubthread.start()

    # Start the GUI window.
    rospy.loginfo("Float-Publisher GUI starting...")
    status = app.exec_()
    rospy.loginfo("Float-Publisher GUI ended.")

    # Having re-mapped the sigint handler, we have to shutdown ROS
    # manually (and thus the publisher thread).
    rospy.signal_shutdown("")
    fpubthread.join()
    
    # And exit.
    sys.exit(status)
