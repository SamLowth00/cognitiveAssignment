import rospy
import os
import miro2 as miro
import time
from gi.repository import GLib
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String
		
front_left, mid_left, rear_left, front_right, mid_right, rear_right = range(6)
illum = UInt32MultiArray()
illum.data = [0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0]
topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
pub_illum = rospy.Publisher(topic_base_name + "/control/illum", UInt32MultiArray, queue_size=0)
rospy.init_node("client_illum")

value = 0xFFFFFFFF
for x in range(6):
	illum.data[x] = value



def update_colours():
	pub_illum.publish(illum)
	return True

print("penis")
while True:
	update_colours()
	time.sleep(1)
