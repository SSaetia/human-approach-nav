import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class HumanFollowingNode(Node):
    def __init__(self):
        super().__init__('human_following_node')
        self.human_pose_sub = self.create_subscription(
            PoseStamped,
            'human_pose_topic',
            self.human_pose_callback,
            10
        )
        self.human_pose_sub  # prevent unused variable warning

    def human_pose_callback(self, msg: PoseStamped):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.yaw = msg.pose.orientation.z
        print(f"Human position: x={self.x}, y={self.y}, yaw={self.yaw}")
        return(self.x, self.y, self.yaw)
    
    def cal_goal_pose(self):
        
        return()

def main(args=None):
    rclpy.init(args=args)
    node = HumanFollowingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
