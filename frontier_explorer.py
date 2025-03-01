import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

        # Subscribers and Publishers
        self.map_subscription = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_publisher = self.create_publisher(
            PoseStamped, '/goal_pose', 10)

        # State variables
        self.map_data = None
        self.visited_frontiers = set()
        self.safety_radius = 2  # 5x5 검증 반경 (픽셀)
        self.obstacle_threshold = 100  # 장애물 값
        self.goal_pose = None

        self.get_logger().info("Frontier Explorer initialized!")

    def map_callback(self, msg):
        """맵 데이터 수신 콜백 함수"""
        self.map_data = msg
        self.get_logger().info("Map data received.")
        self.detect_and_publish_frontier()

    def detect_and_publish_frontier(self):
        """프론티어 탐지 및 목표 발행"""
        if self.map_data is None:
            return

        # 프론티어 탐지
        frontiers = self.detect_frontiers()

        if not frontiers:
            self.get_logger().info("No frontiers found!")
            return

        # 프론티어 우선순위 계산
        frontiers = sorted(frontiers, key=self.evaluate_frontier)

        for frontier in frontiers:
            if frontier not in self.visited_frontiers and self.is_safe_frontier(frontier):
                self.publish_goal(frontier)
                self.visited_frontiers.add(frontier)
                return

        self.get_logger().info("No suitable frontiers found.")

    def detect_frontiers(self):
        """맵 데이터에서 프론티어를 탐지"""
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        map_array = np.array(self.map_data.data).reshape((height, width))
        frontiers = []

        for y in range(height):
            for x in range(width):
                if map_array[y, x] == 0 and self.is_frontier_cell(x, y, map_array):
                    world_x = origin_x + (x * resolution)
                    world_y = origin_y + (y * resolution)
                    frontiers.append((world_x, world_y))

        self.get_logger().info(f"{len(frontiers)} frontiers detected.")
        return frontiers

    def is_frontier_cell(self, x, y, map_array):
        """프론티어 셀 여부 확인 (미탐지 영역과 인접한 자유 공간)"""
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < map_array.shape[1] and 0 <= ny < map_array.shape[0]:
                if map_array[ny, nx] == -1:  # 미탐지 영역
                    return True
        return False

    def is_safe_frontier(self, frontier):
        """프론티어가 안전한지 확인"""
        resolution = self.map_data.info.resolution
        x_index = int((frontier[0] - self.map_data.info.origin.position.x) / resolution)
        y_index = int((frontier[1] - self.map_data.info.origin.position.y) / resolution)

        width = self.map_data.info.width
        height = self.map_data.info.height
        map_array = np.array(self.map_data.data).reshape((height, width))

        # 5x5 영역 확인
        values = []
        for dx in range(-self.safety_radius, self.safety_radius + 1):
            for dy in range(-self.safety_radius, self.safety_radius + 1):
                nx, ny = x_index + dx, y_index + dy
                if 0 <= nx < width and 0 <= ny < height:
                    values.append(map_array[ny, nx])

        # 장애물이 포함되면 False
        if any(v == self.obstacle_threshold for v in values):
            return False

        # 평균값이 안전 기준 (0과 -1 사이)일 경우만 True
        avg_value = np.mean(values)
        return -1.0 <= avg_value <= 0.0

    def evaluate_frontier(self, frontier):
        """프론티어 평가 함수: 중앙성 및 장애물 거리 우선"""
        # 맵 중심 계산
        map_center_x = self.map_data.info.origin.position.x + (self.map_data.info.width * self.map_data.info.resolution) / 2
        map_center_y = self.map_data.info.origin.position.y + (self.map_data.info.height * self.map_data.info.resolution) / 2

        # 중앙에서의 거리 계산
        center_distance = ((frontier[0] - map_center_x)**2 + (frontier[1] - map_center_y)**2)**0.5

        # 장애물에서 안전 거리 확보
        safety_score = self.is_safe_frontier(frontier)

        # 가중치 적용: 중앙 거리보다 안전성을 우선
        return center_distance - (2.0 * safety_score)

    def publish_goal(self, frontier):
        """프론티어를 목표로 발행"""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = frontier[0]
        goal.pose.position.y = frontier[1]
        goal.pose.orientation.w = 1.0
        self.goal_publisher.publish(goal)
        self.get_logger().info(f"Published goal: {frontier}")


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
