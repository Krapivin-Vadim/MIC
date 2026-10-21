from dto.DroneLocationDto import DroneLocationDto
from interface.MavrosManager import MavrosManager
import rospy
from sensor_msgs.msg import NavSatFix
from geographic_msgs.msg import GeoPoseStamped


class MavrosManagerImpl(MavrosManager):
    def __init__(self,
                 sub_topic: str,
                 pub_topic: str,
                 timeout: float = 5.0):
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.timeout = timeout
        self.publisher = rospy.Publisher(
            self.pub_topic,
            GeoPoseStamped,
            queue_size=10
        )

    def get_drone_location(self) -> DroneLocationDto:
        try:
            msg: NavSatFix = rospy.wait_for_message(
                self.sub_topic,
                NavSatFix,
                timeout=self.timeout
            )

            return DroneLocationDto(msg.latitude, msg.longitude, msg.altitude)
        except rospy.ROSException as e:
            rospy.logerr(f"Timeout: Не удалось получить координаты от PixHawk за {self.timeout} сек. Ошибка: {e}")
            raise TimeoutError("PixHawk не отвечает на запрос координат.")


    def set_drone_location(self, drone_location: DroneLocationDto) -> None:
        msg = GeoPoseStamped()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        msg.pose.position.latitude = drone_location.latitude
        msg.pose.position.longitude = drone_location.longitude
        msg.pose.position.altitude = drone_location.altitude

        msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)

        rospy.loginfo(f"POST: Целевые координаты отправлены на PixHawk: {drone_location}")