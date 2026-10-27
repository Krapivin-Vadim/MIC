from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Camera
    cam_id: int
    cam_timeout: int
    cam_frame_size: int
    cam_fps: int

    # Frames / GeoTIFF
    frame_base_dir: Path
    frame_source_crs: str
    frame_target_crs: str
    frame_fov_v: float
    frame_fov_h: float
    frame_crop_margin: float

    # AI
    ai_max_keypoints: int
    ai_device: int

    # ROS
    ros_source_topic: str
    ros_target_topic: str
    ros_timeout: int


settings = Settings()