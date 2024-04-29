#standard imports
import cv2
import numpy as np
from typing import Tuple

#project imports
from core.sensors.sensor import SensorService
from quanser_pkgs.pal.utilities.vision import Camera3D
from quanser_pkgs.pal.products.qcar import QCarRealSense, QCarCameras
from quanser_pkgs.hal.utilities.image_processing import ImageProcessing

class VirtualRGBDCamera(SensorService):
    def __init__(
        self, 
        mode: str = "RGB, Depth", 
        image_shape: Tuple[float, float] = (160, 120),
        rate: int = 20,
        device_id: str = "0@tcpip://localhost:18965",
        apply_lighting_noise: bool = False
    ):
        self.image_shape = image_shape
        self.rate = rate
        self.apply_lighting_noise = apply_lighting_noise

        self.camera = QCarRealSense(
            mode='RGB&DEPTH',
            frameWidthRGB=image_shape[0],
            frameHeightRGB=image_shape[1],
            frameRateRGB=rate,
            frameWidthDepth=image_shape[0],
            frameHeightDepth=image_shape[1],
            frameRateDepth=rate
        )

    def terminate(self) -> None: 
        self.camera.terminate()

    def get_image(self) -> np.ndarray:
        rgb = self.camera.read_RGB()
        depth = self.camera.read_depth(dataMode="PX")

        if rgb == -1 or depth == -1:
            print("Invalid RGB or depth image!")
            return np.zeros((self.image_shape[1], self.image_shape[0], 4), dtype=np.uint8)

        rgb_image = self.camera.imageBufferRGB
        depth_image = self.camera.imageBufferDepthPX

        if self.apply_lighting_noise:
            # randomly change lighting intensity
            alpha = np.random.uniform(0.5, 1.0)
            beta = np.random.uniform(-50, 0)
            rgb_image = cv2.convertScaleAbs(rgb_image, alpha=alpha, beta=beta)

            tint_qty = np.random.randint(128, 192)
            tint_image  = np.full(rgb_image.shape, (0, int(255 - tint_qty), tint_qty), np.uint8)

            # Merge the adjusted color channels
            rgb_image  = cv2.addWeighted(rgb_image, 0.9, tint_image, 0.1, 0)

        #Threshold the depth image based on min and max distance set above, and cast it to uint8
        binary_depth = ImageProcessing.binary_thresholding(depth_image, 0.2, 0.5).astype(np.uint8)[..., None]
        rgbd_image = np.concatenate([rgb_image, binary_depth], axis=-1)

        return rgbd_image