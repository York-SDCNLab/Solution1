import numpy as np
from typing import Union, Tuple
from quanser_pkgs.pal.products.qcar import QCarCameras

#project imports
from core.sensors.sensor import SensorService

class VirtualCSICamera(SensorService):
    def __init__(
        self,
        image_shape: Tuple[int, int] = (820, 410),
        rate: int = 30,
        enable_front: bool = True,
        enable_right: bool = False,
        enable_left: bool = False,
        enable_back: bool = False
    ):
        self.image_shape = image_shape
        self.rate = rate
        self.enable_front = enable_front
        self.enable_right = enable_right
        self.enable_left = enable_left
        self.enable_back = enable_back

        self.camera = QCarCameras(
            frameWidth = image_shape[0],
            frameHeight = image_shape[1],
            frameRate = rate,
            enableFront=enable_front,
            enableRight=enable_right,
            enableLeft=enable_left,
            enableBack=enable_back
        )

    def terminate(self):
        for c in self.camera.csi:
            if c is not None:
                c.terminate()

    def get_images(self) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
        self.camera.readAll()

        front, right, left, back = None, None, None, None
        if self.enable_front:
            front = self.camera.csiFront.imageData

        if self.enable_right:
            right = self.camera.csiRight.imageData

        if self.enable_left:
            left = self.camera.csiLeft.imageData

        if self.enable_back:
            back = self.camera.csiBack.imageData

        return front, right, left, back