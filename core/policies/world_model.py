import cv2
import torch
import numpy as np

#project imports
from core.models.torch.world_model import WorldModel
from core.data.preprocessor import Preprocessor
from core.policies.network import NetworkPolicy
from core.utils.aggregation_utils import map_structure

#testing
from core.policies.pure_pursuit import PurePursuitPolicy

class WorldModelPolicy(NetworkPolicy):
    def __init__(self, device):
        self.device = device
        self.model = WorldModel(device=device)
        self.policy = PurePursuitPolicy(max_lookahead_distance=0.75)

        self.preprocessor = Preprocessor(
            image_key="image"
        )

    def reset_state(self):
        self.recurrent_state = self.model.init_state()

    def __call__(self, obs):
        batch = self.preprocessor.apply(obs, expand_tb=True)
        obs_model: Dict[str, torch.Tensor] = map_structure(map_structure(batch, torch.from_numpy), lambda x: x.to(self.device))
        pred, recurrent_state, metrics = self.model.forward(obs_model, self.recurrent_state)
        self.recurrent_state = recurrent_state

        image_raw = pred["image"].squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image_raw + 0.5) * 255
        image = image.astype(np.uint8)

        cv2.imshow("Reconstruction", cv2.resize(image, (640, 480)))
        cv2.waitKey(1)

        #steer = np.random.normal(loc=0.0, scale=0.3)
        #action = np.array([0.04, steer])
        action, _ = self.policy(obs)

        return action, metrics