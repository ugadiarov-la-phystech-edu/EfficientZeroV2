import gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ocr.cswm.modules import EncoderCNN, EncoderMLP
from ocr.tools import Tensor
from ocr.transformer.transformer_module import Transformer_Module


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, slate, transformer_kwargs):
        transformer = Transformer_Module(**transformer_kwargs)
        super().__init__(observation_space, features_dim=transformer.config.d_model)
        self.transformer = transformer

        # Have to do this as SLATE has bool parameters and thus brakes polyak updates in sb3
        self.slate = [slate]

    def forward(self, observations: Tensor) -> Tensor:
        return self.transformer(self.slate[0](observations))


class SLATEMean(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, slate):
        super().__init__(observation_space, features_dim=slate._config.slotattr.slot_size)
        self.slate = [slate]

    def forward(self, observations: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.slate[0](observations).mean(dim=1)


class CSWMSlotExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, num_slots, slot_size, cnn_hidden_dim=32, mlp_hidden_dim=128):
        super().__init__(observation_space, features_dim=num_slots * slot_size)
        self._num_slots = num_slots
        self._slot_size = slot_size
        self._cnn_hidden_dim = cnn_hidden_dim
        self._mlp_hidden_dim = mlp_hidden_dim

        num_channels = observation_space.shape[0]
        self._obj_extractor = EncoderCNN(
            input_dim=num_channels,
            hidden_dim=self._cnn_hidden_dim,
            num_objects=self._num_slots
        )
        sample = th.as_tensor(observation_space.sample(), dtype=th.float32).unsqueeze(0)
        feature_maps = self._obj_extractor(sample)

        self._obj_encoder = EncoderMLP(
            input_dim=feature_maps.flatten(start_dim=-2).size(-1),
            hidden_dim=self._mlp_hidden_dim,
            output_dim=self._slot_size,
        )

    def forward(self, observations: th.Tensor, *args, **kwargs) -> th.Tensor:
        object_feature_maps = self._obj_extractor(observations)
        return self._obj_encoder(object_feature_maps)


if __name__ == '__main__':
    import torch

    observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 100, 100), dtype=np.uint8)
    slot_extractor = CSWMSlotExtractor(observation_space, num_slots=5, slot_size=128, hidden_dim=512)
    obs = torch.ones((32, *observation_space.shape), dtype=torch.float32)
    slots = slot_extractor(obs)
    print()
