import random
import numpy as np
import torch
from monai.transforms.transform import RandomizableTransform, MapTransform
from src.TumorGenerated.utils import SynthesisTumor, get_predefined_texture

class TumorGenerated(RandomizableTransform, MapTransform):
    def __init__(self, keys, prob=0.1, tumor_prob=[0.2, 0.2, 0.2, 0.2, 0.2], allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        #RandomizableTransform.__init__(self, prob)
        self.tumor_types = ['tiny', 'small', 'medium', 'large', 'mix']
        assert len(tumor_prob) == 5
        self.tumor_prob = np.array(tumor_prob)
        self.textures = []  # Pre-defined textures, initialize this based on your requirements
        self.init_textures()  # Initialize self.textures here
        print("All predefined textures have been generated.")

    def randomize(self, data=None):
        pass  # No need to implement this method if randomization is not required.

    def init_textures(self):
        # Initialize textures based on your requirements
        sigma_as = [3, 6, 9, 12, 15]
        sigma_bs = [4, 7]
        predefined_texture_shape = (512, 512, 4)
        for sigma_a in sigma_as:
            for sigma_b in sigma_bs:
                texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
                self.textures.append(texture)

    def __call__(self, image, label):
        if torch.max(label) <= 1:
            self.randomize()
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            texture = random.choice(self.textures)
            label = SynthesisTumor(image, label, tumor_type, texture)
        return label
