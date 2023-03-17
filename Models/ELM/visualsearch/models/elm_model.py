import numpy as np
from scipy.stats import entropy
from ..utils import utils

class ELMModel:
    def __init__(self, grid_size, visibility_map, save_probability_maps):
        self.grid_size              = grid_size
        self.visibility_map         = visibility_map
        self.current_entropy_map    = np.empty(shape=grid_size)
        self.last_fixation          = None
        # self.norm_cdf_table = self.create_norm_cdf_table(norm_cdf_tolerance)
        # self.number_of_processes = number_of_processes
        self.save_probability_maps = save_probability_maps
    
    def expected_information_gain_map(self, expected_ig_map, posterior):
        """Computes the expected information gain for each cell in the grid"""
        # H = entropy(posterior)
        # necesito tener la fijacion actual
        # para cada posible fijacion futura tengo que calcular la ganancia de informacion de esa fijacion
        for w in range(self.grid_size[0]):
            for h in range(self.grid_size[1]):
                # la ganancia esperada de realizar una fijacion a la posicion (w,h)
                posterior_weighted  = posterior * (self.visibility_map.at_fixation((w,h))**2)
                neg_flag_posteriorw = posterior_weighted < 0
                neg_flag_visibility = self.visibility_map.at_fixation((w,h)) < 0
                if neg_flag_posteriorw.any() or neg_flag_visibility.any():
                    print("posterior_weighted flag: ", neg_flag_posteriorw.any())
                    print("visibility flag: ", neg_flag_visibility.any())
                    breakpoint()
                expected_ig_map[w][h] = 1/2 * posterior_weighted.sum()
        if expected_ig_map.min() < 0:
            breakpoint()
    
    def next_fixation(self, posterior, image_name, fixation_number, output_path):
        
        expected_ig_map = np.empty(shape=self.grid_size)
        # Compute the expected information gain map
        self.expected_information_gain_map(expected_ig_map, posterior)
        
        # Save the entropy map reduction
        if self.save_probability_maps:
            utils.save_probability_map(output_path, image_name, posterior, fixation_number, map_type='probability_maps')
        
        # Get the fixation which minimizes the expected entropy
        coordinates = np.where(expected_ig_map == np.amax(expected_ig_map))
        next_fix    = (coordinates[0][0], coordinates[1][0])
        #breakpoint
        # Update internal state for debug
        self.last_fixation = next_fix
        self.current_entropy_map = expected_ig_map
        return next_fix