class Params:
    def __init__(self, nusc, n_scenes, n_samples_per_scene, train_ratio,
                 downsample_factor, T_close_thresh, params_diff_entropy,
                 verbose=False):
        """
        nusc: The NuScenes class variable.
        n_scenes (int): The total number of scenes.
        n_samples_per_scene (int): The number of samples to take from each scene.
        train_ratio (float): The ratio of scenes to use for training.
        downsample_factor (int): The factor by which to downsample the scenes.
        T_close_thresh (float): Threshold for considering a sample as "close".
        params_diff_entropy (dict): Parameters for the differential entropy calculation.
        verbose (bool): If True, print additional information.
        """
        # Set dataset parameters
        self.nusc = nusc
        self.n_scenes = n_scenes
        self.n_samples_per_scene = n_samples_per_scene
        self.train_ratio = train_ratio

        # Set point cloud variables
        self.downsample_factor = downsample_factor
        self.T_close_thresh = T_close_thresh

        # Set differential entropy parameters
        self.set_params_diff_entropy(params_diff_entropy)

        # Set display parameters
        self.verbose = verbose

    def set_params_diff_entropy(self, params_diff_entropy):
        self.params_diff_entropy = params_diff_entropy
