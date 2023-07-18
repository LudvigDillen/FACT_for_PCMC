import torch


class Params:
    def __init__(self, nusc, n_scenes, n_samples_per_scene, train_ratio,
                 downsample_factor, T_close_thresh, params_diff_entropy,
                 verbose=False, hpr_radius=3.25, preprocess=True, pointwise=True,
                 do_fps=True, N_fps_points=9192,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        nusc: The NuScenes class variable.
        n_scenes (int): The total number of scenes.
        n_samples_per_scene (int): The number of samples to take from each scene.
        train_ratio (float): The ratio of scenes to use for training.
        downsample_factor (int): The factor by which to downsample the scenes.
        T_close_thresh (float): Threshold for considering a sample as "close".
        params_diff_entropy (dict): Parameters for the differential entropy calculation.
        verbose (bool): If True, print additional information.
        hpr_radius (float): Proportional to the radius of the HPR operator.
        preprocess (bool): whether to preprocess the point cloud or not
        pointwise (bool): Whether to process pointcloud pointwise or not. Must be true
                          if we want to feed it to the DNN.
        do_fps (bool) : Whether or not to apply farthest point sampling
        fps_N_points (int): How many points we should evaluate each metric for, note
                            that the hole point cloud is considered when stuying each
                            sampled points' neighborhoods.
        device (torch.device)      : Which device to perform calculations on (cuda or cpu).
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

        # Set preprocess settings and parameters
        self.preprocess = preprocess
        self.pointwise = pointwise
        self.do_fps = do_fps
        self.N_fps_points = N_fps_points

        # Set co-visibility parameters
        self.hpr_radius = hpr_radius

        # Set device
        self.device = device

    def set_params_diff_entropy(self, params_diff_entropy):
        self.params_diff_entropy = params_diff_entropy

    def set_which_features_to_use(self, features):
        self.use_label = features.use_label  # class label
        self.use_de = features.use_de  # differential entropy
        self.use_wd = features.use_wd  # wasserstein distance
        self.use_c = features.use_c  # covisibility weight
        self.use_s = features.use_s  # static weight
        self.use_cj = features.use_cj  # joint cardinality ratio
        self.use_cs = features.use_cs  # separate cardinality ratio
        # All these feature belows are calculated in neighborhoods (spheres) around points
        self.study_neighborhoods = (self.use_de or self.use_wd or self.use_cj or self.use_cs)
        self.calc_joint_neighbors = (self.use_de or self.use_cj)
        self.calc_sep_neighbors = (self.use_de or self.use_cs)
