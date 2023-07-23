import torch

from utils.data_handling import count_decimal_digits


class Params:
    def __init__(self, nusc, args, T_close_thresh, params_diff_entropy,
                 verbose=False, hpr_radius=3.25, preprocess=True, pointwise=True,
                 do_fps=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        nusc: The NuScenes class variable.
        
        train_ratio (float): The ratio of scenes to use for training.
        downsample_factor 
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
        batch_size_feature_extraction (int): How many points to handle in each batch.
        """
        # Set dataset parameters
        self.nusc = nusc
        self.n_scenes = args.n_scenes  # (int): The total number of scenes.
        # (int): The number of samples to take from each scene.
        self.n_samples_per_scene = args.n_samples_per_scene
        self.train_ratio = args.train_ratio

        # Set args
        self.args = args  # TODO: Maybe, just leave everything in args ...

        # Set point cloud variables
        self.downsample_factor = args.downsample_factor  # (int): The factor to downsample the scenes with.
        self.T_close_thresh = T_close_thresh

        # Set differential entropy parameters
        self.set_params_diff_entropy(params_diff_entropy)

        # Set display parameters
        self.verbose = verbose

        # Set preprocess settings and parameters
        self.preprocess = preprocess
        self.pointwise = pointwise
        self.do_fps = do_fps
        self.N_fps_points = args.num_point

        # Set co-visibility parameters
        self.hpr_radius = hpr_radius

        # Set device
        self.device = device
        self.batch_size_feature_extraction = args.batch_size_feature_extraction

        self.perturb_settings = args.perturb_settings
        self.set_class_names()

    def set_params_diff_entropy(self, params_diff_entropy):
        self.params_diff_entropy = params_diff_entropy

    def set_which_features_to_use(self, features):
        self.use_label = features.use_label  # class label
        self.use_jde = features.use_jde  # joint differential entropy
        self.use_sde = features.use_sde  # separate differential entropy
        self.use_wd = features.use_wd  # wasserstein distance
        self.use_c = features.use_c  # covisibility weight
        self.use_s = features.use_s  # static weight
        self.use_cj = features.use_cj  # joint cardinality ratio
        self.use_cs = features.use_cs  # separate cardinality ratio
        # All these feature belows are calculated in neighborhoods (spheres) around points
        self.study_neighborhoods = (self.use_jde or self.use_sde or self.use_wd or self.use_cj or self.use_cs)
        self.calc_joint_neighbors = (self.use_jde or self.use_cj)
        self.calc_sep_neighbors = (self.use_sde or self.use_cs)

    def set_class_names(self):
        R_digits = count_decimal_digits(self.perturb_settings.r_bin)
        t_digits = count_decimal_digits(self.perturb_settings.t_bin)
        self.class_names = {}
        for i in range(self.perturb_settings.n_classes):
            Roff = round(self.args.perturb_settings.r_bin*i, R_digits)
            toff = round(self.args.perturb_settings.t_bin*i, t_digits)
            class_name = (f'class_category_{i}' + '_R_offset_' + str(Roff) +
                          '_t_offset_' + str(toff))
            class_name = class_name.replace('.', '_')
            self.class_names[i] = class_name
