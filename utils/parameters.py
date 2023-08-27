import torch

from utils.data_handling import count_decimal_digits


class Params:
    def __init__(self, nusc, args, pointwise=True, do_fps=True):
        """
        nusc: The NuScenes class variable.

        train_ratio (float): The ratio of scenes to use for training.
        downsample_factor 
        T_close_thresh (float): Threshold for considering a sample as "close".
        hpr_radius (float): Proportional to the radius of the HPR operator.
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

        # Set display parameters
        self.verbose = args.verbose

        # Set preprocess settings and parameters
        self.pointwise = pointwise
        self.do_fps = do_fps
        assert args.num_point % 2 == 0, "args.num_point must be even"
        self.N_fps_points = int(args.num_point/2)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size_feature_extraction = args.batch_size_feature_extraction

        self.set_class_names()

        # Neighborhood sin alpha
        alpha = torch.tensor(
            args.sensor_settings.vertical_angular_res*args.neighborhood.alpha_scale)
        self.sin_alpha = torch.sin(alpha)

        # Init feature usage
        self.use_label = False  # class label
        self.use_jde = False  # joint differential entropy
        self.use_sde = False  # separate differential entropy
        self.use_sd = False  # sinkhorn divergence
        self.use_c = False  # covisibility weight
        self.use_s = False  # static weight
        self.use_cj = False  # joint cardinality ratio
        self.use_cs = False  # separate cardinality ratio
        self.use_csj = False  # cardinality ratio sep and joint neighborhood
        # All these feature belows are calculated in neighborhoods (spheres) around points
        self.study_neighborhoods = False
        self.calc_joint_neighbors = False
        self.calc_sep_neighbors = False

    def set_which_features_to_use(self, features):
        self.use_label = features.use_label  # class label
        self.use_jde = features.use_jde  # joint differential entropy
        self.use_sde = features.use_sde  # separate differential entropy
        self.use_sd = features.use_sd  # sinkhorn divergence
        self.use_c = features.use_c  # covisibility weight
        self.use_s = features.use_s  # static weight
        self.use_cj = features.use_cj  # joint cardinality ratio
        self.use_cs = features.use_cs  # separate cardinality ratio
        self.use_csj = features.use_csj  # cardinality ratio sep and joint neighborhood
        # All these feature belows are calculated in neighborhoods (spheres) around points
        self.study_neighborhoods = (self.use_jde or self.use_sde or self.use_sd or
                                    self.use_cj or self.use_cs or self.use_csj)
        self.calc_joint_neighbors = (self.use_jde or self.use_cj or self.use_csj)
        self.calc_sep_neighbors = (self.use_sde or self.use_cs or self.use_csj)

    def set_class_names(self):
        r_digits = count_decimal_digits(self.args.perturb_settings.r_bin)
        t_digits = count_decimal_digits(self.args.perturb_settings.t_bin)
        self.class_names = {}
        for i in range(self.args.perturb_settings.n_classes):
            roff = round(self.args.perturb_settings.r_bin*i, r_digits)
            toff = round(self.args.perturb_settings.t_bin*i, t_digits)
            class_name = (f'class_category_{i}' + '_R_offset_' + str(roff) +
                          '_t_offset_' + str(toff))
            class_name = class_name.replace('.', '_')
            self.class_names[i] = class_name
