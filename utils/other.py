import debugpy


def start_debug():
    debugpy.listen(5678)
    print("Wait for debugger!")
    debugpy.wait_for_client()
    print("Attached!")


def display_to_logger_before(i, args, logger):
    logger.info(f"STARTING RUN {i + 1}. Settings:")
    logger.info(f"args.classifier {args.classifier}")
    logger.info(f"args.feature_folder {args.feature_folder}")
    logger.info(f"scenes {args.n_scenes}")
    logger.info(f"samples per scene {args.n_samples_per_scene}")
    logger.info(f"args.re_use_data {args.re_use_data}")
    logger.info(f"args.preprocessing.T_close: {args.preprocessing.T_close}")
    logger.info(f"args.features_to_create.use_c: {args.features_to_create.use_c}")
    logger.info(f"args.features_to_use.use_c: {args.features_to_use.use_c}")
    logger.info(f"args.fps.num_point: {args.fps.num_point}")
    logger.info(f"args.batch_size: {args.batch_size}")
    logger.info(f"args.epoch: {args.epoch}")
    logger.info(f"args.lr_gamma: {args.lr_gamma}")
    logger.info(f"args.model_identifier: {args.model_identifier}")


def display_to_logger_after(i, args, logger):
    logger.info(f"FINISHED RUN {i + 1}. Settings:")
    logger.info(f"args.classifier {args.classifier}")
    logger.info(f"args.feature_folder {args.feature_folder}")
    logger.info(f"scenes {args.n_scenes}")
    logger.info(f"samples per scene {args.n_samples_per_scene}")
    logger.info(f"args.re_use_data {args.re_use_data}")
    logger.info(f"args.preprocessing.T_close: {args.preprocessing.T_close}")
    logger.info(f"args.features_to_create.use_c: {args.features_to_create.use_c}")
    logger.info(f"args.features_to_use.use_c: {args.features_to_use.use_c}")
    logger.info(f"args.fps.num_point: {args.fps.num_point}")
    logger.info(f"args.batch_size: {args.batch_size}")
    logger.info(f"args.epoch: {args.epoch}")
    logger.info(f"args.lr_gamma: {args.lr_gamma}")
    logger.info(f"args.model_identifier: {args.model_identifier}")