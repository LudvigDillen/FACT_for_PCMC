import sys
import subprocess
import os
import hydra


@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls_registration")
def geotransformer_with_fact(args):
    # Set the environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Define the path to the test script and the argument
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../', 
        'GeoTransformer_202407', 
        'experiments', 
        'geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn', 
        'test.py'
    ))

    # Define the relative path to the snapshot file
    snapshot_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../',
        'GeoTransformer_202407',
        'weights', 
        'geotransformer-kitti.pth.tar'
    ))

    # Call the test script with the argument
    subprocess.run(['python', script_path, f'--snapshot={snapshot_path}', f'--fact_args={args}'])


# import torch
# import nuscenes as ns
# import numpy as np
# import copy



# # Add the path to the FACT repository
# fact_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FACT'))
# sys.path.append(fact_path)
# from utils.parameters import Params

# def get_input_data_fact(src_points, ref_points, gt_trans, est_trans):
#     params = Params(nusc=None, args=args, pointwise=True)

#     return {
#         'src': src_points,
#         'ref': ref_points,
#     }