import torch
from tqdm.auto import tqdm
import numpy as np


from visualization.point_e.point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from visualization.point_e.point_e.diffusion.sampler import PointCloudSampler
from visualization.point_e.point_e.models.download import load_checkpoint
from visualization.point_e.point_e.models.configs import MODEL_CONFIGS, model_from_config


def get_point_e_model(text="an orange car"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''),  # Do not condition the upsampler at all
    )

    # Set a prompt to condition on.
    prompt = text

    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x

    return samples, sampler


# My own visualizations
def scatter_point(x, y, z, ax, col="r", size=0.50):
    # Add a 3D point (x, y, z)
    ax.scatter([x], [y], [z], c=col, marker='o', s=20)
    # Add dotted lines from the point to the respective axes
    ax.plot([x, x], [y, y], [z, -size], col + '--', linewidth=2, alpha=1.0)  # to x-axis
    ax.plot([x, x], [y, size], [z, z], col + '--', linewidth=2, alpha=1.0)  # to y-axis
    ax.plot([x, -size], [y, y], [z, z], col + '--', linewidth=2, alpha=1.0)  # to z-axis
    return ax


def scatter_spheres(x, y, z, ax, col="r", size=0.50):
    ax.scatter([x], [y], [z], c=col, marker='o', s=20)  # Increase the size of the point

    # Add "dotted" lines from the point to the respective axes by creating a series of small spheres along
    # the line
    t = np.linspace(-size, size, num=50)  # Adjust num to add more points along the line
    ax.scatter(np.full_like(t, x), np.full_like(t, y), t, c=col, marker='o', s=3, alpha=1.0)  # to x-axis
    ax.scatter(t, np.full_like(t, y), np.full_like(t, z), c=col, marker='o', s=3, alpha=1.0)  # to y-axis
    ax.scatter(np.full_like(t, x), t, np.full_like(t, z), c=col, marker='o', s=3, alpha=1.0)  # to z-axis
    return ax
