# FACT

## FACT: Multinomial Misalignment Classification for Point Cloud Registration

## Description
This project is the code-base for FACT. Given two point clouds, and an estimated rigid transformation between the point cloud, FACT predicts the registration quality of the alignment.

The abstract of the paper: "We present FACT, a method for predicting alignment quality (i.e., registration error) of registered lidar point cloud pairs. This is useful e.g. for quality assurance in large, automatically registered 3D models. FACT extracts local features from a registered pair and processes them with a point transformer-based network to predict a misalignment class. We generalize prior work that study binary alignment classification of registration errors, by recasting it as multinomial misalignment classifi- cation. To achieve this, we introduce a custom regression-by-classification loss function that combines the cross-entropy and Wasserstein losses, and demonstrate that it outperforms both direct regression and prior binary classification. FACT successfully classifies point-cloud pairs registered with both the classical ICP and GeoTransformer, while other choices, such as standard point-cloud-quality metrics and registration residuals are shown to be poor choices for predicting misalignment. On a synthetically perturbed point-cloud task introduced by the CorAl method, we show that FACT achieves substantially better performance than CorAl. Finally, we demonstrate how FACT can assist experts in correcting misaligned point-cloud maps. Our code will be made publicly available."

Good sources for this project is the paper with the name: "FACT: Multinomial Misalignment Classification for Point Cloud Registration" (not publically available yet) and this [master thesis](https://liu.diva-portal.org/smash/get/diva2:1803604/FULLTEXT01.pdf).

Point cloud pairs that we try to correct can look something like this. GT class 0 means that the actual error is zero. GT class 9 is the highest error class for this experiment.
![alt text](docs/point_cloud_pair.png)

The pipeline of FACT looks like this:
![alt text](docs/flowchart_v11.jpg)


## Installation
This project uses [Poetry](https://python-poetry.org/) for dependency management. Follow these steps to set up the environment:

### Code and Dependency Setup
1. **Clone the repository:**

   ```bash
   git clone https://github.com/LudvigDillen/FACT.git
   cd FACT
   ```

2. **Install poetry**
Follow the instructions on Poetry's official installation guide.

3. **Install the project dependencies**

    ```bash
    poetry install
    ```

### Data Setup
To run experiments, you need to download nuScenes and potentially KITTI.

1. **Download nuScenes lidar data**
    1. Go to [nuscenes](https://www.nuscenes.org/nuscenes)
    2. Create an account
    3. Login
    4. Go to nuScenes/Downloads/Full dataset (v1.0) and download Mini and Trainval. For Trainval, it suffices to download the Metadata + Lidar blobs for all parts (i.e. part 1 - part 10) (so 850 scenes in total).
The full download should be 221 GB (I think)

2. **Download KITTI**
    1. Go to [kitti](https://www.cvlibs.net/datasets/kitti/)
    2. Create an account
    3. Login (once the account is accepted, can take some time)
    4. Follow [Geotransformer](https://github.com/qinzheng93/GeoTransformer?tab=readme-ov-file) for how to download and organize the KITTI data.
The full download (after following steps in Geo)


## Usage
To run the code there are a few things to know. First off, the all configuration files can be found in `classifiers/PointTransformers/config/`, for different experiments, different configuration files are suitable to use. We'll get back to which one to use for which experiment, and also a bit of what it contains.

Where are going to go through how to reproduce the result of the paper and by doing so, also go through how to run the code.

Exp 1:

**Table 1.** Classification accuracy for CorAl and FACT on two classification tasks.

|             **Task**             | **[CorAl](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9568846)** | **Mapped FACT** |
|:--------------------------------:|:------------------------------:|:---------------:|
| $(\theta, e_d)=(0.01, 0.1)$       | 75.3%                         | **97.4%**      |
| $(\theta, e_d)=(0.03, 0.3)$       | 95.6%                         | **100.0%**     |

Exp 2:
![alt text](docs/confusion_rbc_vs_regression.png)

Exp 3:

**Table 2.** Comparison between regression-by-classification (RbC) and regression.  
*Note: $\xi_k$ represents the fraction of samples where the predicted label is at least $k$ classes away from the true label.*

|                 | $\xi_1$      | $\xi_2$      | $\xi_3$      | $\xi_4$      |
|-----------------|--------------|--------------|--------------|--------------|
| **RbC**       | **18.24%**   | **0.88%**    | **0.05%**   | **0.00%**   |
| **Regression**| 23.57%       | 1.07%       | **0.05%**   | **0.00%**   |


Exp 4:
For scene 708, FACT gets this result. The idea is to get map (b) to look like map (c) which it largely does here. The color denotes the $z$ coordinate. Figure (d) show the ground
![alt text](docs/scene708.png)

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
