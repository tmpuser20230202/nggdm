# nggdm
Repository for Novel Graph Generation with GMM-based Diffusion Models

## Setup
To build the environment, run the following script:
```bash
$ conda create -f=nggdm.yaml
```

The following libraries are installed. 
- Python 3.10.3
- NumPy 1.26.2
- SciPy 1.11.4
- PyTorch 1.12.0
- PyTorch Geometric 2.4.0
- PyTorch Scatter 2.1.0
- PyTorch Sparse 0.6.16
- Scikit-Learn 1.3.2
- NetworkX 3.2.1

To pull the repositories for diffusion model with Gaussian mixture model from github, run the following script:
```bash
$ mkdir repos
$ cd repos
$ git clone https://github.com/networkslab/gmcd.git
$ git clone https://github.com/harryjo97/DruM.git
```

For calculating the parametric complexity, run the following script in the directory `util`:
```bash
$ cd ../util
$ conda install -c conda-forge cxx-compiler
$ cythonize -3 -a -i cython_normterm_discrete.pyx
```

## Synthetic Dataset
Copy the config files from the repositories downloaded from github:
```bash
$ cd ../synthetic
$ mkdir config
$ cp ../repos/DruM/DruM_2D/config/planar.yaml config
$ cp ../repos/DruM/DruM_2D/config/sbm.yaml config
```

```bash
$ mkdir data
```

Download the pickle files from Google drive and move them to `data` directory
- [Planar dataset(`planar_64_200.pt`): ](https://drive.google.com/drive/folders/13esonTpioCzUAYBmPyeLSjXlDoemXXQB?usp=sharing)
- [SBM dataset(`sbm_200pt`): ](https://drive.google.com/drive/folders/1imzwi4a0cpVvE_Vyiwl7JCtkr13hv9Da?usp=sharing)

Then, in the directory `synthetic`, Run the following scripts: 
```bash
$ python nggdm_planar.py
$ python nggdm_sbm.py
```

## Real Dataset
In the directory `real`, Run the following scripts:
```bash
$ python nggdm_cora.py
$ python nggdm_pubmed.py
$ python nggdm_citeseer.py
```
