# WiFOP

WiFOP - Wildfire Outbreak Prediction dashboard.

### Installing the environments:

We have two environments, wifop-ml and wifop-dl, one for performing tasks using classical machine learning algorithms and the other for using deep learning algorithms.

wifop-ml environment has the following tools installed in it:
  - python=3.12
  - ipykernel
  - scikit-learn
  - matplotlib
  - seaborn
  - xgboost

wifop-dl environment has the following tools installed in it:
  - python=3.12
  - tensorflow
  - matplotlib
  - seaborn
  - ipykernel

The environments can  be installed using the following commands:
```
conda env create -f wifop-ml.yml
conda env create -f wifop-dl.yml
```

Once installed the conda environments can be activated and deactivated using the following commands:

```
conda activate wifop-ml
conda deactivate wifop-ml
```
```
conda activate wifop-dl
conda deadctivate wifop-dl
```

The following command is helpful in the event that we install more tools in the environments and need to update them without having to remove and reinstall the environment:

```
conda env update --name <environment_name> --file environment.yml --prune
```
