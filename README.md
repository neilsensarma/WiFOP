# WiFOP

WiFOP - Wildfire Outbreak Prediction dashboard.

### Installing the environments:

We have one environment, wifop, which has the following tools installed in it for performing classification tasks using classical machine learning and deep learning models:
  - python=3.12
  - tensorflow
  - matplotlib
  - seaborn
  - ipykernel
  - scikit-learn
  - xgboost

The environment can  be installed using the following commands:
```
conda env create -f wifop.yml
```

Once installed the conda environments can be activated and deactivated using the following commands:

```
conda activate wifop
conda deactivate wifop
```

The following command is helpful in the event that we install more tools in the environments and need to update them without having to remove and reinstall the environment:

```
conda env update --name <environment_name> --file environment.yml --prune
```

### Datasets:

  - NASA POWER: Source - [https://power.larc.nasa.gov/api/temporal/daily/regional](https://power.larc.nasa.gov/)
      - T2M: Temperature 2 Meters (°C), daily mean
      - RH2M: 2-meter Relative Humidity, daily mean
      - PRECTOT: Total Precipitation, daily total
      - WS10M: 10-meter Wind Speeds (m/s), daily mean
      - LAT: Latitude
      - LON: Longitude
      - DATE: Date of collection
   
      - Cleaning steps:
          - unified the data by concatenating all the datasets for each of the factors.
          - changed the date to be the standard date format
          - total of 10k cells.
