- [TIHM-Dataset](#tihm-dataset)
  * [Summary of Data Records](#summary-of-data-records)
    + [Activity](#activity)
    + [Labels](#labels)
    + [Physiology](#physiology)
    + [Sleep](#sleep)
    + [Demographics](#demographics)
  * [Running the code](#running-the-code)
  
# TIHM-Dataset
TIHM: An open dataset for remote healthcare monitoring in dementia.
The dataset is available on its corresponding [Zenodo repository](https://zenodo.org/record/7622128).

The dataset is provided for research and cannot be used for commercial purposes. 

Please acknowledge the Surrey and Borders Partnership NHS Foundation Trust in any publication or use of this dataset. 


## Summary of Data Records
The dataset is organised in five separate tables stored as separate CSV files, including, Activity, Sleep, Physiology, Labels and Demographics. Data can be cross-referenced across the files. 

### Activity

|               | Value Type        | Number of Values | Description                                     |
|---------------|-------------------|------------------|-------------------------------------------------|
| patient_id    | CategoricalDtype  | 56               | hash code                                       |
| location_name | CategoricalDtype  | 8               | Hallway,Lounge,Fridge Door,Bedroom,Kitchen,etc.       |
| date    | dtype[datetime64] | N/A              | from 2019-04-01 to 2019-06-30 |

### Labels

|            | Value Type        | Number of Values | Description                                         |
|------------|-------------------|------------------|-----------------------------------------------------|
| patient_id | CategoricalDtype  | 49               | hash code                                           |
| date | dtype[datetime64] | N/A              | from 2019-04-04  to 2019-06-30     |
| type       | CategoricalDtype  | 6                | Agitation,Body temperature,Weight,etc. |


### Physiology

|             | Value Type        | Number of Values | Description                                                                 |
|-------------|-------------------|------------------|-----------------------------------------------------------------------------|
| patient_id  | CategoricalDtype  | 55               | hash code                                                                   |
| date  | dtype[datetime64] | N/A              | from 2019-04-01 to 2019-06-30                             |
| device_type | CategoricalDtype  | 8                | Skin Temperature,Diastolic blood pressure,Heart rate,O/E - muscle mass,etc. |
| value       | dtype[float64]    | N/A              | min: 0.0, max: 211.0                                                        |
| unit        | CategoricalDtype  | 5                | %,kg,mm[Hg],beats/min,etc.                                                  |

### Sleep

|                  | Value Type        | Number of Values | Description                                     |
|------------------|-------------------|------------------|-------------------------------------------------|
| patient_id       | CategoricalDtype  | 17               | hash code                                       |
| date       | dtype[datetime64] | N/A              | from 2019-04-01 to 2019-06-30 |
| state            | CategoricalDtype  | 4                | LIGHT,AWAKE,DEEP,REM                            |
| heart_rate       | dtype[float64]    | N/A              | min: 37.0, max: 107.0                           |
| respiratory_rate | dtype[float64]    | N/A              | min: 8.0, max: 31.0                             |
| snoring          | dtype[bool]      | 2                | True or False                                   |


### Demographics

|               | Value Type        | Number of Values | Description                                     |
|---------------|-------------------|------------------|-------------------------------------------------|
| patient_id    | CategoricalDtype  | 56               | hash code                                       |
| sex | CategoricalDtype  | 2               | Male, Female       |
| age    | CategoricalDtype | 3              | (70, 80],(80, 90],(90, 110] |


## Running the code
We have provided raw data and guidelines on how to access, visualise, manipulate and predict health-related events within the dataset.  The Jupyter Notebooks have been developed using Python 3.9. 
For reproducing the code, an Anaconda virtual environment is also included. The virtual environment can be created using the following line of code in the Anaconda Terminal:
```
conda env create -f tihm.yml
```
After creating and activating the virtual environment, each notebook can be run individually. Please be careful to change the DPATH variable in each notebook with the folder in which the dataset has been downloaded.
```
DPATH = '../Dataset/'
```
