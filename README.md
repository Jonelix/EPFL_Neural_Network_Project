[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)


> [!NOTE]
> Catalina Regolf, Jonathan Amdam, Miquel Gómez

## Usage of repo

In the run_notebook.ipynb it is shown how to expand the dataset, train and test the models and make the submission for the AI-Croud.

In the run.py is the code to generate and train the best possible models.

All the functions are in the src.

## Repository Structure

The repository is divided into four main parts:

- Data: In this folder should be located all the provided data and the generated one. All the hardcoded paths redirect you here
- SRC: We add all the implementations and methods to train the models here. From data expansion to the models themselves.
- Run: run.py and run_notebook.ipynb are the files used to run the code and train the models. Use the run_notebook.ipynb as interface + examples
- Others: results, results_patches and dummy_submission are images and files resulting of the delivery to the AI-crowd.


```
├── data                        <- Data source from https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files, not uploaded...
│   ├── sample_submition.csv                       
│   ├── test_set_images                         
│   │   ├── ...                 <- Folders with a single image. Each for a different test
│   ├── training                         
│   │   ├── groundtruth         
│   │   │   ├── ...             <- Images groundtruth...  
│   │   ├── images                 
│   │       ├── ...             <- Images for training...  
│   ├── custom  
│   │   ├── groundtruth         
│   │   │   ├── ...             <- Images groundtruth...  
│   │   ├── images                 
│   │       ├── ...             <- Images for training...           
│   ├── base                         
│   │   ├── groundtruth         
│   │   │   ├── ...             <- Images groundtruth...  
│   │   ├── images                 
│   │       ├── ...             <- Images for training...  
│   └── ...
│
├── helpers                     <- Provided functions
│   ├── mask_to_submission.py       <- get a submission for AI crowd from a mask image
│   ├── submission_to_mask.py       <- get a mask image from a submission
│   ├── segment_aerial_images.ipynb <- example and presentation of the task
│   └── tf_aerial_images.py         <- example and presentation of the task
├── results
│   ├── ...                     <- Resulting groundtruth from test images ...
├── results_patch
│   ├── ...                     <- Resulting groundtruth from test images as would be seen by AICrowd...
├── src                         <- Implemented functionality
│   ├── data_processing                     <- Data related functions     
│   │   ├── data.py                         <- Load, save and visualize data   
│   │   ├── image_augmentation.py           <- Augmentation functions to expand the dataset
│   │   └── image_transformation.py         <- Image transformation used to expand the dataset
│   ├── models                  <- Models creation and related functions        
│   │   ├── model_settings.py               <- The best settings for each model and a function to load them 
│   │   ├── model_testing.py                <- Functions to test the model
│   │   ├── model_training.py               <- Functions to train the model
│   │   ├── segmentation_segnet.py          <- Model structure definition
│   │   ├── segmentation_transformer.py     <- Model structure definition
│   │   └── segmentation_unet.py            <- Model structure definition
│   ├── scripts
│   │   ├── hyperparamter_optimization.py        <- Code for validation and finding best parameters for each model 
│   │   ├── image_transformation_optimizatin.py  <- Code for validation and finding which tranformations wrok the best
│   │   └── pipelines.py
│   └── utils                   <- Common functions      
│       ├── loss_functions.py               <- Declaration of the used loss funcitons for trainig, testing and validation
│       ├── settings.py                     <- Declaration of the settings to train, test and execute
│       └── utils.py                        <- Utility functions used along the repository
├── trained_models              <- Already trained (Not uploaded becase of size)
├── trained_models_test         <- Trained model during testing (Not uploaded becase of size)
├── .gitignore                  <- List of files ignored by git
├── run_notebook.ipynb          <- Interface for all the work: Data cleaning, model training...
├── run.py                      <- Single script that executes alll the processes (Data cleaning, model training...) and gets a submission.
├── dummy_submission.csv        <- AICrowd submission
└── README.md                   <- This file :I
```
