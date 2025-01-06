# Functions for data
from src.data_processing import data as dt
from src.data_processing import image_transformation as it
from data_processing import image_aumentation as ia
from src.scripts import pipelines as pipe

# Utils
from src.utils.settings import Settings
from src.utils.loss_functions import Loss, get_loss_name
from src.utils import utils

# Models
from src.models import model_testing as m_test 
from src.models import model_training as m_train 
from src.models import model_settings as m_settings

# Lybraries
import torch
import os

def create_custsom_data():
    settings = Settings()

    ia.create_training_data(
        sample_directory=r"data\training",
        output_directory=r"data", 
        output_directory_name=r"custom", #Creates another folder inside "output_directory" with two folders: Images and GroundTruth
        pipelines=pipe.full_pipelines, # Select the desired pipeline
        settings=settings,
        n_divisions=3
    )

def train_segnet():
    settings = Settings()
    model_settings = m_settings.Segnet()
    model_name = f"segnet_{get_loss_name(settings.LOSS)}_k{model_settings.KS}_lr{model_settings.LR}"
    model_to_test = m_settings.get_model_from_settings(model_settings)

    m_train.start_training_settings(
        model=model_to_test, 
        model_name=model_name,
        settings=settings,
        model_settings=model_settings,
        origin="Custom"
    )
 
    m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Custom")                      
    del model_to_test
    
    
def train_unet():
    settings = Settings()
    model_settings = m_settings.UNet()
    model_name = f"unet_ori_{get_loss_name(settings.LOSS)}_lr{model_settings.LR}"

    model_to_test = m_settings.get_model_from_settings(model_settings)
    m_train.start_training_settings(
        model=model_to_test, 
        model_name=model_name,
        settings=settings,
        model_settings=model_settings,
        origin="Ori"
    )
    
    m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori")           
    del model_to_test
    
def train_vit():
    settings = Settings(LOSS=Loss.DICE)
    model_settings = m_settings.ViT()
    model_name = f"vit_{get_loss_name(settings.LOSS)}_lr{model_settings.LR}"

    model_to_test = m_settings.get_model_from_settings(model_settings)
    m_train.start_training_settings(
        model=model_to_test, 
        model_name=model_name,
        settings=settings,
        model_settings=model_settings,
        origin="Custom"
    )
                
    m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Custom")           
    del model_to_test

def main():
    create_custsom_data()
    
    train_segnet()
    train_unet()
    train_vit()

if __name__ == "__main__":
    main()