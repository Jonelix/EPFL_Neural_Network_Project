from dataclasses import dataclass
import os
from ..utils.loss_functions import Loss
    
@dataclass
class Settings():
    """
    Configuration settings for training, validation, and testing of the segmentation model.

    Parameters include file paths for images and ground truth data, model training settings such as 
    epochs, batch size, and loss functions, as well as thresholds for image processing and splitting 
    data into different sets.

    Attributes
    ----------
    ORIGINAL_IMAGE_RUTE : str
        Path to the original training images.
    ORIGINAL_GT_RUTE : str
        Path to the original ground truth images.
    BASE_IMAGE_RUTE : str
        Path to the base images.
    BASE_GT_RUTE : str
        Path to the base ground truth images.
    CUSTOM_IMAGE_RUTE : str
        Path to custom images.
    CUSTOM_GT_RUTE : str
        Path to custom ground truth images.
    SUBMISSION_IMAGE_RUTE : str
        Path to the test set images.
    BLOCK_SEED : int
        Seed for random block generation.
    PIPELINES_X_IMAGES : int
        Number of images per pipeline.
    IMAGE_FORMAT : str
        Image format (e.g., ".png").
    IMAGE_PATH_DIGITS : int
        Number of digits in the image path.
    FOREGROUND_THRESHOLD : float
        Threshold for foreground segmentation.
    START_TRAIN_P : float
        Proportion of data to use for training.
    END_TRAIN_P : float
        End proportion of data for training.
    START_VALIDATION_P : float
        Start proportion of data for validation.
    END_VALIDATION_P : float
        End proportion of data for validation.
    START_TEST_P : float
        Start proportion of data for testing.
    END_TEST_P : float
        End proportion of data for testing.
    NUM_EPOCHS : int
        Number of training epochs.
    VERBOSE_INTERVAL : int
        Interval for printing training status.
    LOSS : Loss
        Loss function to be used (e.g., DICE).
    BACHT_SIZE : int
        Batch size for training.
    PATIENCE : int
        Patience for early stopping.
    PATIENCE_EPS : float
        Patience margin for early stopping.
    """
    
    ORIGINAL_IMAGE_RUTE: str = r"data\training\images"#\satImage_"
    ORIGINAL_GT_RUTE: str = r"data\training\groundtruth"#\satImage_"
    BASE_IMAGE_RUTE: str = r"data\base\images"
    BASE_GT_RUTE: str = r"data\base\groundtruth"
    CUSTOM_IMAGE_RUTE: str = r"data\custom\images"
    CUSTOM_GT_RUTE: str = r"data\custom\groundtruth"
    SUBMISSION_IMAGE_RUTE: str = r"data\test_set_images"

    BLOCK_SEED: int = 42
    PIPELINES_X_IMAGES: int = 1 #N <= 1
    
    IMAGE_FORMAT: str = ".png"
    IMAGE_PATH_DIGITS: int = 3
    
    FOREGROUND_THRESHOLD: int = 0.25 # to conver all the gt images to 0 or 1
    
    # proportion % or index 
    START_TRAIN_P: float = 0.00
    END_TRAIN_P: float = 0.90
    START_VALIDATION_P: float = 0.90
    END_VALIDATION_P: float = 0.95
    START_TEST_P: float = 0.95
    END_TEST_P: float = 1.00    
    
    NUM_EPOCHS: int = 100
    VERBOSE_INTERVAL: int = 250
    LOSS: Loss = Loss.DICE # 0: CrossEntropy, 1: IoU Normal, 2: IoU Stable
    BACHT_SIZE: int = 10
    PATIENCE: int = 5
    PATIENCE_EPS: float = 0.01
    
    
    def __str__(self): 
        format_str = format_etr = ""
        if 0 <= self.START_TRAIN_P <= 1 and 0 <= self.END_TRAIN_P <= 1:
            format_str = format_etr = ".2%"
            
        format_sv = format_eva = ""
        if 0 <= self.START_VALIDATION_P <= 1 and 0 <= self.END_VALIDATION_P <= 1:
            format_sv = format_eva = ".2%"
            
        format_ste = format_ete = ""
        if 0 <= self.START_TEST_P <= 1 and 0 <= self.END_TEST_P <= 1:
            format_ste = format_ete = ".2%"

        return (
            f" · General Settings for trainig:\n"
            # f"   - Image Path: {self.CUSTOM_IMAGE_RUTE}\n"
            # f"   - Ground Truth Path: {self.CUSTOM_GT_RUTE}\n"
            # f"   - Image Format: {self.IMAGE_FORMAT}\n"
            # f"   - Image Path Digits: {self.IMAGE_PATH_DIGITS}\n"
            f"   - Foreground Threshold: {self.FOREGROUND_THRESHOLD}\n"
            f"   - Training Range: {self.START_TRAIN_P:{format_str}} to {self.END_TRAIN_P:{format_etr}}\n"
            f"   - Validation Range: {self.START_VALIDATION_P:{format_sv}} to {self.END_VALIDATION_P:{format_eva}}\n"
            f"   - Test Range: {self.START_TEST_P:{format_ste}} to {self.END_TEST_P:{format_ete}}\n"
            f"   - Max Nº of Epochs: {self.NUM_EPOCHS}\n"
            f"   - Verbose Interval: {self.VERBOSE_INTERVAL}\n"
            f"   - Loss: {self.LOSS}\n"
            f"   - Batch Size: {self.BACHT_SIZE}\n"
            f"   - Patience for Early Stopping: {self.PATIENCE}\n"
            f"   - Patience margin for Early Stopping: {self.PATIENCE_EPS}"
        )