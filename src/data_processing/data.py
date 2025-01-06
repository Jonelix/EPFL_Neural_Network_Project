#CREATED FILE data.py
from src.data_processing import image_transformation as it
from src.utils.settings import Settings

import random
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
from pathlib import Path
from enum import Enum
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Literal
    
##########################################
#           LOADER SETTINGS
##########################################

# For further use in other parts of the project
ORI_LIT = Literal["Ori", "Base", "Custom", "Submission"] 
PUR_LIT = Literal["Train", "Val", "Test", "All"]

def get_loader_settings(
    settings: Settings, 
    origin: ORI_LIT = "Ori", 
    purpose: PUR_LIT = "Train",
    origin_img_custom: str = None,
    origin_gt_custom: str = None,
    exclude: list = []
):
    """
    Generates data loader settings based on the specified parameters.

    This function determines the paths for image and ground truth data based on the origin parameter 
    and sets up the data range and batch size according to the purpose of the operation (e.g., Train, Val, Test).

    Parameters
    ----------
    settings : Settings
        A configuration object containing paths and settings for data loading.
    origin : ORI_LIT, optional
        The source of the data paths. Options include "Ori", "Base", "Custom", or "Submission".
        Defaults to "Ori".
    purpose : PUR_LIT, optional
        The purpose for loading the data. Options include "Train", "Val", "Test", or "All".
        Defaults to "Train".
    origin_img_custom : str, optional
        A custom path for the image data. Overrides the `origin` setting if provided.
        Defaults to None.
    origin_gt_custom : str, optional
        A custom path for the ground truth data. Overrides the `origin` setting if provided.
        Defaults to None.
    exclude : list, optional
        A list of data items to exclude from the loader. Defaults to an empty list.

    Returns
    -------
    tuple
        A tuple containing:
            - image_path (str): Path to the image data.
            - gt_path (str): Path to the ground truth data.
            - batch_size (int): Batch size for data loading.
            - start_p (float): Start proportion of the dataset.
            - end_p (float): End proportion of the dataset.
            - exclude (list): Excluded items.

    Example
    -------
    settings = Settings()
    image_path, gt_path, batch_size, start_p, end_p, exclude = get_loader_settings(settings, origin="Ori", purpose="Train")
    """

    if (origin_img_custom is not None) or (origin_gt_custom is not None):
        image_path = origin_img_custom
        gt_path = origin_gt_custom
    elif origin == "Ori":
        image_path = settings.ORIGINAL_IMAGE_RUTE
        gt_path = settings.ORIGINAL_GT_RUTE
    elif origin == "Base":
        image_path = settings.BASE_IMAGE_RUTE
        gt_path = settings.BASE_GT_RUTE
    elif origin == "Custom":
        image_path = settings.CUSTOM_IMAGE_RUTE
        gt_path = settings.CUSTOM_GT_RUTE
    else: #if origin == "Submission":
        image_path = settings.SUBMISSION_IMAGE_RUTE
        gt_path = None
        
    if purpose == "Train":
        start_p = settings.START_TRAIN_P
        end_p = settings.END_TRAIN_P
    elif purpose == "Val":
        start_p = settings.START_VALIDATION_P
        end_p = settings.END_VALIDATION_P
    elif purpose == "Test":
        start_p = settings.START_TEST_P
        end_p = settings.END_TEST_P
    else: #if purpose == "All":
        start_p = 0
        end_p = 1 # 100% of the data
    
    return (image_path, gt_path, settings.BACHT_SIZE, start_p, end_p, exclude)    


def count_images_from_settings(image_path, gt_path, BACHT_SIZE, start_p, end_p):
    """
    Counts the number of images available for loading based on the provided settings.

    This function validates the input proportions or indices and calculates the number of images 
    and batches to be processed.

    Parameters
    ----------
    image_path : str
        Path to the directory containing image data.
    gt_path : str
        Path to the directory containing ground truth data.
    BACHT_SIZE : int
        The batch size for data processing.
    start_p : float
        The start proportion or index of the dataset.
    end_p : float
        The end proportion or index of the dataset.

    Returns
    -------
    tuple
        A tuple containing:
            - count (int): Total number of images to process.
            - full_batches (int): Number of full batches.
            - remaining (int): Number of remaining images not fitting into a full batch.

    Raises
    ------
    AssertionError
        If the number of images and ground truth files do not match or if the proportions are invalid.

    Example
    -------
    count, full_batches, remaining = count_images_from_settings(image_path, gt_path, 32, 0.0, 0.8)
    """

    train_images = get_image_paths(image_path)
    gt_images = get_image_paths(gt_path)
    nt = len(train_images)
    ng = len(gt_images)
    assert nt == ng, "THERE SHOUDL BE THE SAME NUMBER OF IMAGES AND GTs"
    
    if not ((0 <= start_p <= 1) and (0 <= end_p <= 1)):    
        # range of images is by "index" insted of %
        assert (0 <= start_p <= nt) and (0 <= end_p <= nt), "THE IDX SHOULD BE IN THE IN {0...Number of images in path}"
        start_p = start_p / len(train_images) 
        end_p = end_p / len(gt_images)
        
    assert (0 <= start_p <= 1) and (0 <= end_p <= 1), "THE PROPORTIOS SHOULD BE BETWEEN 1 AND 0"
    assert start_p <= end_p, "START PROPORTION SHOULD BE SMALLER OR EQUAL THAN THE END PROPORTION"
    
    start_idx = int(nt * start_p)
    end_idx = int(nt * end_p) 
    
    count = (end_idx-start_idx + 1)
    return count, count//BACHT_SIZE, count%BACHT_SIZE
    
##########################################
#           DISPLAY IMAGES
##########################################

def display_images_side_by_side_pil(sat_img, truth_img, left_text = "Satellite Image", right_text = "Ground Truth"):

    """
    Displays two images side by side: a satellite image and its corresponding ground truth.

    Parameters
    ----------
    sat_img : Image.Image
        The satellite image as a PIL Image object.
    truth_img : Image.Image
        The ground truth image as a PIL Image object.
    left_text : str, optional
        The title for the left image (satellite image). Defaults to "Satellite Image".
    right_text : str, optional
        The title for the right image (ground truth image). Defaults to "Ground Truth".

    Returns
    -------
    None
        Displays the two images side by side using matplotlib.

    Example
    -------
    from PIL import Image
    sat_img = Image.open("sat_image.png")
    truth_img = Image.open("ground_truth.png")
    display_images_side_by_side_pil(sat_img, truth_img)
    """

    # Convert PIL images to NumPy arrays for display with matplotlib
    sat_img_array = np.array(sat_img)
    truth_img_array = np.array(truth_img)
    
    if sat_img_array.shape[-1] != 3:
        sat_img_array = np.transpose(sat_img_array, (1, 2, 0))
        
    if truth_img_array.shape[-1] != 3:
        truth_img_array = np.transpose(truth_img_array, (1, 2, 0))
        

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the satellite image
    axes[0].imshow(sat_img_array)
    axes[0].set_title(left_text)
    axes[0].axis('off')

    # Display the ground truth image, handling grayscale correctly
    if len(truth_img_array.shape) == 2:  # Grayscale image
        axes[1].imshow(truth_img_array, cmap='gray')
    else:  # RGB image
        axes[1].imshow(truth_img_array)
    axes[1].set_title(right_text)
    axes[1].axis('off')

    # Show the plots
    plt.show()

def display_6_images(satellites, groundtruth, predictions):
    """
    Displays six images each from satellite, ground truth, and prediction datasets in a grid.

    The function organizes the images into three rows:
    1. Satellite images.
    2. Ground truth images.
    3. Prediction images.

    Parameters
    ----------
    satellites : list or torch.Tensor
        A list or tensor of satellite images.
    groundtruth : list or torch.Tensor
        A list or tensor of ground truth images.
    predictions : list or torch.Tensor
        A list or tensor of prediction images.

    Returns
    -------
    None
        Displays the images in a 3x6 grid using matplotlib.

    Raises
    ------
    AssertionError
        If any of the input arrays have fewer than 6 images.

    Example
    -------
    display_6_images(satellites, groundtruth, predictions)
    """


    # Ensure the input lists have at least 6 images
    num_images = 6
    assert len(satellites) >= num_images, f"Satellites array must have at least {num_images} images."
    assert len(groundtruth) >= num_images, f"Groundtruth array must have at least {num_images} images."
    assert len(predictions) >= num_images, f"Predictions array must have at least {num_images} images."

    # Resize images to make them bigger (e.g., 256x256)
    def resize_images(images, size=(256, 256)):
        # return [img.resize(size) for img in images]
        # return torch.nn.functional.interpolate(images, size=size, mode='bilinear', align_corners=False)
        if isinstance(images, list):
            images = torch.stack([torch.tensor(img) for img in images], dim=0)
    
    # Ensure images are in the right shape (B, C, H, W)
        if len(images.shape) == 3:  # In case images are in (C, H, W) format (single image)
            images = images.unsqueeze(0)  # Add batch dimension
        return torch.nn.functional.interpolate(images, size=size, mode='bilinear', align_corners=False)

    satellites = resize_images(satellites[:num_images])
    groundtruth = resize_images(groundtruth[:num_images])
    predictions = resize_images(predictions[:num_images])
    satellites = np.transpose(satellites, (0, 2, 3, 1))
    groundtruth = np.transpose(groundtruth, (0, 2, 3, 1))
    predictions = np.transpose(predictions, (0, 2, 3, 1))
    # Plot a 2x6 grid
    def plot_grid(images, title, axes_row):
        for col, ax in enumerate(axes_row):
            idx = col
            if idx < len(images):
                ax.imshow(np.array(images[idx]))
                ax.axis('off')
        # Add a title over the row
        axes_row[0].annotate(title, xy=(0, 0), xycoords="axes fraction", fontsize=12, weight="bold",
                             xytext=(-10, 1.5), textcoords="offset points", ha='left', va='center')

    # Adjust figure size and plot grids with closer spacing
    plt.close('all')  # Close previous plots to free memory
    fig, axes = plt.subplots(3, 6, figsize=(16, 10))  # 3 rows, 6 columns, bigger figure size
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Reduce space between subplots

    # Display each grid
    plot_grid(satellites, "", axes[0])
    plot_grid(groundtruth, "", axes[1])
    plot_grid(predictions, "", axes[2])

    # Show the grids
    plt.show()

def display_three_images(original_img, sat_img, truth_img, left_text = "Satellite Image", mid_text="Modified", right_text = "Groundtruth"):
    """
    Displays three images side by side: the original image, a satellite image, and its ground truth.

    Parameters
    ----------
    original_img : Image.Image
        The original image as a PIL Image object.
    sat_img : Image.Image
        The satellite image as a PIL Image object.
    truth_img : Image.Image
        The ground truth image as a PIL Image object.
    left_text : str, optional
        The title for the first image (original image). Defaults to "Satellite Image".
    mid_text : str, optional
        The title for the second image (satellite image). Defaults to "Modified".
    right_text : str, optional
        The title for the third image (ground truth image). Defaults to "Groundtruth".

    Returns
    -------
    None
        Displays the three images side by side using matplotlib.

    Example
    -------
    display_three_images(original_img, sat_img, truth_img)
    """

    # Convert PIL images to NumPy arrays for display with matplotlib
    original_img_array = np.array(original_img)
    sat_img_array = np.array(sat_img)
    truth_img_array = np.array(truth_img)
    
    # print(f"{original_img_array.shape = }")
    # print(f"{sat_img_array.shape = }")
    # print(f"{truth_img_array.shape = }")
    
    if original_img_array.shape[-1] != 3:
        original_img_array = np.transpose(original_img_array, (1, 2, 0))
        
    if sat_img_array.shape[-1] != 3:
        sat_img_array = np.transpose(sat_img_array, (1, 2, 0))
    
    # if truth_img_array.shape[-1] != 3:
        # truth_img_array = np.transpose(truth_img_array, (1, 2, 0))
    if len(truth_img_array.shape) == 2:  # Single channel
        truth_img_array = np.repeat(truth_img_array[..., None], 3, axis=-1)
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original image
    axes[0].imshow(original_img_array)
    axes[0].set_title(left_text)
    axes[0].axis('off')

    # Display the satellite image
    axes[1].imshow(sat_img_array, cmap='gray')
    axes[1].set_title(mid_text)
    axes[1].axis('off')

    # Display the ground truth image, handling grayscale correctly
    if len(truth_img_array.shape) == 2:  # Grayscale image
        axes[2].imshow(truth_img_array, cmap='gray')
    else:  # RGB image
        axes[2].imshow(truth_img_array, cmap='gray')
    axes[2].set_title(right_text)
    axes[2].axis('off')

    # Show the plots
    plt.show()
    
##########################################
#           LOAD IMAGES
##########################################
     
def load_image_from_directory(image_path):
    """
    Loads an image from a specified file path.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    Image.Image or None
        The loaded image as a PIL Image object, or None if an error occurs.

    Example
    -------
    image = load_image_from_directory("path/to/image.png")
    """

    try:
        # Open the image
        img = Image.open(image_path)
        return img
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def load_images_from_directory(directory_path):
    """
    Loads all images from a specified directory and its subdirectories.

    This function searches for `.png` files in the given directory (and subdirectories) 
    and loads them as PIL Image objects.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing image files.

    Returns
    -------
    list
        A list of PIL Image objects. Returns an empty list if no images are found or an error occurs.

    Example
    -------
    images = load_images_from_directory("path/to/images/")
    """

    images = []
    
    try:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.lower().endswith(('.png')):
                    try:
                        # Open and append the image to the list
                        img = Image.open(file_path)
                        images.append(img)
                    except Exception as e:
                        print(f"An error occurred while loading '{file_path}': {e}")
        
        if not images:
            print(f"No images found in the directory: {directory_path}")
            
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return images

def load_image_batch_by_batch_all(base_image_rute, base_gt_rute, batch_size):
    """
    Loads images and ground truth data in batches, covering the entire dataset.

    This function acts as a wrapper for `load_image_batch_by_batch` with start and end proportions
    set to cover 100% of the data.

    Parameters
    ----------
    base_image_rute : str
        Path to the directory containing the image data.
    base_gt_rute : str
        Path to the directory containing the ground truth data.
    batch_size : int
        Number of images to load in each batch.

    Returns
    -------
    generator
        A generator yielding batches of image tensors and corresponding target tensors.

    Example
    -------
    for inputs, targets in load_image_batch_by_batch_all(image_path, gt_path, batch_size):
        # Process the batch
    """

    return load_image_batch_by_batch(base_image_rute, base_gt_rute, batch_size, 0, 1)
        
def load_image_batch_by_batch(base_image_rute, base_gt_rute, batch_size, start_p, end_p, exclude):
    """
    Loads images and ground truth data in batches based on specified proportions or indices.

    This function divides the dataset into batches, allowing for partial loading based on 
    start and end proportions. Images are transformed into tensors and resized to a fixed size.

    Parameters
    ----------
    base_image_rute : str
        Path to the directory containing the image data.
    base_gt_rute : str
        Path to the directory containing the ground truth data.
    batch_size : int
        Number of images to load in each batch.
    start_p : float
        Start proportion or index of the dataset (between 0 and 1).
    end_p : float
        End proportion or index of the dataset (between 0 and 1).
    exclude : list
        List of transformations or file types to exclude.

    Returns
    -------
    generator
        A generator yielding tuples of:
            - inputs (torch.Tensor): Batched image tensors.
            - targets (torch.Tensor): Batched ground truth tensors.

    Raises
    ------
    AssertionError
        If the number of images and ground truth files do not match, or if the proportions are invalid.

    Example
    -------
    for inputs, targets in load_image_batch_by_batch(image_path, gt_path, 32, 0.0, 0.5, exclude=[]):
        # Process the batch
    """

    train_images = get_image_paths(base_image_rute)
    gt_images = get_image_paths(base_gt_rute)
    nt = len(train_images)
    ng = len(gt_images)
    assert nt == ng, "THERE SHOUDL BE THE SAME NUMBER OF IMAGES AND GTs"
    
    if not ((0 <= start_p <= 1) and (0 <= end_p <= 1)):    
        # range of images is by "index" insted of %
        assert (0 <= start_p <= nt) and (0 <= end_p <= nt), "THE IDX SHOULD BE IN THE IN {0...Number of images in path}"
        start_p = start_p / len(train_images) 
        end_p = end_p / len(gt_images) 
        
        
    assert (0 <= start_p <= 1) and (0 <= end_p <= 1), "THE PROPORTIOS SHOULD BE BETWEEN 1 AND 0"
    assert start_p <= end_p, "START PROPORTION SHOULD BE SMALLER OR EQUAL THAN THE END PROPORTION"
    
    start_idx = int(nt * start_p)
    end_idx = int(nt * end_p)
    
    print(f"  - loading {end_idx-start_idx} images...")
        
    train_images = train_images[start_idx:end_idx]
    gt_images = gt_images[start_idx:end_idx]
    
    
    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # Resize to the required size
        transforms.ToTensor()
    ]) 

    inputs = []
    target = []
    for img_path, gt_path in zip(train_images, gt_images):
        # If we are training without a certain type of transformations dont use this images
        for excluded_trans in exclude:
            if excluded_trans in it.Transformation.name_to_pipe(img_path):
                continue
        
        inputs.append(transform(Image.open(img_path)))
        target_raw = transform(Image.open(gt_path))
        target.append(
            torch.where(
                target_raw > Settings.FOREGROUND_THRESHOLD, 
                torch.tensor(1.0), torch.tensor(0.0)
            ) 
        )

        if len(inputs) == batch_size:
            yield torch.stack(inputs), torch.stack(target)
            inputs, target = [], []
            
    if inputs and target:
        yield torch.stack(inputs), torch.stack(target)   
        
def load_single_image_batch(base_image_rute, base_gt_rute, batch_size, start_p, end_p, exclude = [], special_sort = False):
    """
    Loads images in batches from a specified directory, optionally sorted in a special way.

    This function generates batches of images as tensors. A subset of the dataset can be selected 
    using start and end proportions or indices. Images can also be sorted based on custom criteria.

    Parameters
    ----------
    base_image_rute : str
        Path to the directory containing the image data.
    base_gt_rute : str
        Path to the directory containing the ground truth data (not used here but retained for consistency).
    batch_size : int
        Number of images to load in each batch.
    start_p : float
        Start proportion or index of the dataset (between 0 and 1).
    end_p : float
        End proportion or index of the dataset (between 0 and 1).
    exclude : list, optional
        List of transformations or file types to exclude. Defaults to an empty list.
    special_sort : bool, optional
        Whether to sort the images using a special key based on regex matching. Defaults to False.

    Returns
    -------
    generator
        A generator yielding batches of image tensors.

    Example
    -------
    for batch in load_single_image_batch(image_path, None, 32, 0.0, 0.5, special_sort=True):
        # Process the batch
    """
    train_images = get_image_paths(base_image_rute)
    if special_sort:
        # train_images = sorted(train_images, key=lambda path: int(re.search(r'_(\d+)', os.path.basename(os.path.dirname(path))).group(1)))
        train_images = sorted(train_images, key=lambda path: int(re.search(r'_(\d+)', os.path.basename(path)).group(1)))
    # print("-" + " ".join(train_images))
    
    if not ((0 <= start_p <= 1) and (0 <= end_p <= 1)):    
        # range of images is by "index" insted of %
        start_p = start_p / len(train_images) 
        end_p = end_p / len(train_images) 
    
    nt = len(train_images)
    
    start_idx = int(nt * start_p)
    end_idx = int(nt * end_p)
        
    train_images = train_images[start_idx:end_idx]
    
    transform = transforms.Compose([
        # transforms.Resize((400, 400)),  # Resize to the required size
        transforms.ToTensor()
    ]) 

    inputs = []
    for img_path in train_images:
        inputs.append(transform(Image.open(img_path)))
        
        if len(inputs) == batch_size:
            yield torch.stack(inputs)
            inputs = []
            
    if inputs:
        yield torch.stack(inputs)

def get_image_paths(directory):
    """
    Gets paths to all image files in a specified directory and its subdirectories.

    This function uses a recursive search to find image files with `.png` extensions 
    in the given directory and returns their paths.

    Parameters
    ----------
    directory : str
        Path to the directory to search for image files.

    Returns
    -------
    list
        A list of strings representing the file paths of the images.

    Example
    -------
    image_paths = get_image_paths("path/to/images/")
    """

    image_paths = []
    # Use Path to recursively get image files in subdirectories
    for image_file in Path(directory).rglob('*'):
        if image_file.is_file() and image_file.suffix.lower() in {'.png'}:
            image_paths.append(str(image_file))
    return image_paths

##########################################
#           SAVE IMAGES
##########################################
def save_images(images, path = r"results"):
    """
    Saves a list of image tensors as PNG files to a specified directory.

    This function converts each tensor to a PIL image and saves it with sequential names 
    in the format `test_<number>.png`.

    Parameters
    ----------
    images : list
        A list of image tensors to be saved.
    path : str, optional
        The directory where the images will be saved. Defaults to "results".

    Returns
    -------
    None
        The images are saved to the specified directory.

    Example
    -------
    save_images(image_tensors, path="output/")
    """

    to_pil = transforms.ToPILImage()
    for i, image in enumerate(images, 1):
        to_pil(image).save(fr"{path}\test_{i}.png", format="PNG")
