import os
from PIL import Image
import random
import shutil
from typing import Literal

from src.data_processing import image_transformation as it
from src.data_processing import data as dt
from src.utils import utils
from src.utils.settings import Settings

def create_base_images(sample_directory, output_directory):
    """
    Creates base images and ground truth data by splitting input images into grids.

    This function generates directories for base images and ground truth, splits images 
    into grids of different sizes (2x2, 3x3, 4x4), and saves them with unique names.

    Parameters
    ----------
    sample_directory : str
        Path to the input directory containing 'images' and 'groundtruth' folders.
    output_directory : str
        Path to the output directory where results will be saved.

    Returns
    -------
    None
        The base images and ground truths are saved to the specified output directory.

    Example
    -------
    create_base_images("input_dir", "output_dir")
    """

    # PART ONE: CREATE DIRECTORIES
    custom_dir = os.path.join(output_directory, "base")
    os.makedirs(custom_dir, exist_ok=True)

    images_dir = os.path.join(custom_dir, "images")
    groundtruth_dir = os.path.join(custom_dir, "groundtruth")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)

    # PART TWO: CREATE BASE
    image_files = sorted(os.listdir(os.path.join(sample_directory, "images")))
    truth_files = sorted(os.listdir(os.path.join(sample_directory, "groundtruth")))
    assert len(image_files) == len(truth_files), "Mismatch between images and groundtruth files."

    index_counter = 1

    for image_file, truth_file in zip(image_files, truth_files):
        original_image = Image.open(os.path.join(sample_directory, "images", image_file))
        original_truth = Image.open(os.path.join(sample_directory, "groundtruth", truth_file))

        # Split images into grids
        two_split_image = it.split_image_into_grid(original_image, 2)
        two_split_truth = it.split_image_into_grid(original_truth, 2)

        three_split_image = it.split_image_into_grid(original_image, 3)
        three_split_truth = it.split_image_into_grid(original_truth, 3)

        four_split_image = it.split_image_into_grid(original_image, 4)
        four_split_truth = it.split_image_into_grid(original_truth, 4)

        # Combine all splits into one list
        images = [original_image] + two_split_image + three_split_image + four_split_image
        truths = [original_truth] + two_split_truth + three_split_truth + four_split_truth

        # Save split images and truths with unique names
        for img, truth in zip(images, truths):
            img.save(os.path.join(images_dir, f"satImage_{index_counter}.png"))
            truth.save(os.path.join(groundtruth_dir, f"satImage_{index_counter}.png"))
            index_counter += 1

def apply_transformation(image_in, truth_in, transformations, seed):
    """
    Applies a series of transformations to an image and its ground truth.

    The function modifies the provided image and ground truth based on the list 
    of transformations, using a consistent random seed for reproducibility.

    Parameters
    ----------
    image_in : PIL.Image.Image
        The input image to be transformed.
    truth_in : PIL.Image.Image
        The ground truth image to be transformed.
    transformations : list
        List of transformations to apply.
    seed : int
        Seed for random operations to ensure reproducibility.

    Returns
    -------
    tuple
        A tuple containing the transformed image and ground truth.

    Example
    -------
    transformed_image, transformed_truth = apply_transformation(img, truth, [trans1, trans2], 42)
    """

    image_copy = image_in.copy()
    truth_copy = truth_in.copy()

    for transformation in transformations:
        transform_func = transformation.value[1]
        # Apply transformation to the image_copy
        image_copy = transform_func(image_copy, seed)
        
        # Apply transformation to truth_copy only if it is one of the specified transformations
        if transformation in [
            it.Transformation.ROTATE,
            it.Transformation.MIRROR,
            it.Transformation.SUB,
            it.Transformation.SHUFFLE,
            it.Transformation.CIRCLES
        ]:
            truth_copy = transform_func(truth_copy, seed)

    return image_copy, truth_copy

def apply_pipeline(sample: it.SamplePoint, pipelines, base_index, out_directory, settings: Settings):
    """
    Applies transformation pipelines to a sample image and its ground truth.

    This function processes a sample point by applying a set of transformation 
    pipelines and saves the results to specified directories.

    Parameters
    ----------
    sample : it.SamplePoint
        The sample containing paths to an image and its ground truth.
    pipelines : list
        List of transformation pipelines to apply.
    base_index : int
        Base index for naming the output files.
    out_directory : str
        Path to the output directory.
    settings : Settings
        Configuration settings including seeds and other parameters.

    Returns
    -------
    None
        The transformed images and ground truths are saved to the specified directory.

    Example
    -------
    apply_pipeline(sample, pipelines, 1, "output_dir", settings)
    """

    rng = random.Random(settings.BLOCK_SEED)

    image_dir = out_directory + "/images"
    truth_dir = out_directory + "/groundtruth"

    image, truth = sample.get()
    image_in = dt.load_image_from_directory(image)
    truth_in = dt.load_image_from_directory(truth) 

    seed_for_block = rng.randint(0, 2**32-1)
    seeds = utils.split_seed(seed_for_block, settings.PIPELINES_X_IMAGES)
    for i, seed in enumerate(seeds):
        rng_image = random.Random(seed)
        seed_for_image = rng_image.randint(0, 2**32-1)
        for j, pipeline in enumerate(pipelines):
            image_out, truth_out = apply_transformation(image_in, truth_in, pipeline, seed_for_image)
            truth_out = truth_out.convert("L")
            
            image_path = image_dir + f"/satImage_{base_index}_{i+1}_{j+1}_{it.Transformation.pipe_to_name(pipeline)}.png"
            truth_path = truth_dir + f"/satImage_{base_index}_{i+1}_{j+1}_{it.Transformation.pipe_to_name(pipeline)}.png"
            image_out.save(image_path)
            truth_out.save(truth_path)



def create_training_data(sample_directory, output_directory, output_directory_name, pipelines, settings, n_divisions: Literal[0,1,2,3] = 1):
    """
    Creates training data by splitting images into grids and applying transformations.

    The function generates a dataset by splitting input images into grids of varying sizes 
    and applying specified transformation pipelines.

    Parameters
    ----------
    sample_directory : str
        Path to the input directory containing 'images' and 'groundtruth' folders.
    output_directory : str
        Path to the output directory where results will be saved.
    output_directory_name : str
        Name of the folder within the output directory to store results.
    pipelines : list
        List of transformation pipelines to apply.
    settings : Settings
        Configuration settings including seeds and parameters.
    n_divisions : Literal[0, 1, 2, 3], optional
        Number of divisions for splitting the images. Defaults to 1.

    Returns
    -------
    None
        The training data is saved to the specified output directory.

    Example
    -------
    create_training_data("input_dir", "output_dir", "training_data", pipelines, settings, n_divisions=2)
    """

    
    # PART ONE: CREATE DIRECTORIES
    custom_dir = os.path.join(output_directory, output_directory_name)
    os.makedirs(custom_dir, exist_ok=True)

    images_dir = os.path.join(custom_dir, "images")
    groundtruth_dir = os.path.join(custom_dir, "groundtruth")
    base_image_dir = os.path.join(custom_dir, "base_image")
    base_truth_dir = os.path.join(custom_dir, "base_truth")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    os.makedirs(base_image_dir, exist_ok=True)
    os.makedirs(base_truth_dir, exist_ok=True)

    # PART TWO: CREATE BASE
    image_files = sorted(os.listdir(os.path.join(sample_directory, "images")))
    truth_files = sorted(os.listdir(os.path.join(sample_directory, "groundtruth")))
    assert len(image_files) == len(truth_files), "Mismatch between images and groundtruth files."

    index_counter = 1

    for image_file, truth_file in zip(image_files, truth_files):
        original_image = Image.open(os.path.join(sample_directory, "images", image_file))
        original_truth = Image.open(os.path.join(sample_directory, "groundtruth", truth_file)).convert("L")

        # Split images into grids
        if n_divisions == 0:
            images = [original_image]
            truths = [original_truth]
        else: # n_divisions >= 1
            two_split_image = it.split_image_into_grid(original_image, 2)
            two_split_truth = it.split_image_into_grid(original_truth, 2)
            three_split_image = []
            three_split_truth = []
            four_split_image = []
            four_split_truth = []
            if n_divisions >= 2:
                three_split_image = it.split_image_into_grid(original_image, 3)
                three_split_truth = it.split_image_into_grid(original_truth, 3)
            if n_divisions >= 3:
                four_split_image = it.split_image_into_grid(original_image, 4)
                four_split_truth = it.split_image_into_grid(original_truth, 4)

            # Combine all splits into one list
            images = two_split_image + three_split_image + four_split_image
            truths = two_split_truth + three_split_truth + four_split_truth

        # Save split images and truths with unique names
        for img, truth in zip(images, truths):
            img.save(os.path.join(base_image_dir, f"satImage_{index_counter}.png"))
            truth.convert("L").save(os.path.join(base_truth_dir, f"satImage_{index_counter}.png"))
            index_counter += 1

    # PART THREE: APPLY PIPELINE TO BASE FOLDERS
    # Load all base images and truths into SamplePoint objects
    base_images = sorted(os.listdir(base_image_dir))
    base_truths = sorted(os.listdir(base_truth_dir))
    assert len(base_images) == len(base_truths), "Mismatch between base images and truths."

    sample_points = [
        it.SamplePoint(
            os.path.join(base_image_dir, img),
            os.path.join(base_truth_dir, truth)
        )
        for img, truth in zip(base_images, base_truths)
    ]

    for i, sample in enumerate(sample_points):
        original_image = Image.open(sample.get_image())
        original_truth = Image.open(sample.get_groundtruth()).convert("L") # Black and white monochannel

        # Save the original image and truth to the custom folder
        original_image.save(os.path.join(images_dir, f"satImage_{i + 1}_0_0.png"))
        original_truth.save(os.path.join(groundtruth_dir, f"satImage_{i + 1}_0_0.png"))

        # Apply pipeline transformations
        apply_pipeline(sample, pipelines, i + 1, custom_dir, settings)

    # PART FOUR: DELETE BASE IMAGE AND TRUTH FOLDERS
    shutil.rmtree(base_image_dir)
    shutil.rmtree(base_truth_dir)


def ensure_monochannel(folder_path):
    """
    Ensures all images in a folder are converted to single-channel black-and-white format.

    This function processes all files in the specified folder, converting each image 
    to monochannel format, and overwrites the original files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing images to be converted.

    Returns
    -------
    None
        The images are processed and saved in-place.

    Example
    -------
    ensure_monochannel("path/to/folder")
    """

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            try:
                image = Image.open(file_path)
                monochannel_image = image.convert("L")  # Convert to single-channel
                monochannel_image.save(file_path)  # Overwrite with the same name
                print(f"Processed and saved: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")