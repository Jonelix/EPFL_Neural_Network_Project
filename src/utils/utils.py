import os
import torch
from torchvision import transforms
import random
#FUNCTION FOR RENAMING FILES
def rename_images_with_leading_zero(directory, curr_len):
    """
    Renames image files in a specified directory by adding leading zeros to their numeric identifiers.

    This function processes files named in the format 'satImage_XXX.png' (where XXX is a numeric identifier)
    and renames them to 'satImage_000XXX.png', ensuring uniform naming with a specified numeric length.

    Parameters
    ----------
    directory : str
        The path to the directory containing the image files.
    curr_len : int
        The current length of the numeric part to target for renaming. Files with numeric parts
        of this length will be renamed by adding leading zeros.

    Returns
    -------
    None
        Prints the renaming operations performed (e.g., "Renamed: satImage_123.png -> satImage_0123.png") 
        and skips already properly formatted files.

    Example
    -------
    from src.utils import utils as utils

    directory = "./data/training/training/images/"

    utils.rename_images_with_leading_zero(directory, curr_len=3)
    """
    for filename in os.listdir(directory):
        # Check if the file matches the pattern 'satImage_XXX.png'
        if filename.startswith("satImage_") and filename.endswith(".png"):
            # Extract the numeric part after 'satImage_' and before '.png'
            numeric_part = filename[9:-4]  # Remove 'satImage_' and '.png'
            
            # Check if the numeric part is 3 or 4 characters long
            if numeric_part.isdigit() and len(numeric_part) == curr_len:
                # Add leading zero for 3-digit numeric parts
                new_filename = f"satImage_000{numeric_part}.png"
                
                # Get full paths
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            elif numeric_part.isdigit() and len(numeric_part) > curr_len:
                # Skip if the number is already 4 digits
                print(f"Skipped (already 4 digits): {filename}")


##########################################
#           MODEL FUNCTIONS
##########################################
def to_device(x, device=None):
    """
    Moves a tensor or model to the specified device.

    Parameters
    ----------
    x : torch.Tensor or torch.nn.Module
        The tensor or model to be moved.
    device : torch.device or None, optional
        The target device to move the tensor or model to. If None, it defaults to:
        - CUDA device if available.
        - CPU otherwise.

    Returns
    -------
    torch.Tensor or torch.nn.Module
        The input tensor or model moved to the specified or default device.
    """
    if device is not None:
        return x.to(device)
    elif torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

def save_model(model, filename, folder = r"trained_models_test"):
    """
    Saves the state dictionary of a PyTorch model to a file.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose state dictionary is to be saved.
    filename : str
        The name of the file (without extension) to save the model's state dictionary.
    folder : str, optional
        The folder where the model file will be saved. Defaults to "trained_models_test".

    Returns
    -------
    None
        Saves the model's state dictionary to the specified folder and file.
    """
    torch.save(model.state_dict(), fr"{folder}\{filename}.pth")

def load_model(model, filename, folder = r"trained_models_test"):
    """
    Loads a state dictionary into a PyTorch model from a file.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model into which the state dictionary will be loaded.
    filename : str
        The name of the file (without extension) containing the saved state dictionary.
    folder : str, optional
        The folder where the model file is located. Defaults to "trained_models_test".

    Returns
    -------
    None
        Loads the state dictionary into the given model. The `strict` flag is set to `False` 
        to allow for partial loading.

    Example
    -----
    # Create a new instance of the model
    model = VisionTransformerForSegmentation(...)

    # Load the saved state dictionary
    load_model(model, 'mega_hyper_transformer42')
    """
    model.load_state_dict(torch.load(fr"{folder}\{filename}.pth"),strict=False)

def split_seed(seed, n):
    """
    Splits a single seed into `n` unique deterministic values.

    Parameters
    ----------
    seed : int
        The base seed value used to generate deterministic random numbers.
    n : int
        The number of unique seed values to generate.

    Returns
    -------
    list of int
        A list of `n` unique deterministic values generated from the base seed.

    """
    rng = random.Random(seed)  # Create a random generator with the given seed
    return [rng.randint(0, 2**32 - 1) for _ in range(n)]