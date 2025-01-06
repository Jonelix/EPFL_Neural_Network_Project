from src.utils import utils 
from src.data_processing import data as dt
from src.utils.settings import Loss
from src.utils.settings import Settings
from src.models.model_settings import Model_Settings

import torch
     
def test_model_batch_by_batch(model, settings: Settings, model_settings: Model_Settings, origin: dt.ORI_LIT, print_images = True, print_losses = True):
    """
    Evaluates the model on the test dataset in a batch-by-batch manner, computing loss for each batch. 
    Optionally prints images and loss values.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    settings : Settings
        Settings containing information such as dataset path, batch size, etc.
    model_settings : Model_Settings
        Settings specific to the model, such as the threshold for binary classification.
    origin : dt.ORI_LIT
        The dataset origin, typically 'Test' for the testing dataset.
    print_images : bool, optional
        Whether to display the images during testing (default is True).
    print_losses : bool, optional
        Whether to print the losses after evaluation (default is True).

    Returns
    -------
    list of float
        A list of average loss values for each metric specified in `Loss.list_options()`.
    """
    model.eval()
    utils.to_device(model)
    
    all_inputs = []
    all_targets = []
    all_predictions = []
    total_losses = [0 for _ in range(len(list(Loss.list_options())))]
    total_images = 0
    total_test = 0
        
    data_loader_set = dt.get_loader_settings(settings, origin=origin, purpose="Test")
    data_loader_test = dt.load_image_batch_by_batch(*data_loader_set)
    for inputs, targets in data_loader_test:
        inputs = utils.to_device(inputs)
        with torch.no_grad():  
            predictions = model(inputs)
            
        predictions = (predictions > model_settings.THRESHOLD).int().float()
        
        inputs = utils.to_device(inputs, "cpu")
        predictions = utils.to_device(predictions, "cpu")
        
        loss_list = [
            criterion(predictions, targets) for criterion, _ in Loss.list_options()            
        ]
        
        all_inputs += inputs
        all_targets += targets
        all_predictions += predictions
        
        total_images += len(targets)
        total_test += 1
        for i in range(len(total_losses)):
            total_losses[i] += loss_list[i]
        # for image, gt, pred in zip(image_batch, gt_batch, predictions):
    
    for idx, loss in enumerate(total_losses):
        total_losses[idx] = loss/total_test
        
        
    if print_losses:
        print(f"Model loss over all testing ({total_images} images):")
        print("\n".join([
            f'  - {opt}: {loss:0.4f}' 
            for (funct, opt), loss in zip(Loss.list_options(), total_losses)
        ]))
    
    if print_images:
        dt.display_6_images(all_inputs, all_targets, all_predictions)
    # for image, gt, pred in zip(all_inputs[:print_images], all_targets[:print_images], all_predictions[:print_images]):
    #     dt.display_three_images(all_inputs, all_targets, all_predictions)
        
    return [loss.item() for loss in total_losses]

def test_model_submission(model, settings: Settings, model_settings: Model_Settings):
    """
    Tests the model on the submission dataset and saves the predicted images.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be tested.
    settings : Settings
        Settings containing information such as dataset path, batch size, etc.
    model_settings : Model_Settings
        Settings specific to the model, such as the threshold for binary classification.
        
    Returns
    -------
    None
        This function does not return any value but saves the predicted images to disk.
    """
    model.eval()
    utils.to_device(model)
    all_inputs = []
    all_predictions = []
    
    data_loader_set = dt.get_loader_settings(settings, origin="Submission", purpose="All")
    data_loader_test = dt.load_single_image_batch(*data_loader_set, special_sort=True)
    
    print("Testing...")
    for inputs in data_loader_test:
        inputs = utils.to_device(inputs)
        with torch.no_grad():  
            predictions = model(inputs)
            
        predictions = (predictions > model_settings.THRESHOLD).int().float()
        
        inputs = utils.to_device(inputs, "cpu")
        predictions = utils.to_device(predictions, "cpu")

        
        all_inputs += inputs
        all_predictions += predictions
    
    print("Saving images...")
    dt.save_images(all_predictions)
    dt.display_6_images(all_inputs, all_predictions, all_inputs)
    
def compare_submission_patch_mask(settings, print_times = -1, submission = r"data\test_set_images", results = r"results", results_patch = r"results_patch"):   
    """
    Compares the submission images with the results and patch results, and displays them.

    Parameters
    ----------
    settings : Settings
        Settings containing information such as dataset path, batch size, etc.
    print_times : int, optional
        The number of image comparisons to print. If -1, all comparisons are printed (default is -1).
    submission : str, optional
        Path to the submission images (default is "data/test_set_images").
    results : str, optional
        Path to the results images (default is "results").
    results_patch : str, optional
        Path to the patch results images (default is "results_patch").
        
    Returns
    -------
    None
        This function does not return any value but displays the images using `dt.display_6_images`.
    """
    data_loader_set = dt.get_loader_settings(settings, origin="Submission", purpose="All")
    data_loader_test = dt.load_single_image_batch(*data_loader_set, special_sort=True)
    
    data_loader_set = dt.get_loader_settings(settings, origin_img_custom=results, purpose="All")
    data_loader_results = dt.load_single_image_batch(*data_loader_set, special_sort=True)
    
    data_loader_set = dt.get_loader_settings(settings, origin_img_custom=results_patch, purpose="All")
    data_loader_patch = dt.load_single_image_batch(*data_loader_set, special_sort=True)
    
    for idx, (sub, res, res_patch) in enumerate(zip(data_loader_test, data_loader_results, data_loader_patch)):
        if len(sub) <= 6 or (idx >= print_times and print_times != -1): 
            break
        
        dt.display_6_images(sub, res, res_patch)
    
    