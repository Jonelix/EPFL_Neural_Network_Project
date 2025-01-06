import torch
from torch import nn
from typing import Literal

from ..data_processing import data as dt
from ..utils.utils import to_device
from ..utils.loss_functions import Loss
from ..utils.utils import save_model
from ..utils.settings import Settings
from ..models.model_settings import Model_Settings


def start_training_settings(
    model, settings: Settings, model_settings: Model_Settings,
    model_name: str = None, origin: dt.ORI_LIT = "Ori", verbose = True,
    exclude: list = []
):
    """
    Starts the training process with the provided settings and model.

    This function configures and initiates the training procedure using the specified model and settings, 
    and handles the training and validation loops, early stopping, and model saving.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    settings : Settings
        The settings for the training, including dataset paths, batch size, and number of epochs.
    model_settings : Model_Settings
        The settings specific to the model, such as learning rate and step size.
    model_name : str, optional
        The name of the model for saving purposes. Defaults to None.
    origin : dt.ORI_LIT, optional
        The origin type of the dataset. Defaults to "Ori".
    verbose : bool, optional
        Whether to print the training progress. Defaults to True.
    exclude : list, optional
        List of transformations to exclude from the dataset. Defaults to an empty list.

    Returns
    -------
    None
    """
    data_settings_train = dt.get_loader_settings(settings, origin=origin, purpose="Train", exclude=exclude)
    data_settings_validation = dt.get_loader_settings(settings, origin=origin, purpose="Val", exclude=exclude)
    
    if verbose:
        print("Training with {} images: {} batches + {} extra".format(
            *dt.count_images_from_settings(*data_settings_train),
        ))
        print("Validating with {} images: {} batches + {} extra".format(
            *dt.count_images_from_settings(*data_settings_validation),
        ))
        if len(exclude) > 0:
            print(f"(some may be excluded because of the exclusion of this transformations: {[t.value[0] for t in exclude]})")
            
        print("\nSETTINGS:")
        print(settings)
        print(model_settings)
        print(
            f"------------------------------------------\n"
            f"TRAINING BEGINS...\n"
        )
    
    start_training(
        model=model, 
        model_name=model_name,
        data_settings_train=data_settings_train, 
        data_settings_validation=data_settings_validation, 
        verbose_interval=settings.VERBOSE_INTERVAL,
        epochs=settings.NUM_EPOCHS, 
        criterion=settings.LOSS, 
        patience=settings.PATIENCE, 
        patience_eps=settings.PATIENCE_EPS, 
        lr=model_settings.LR,
        step_size=model_settings.STEP_SIZE,
        gamma=model_settings.GAMMA, 
        verbose=verbose,
    )
    
def start_training(
    model, data_settings_train: tuple, data_settings_validation: tuple, verbose,
    epochs: int, criterion: Loss = Loss.CROSS_ENTROPY, patience: int = 5, patience_eps: float = 0.005, 
    lr: float = 0.001, step_size = 7, gamma = 0.7, verbose_interval: int = None, model_name = None
):
    """
    Starts the model training with the provided parameters.

    This function handles the training loop, early stopping based on validation loss, 
    learning rate adjustment using a scheduler, and saving of the model after each epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    data_settings_train : tuple
        The data loader settings for the training set.
    data_settings_validation : tuple
        The data loader settings for the validation set.
    verbose : bool
        Whether to print detailed progress information.
    epochs : int
        The number of epochs to train for.
    criterion : Loss, optional
        The loss function to use. Defaults to CrossEntropyLoss.
    patience : int, optional
        The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 5.
    patience_eps : float, optional
        The minimum improvement in validation loss required to reset early stopping. Defaults to 0.005.
    lr : float, optional
        The learning rate for the optimizer. Defaults to 0.001.
    step_size : int, optional
        The step size for the learning rate scheduler. Defaults to 7.
    gamma : float, optional
        The gamma value for adjusting the learning rate. Defaults to 0.7.
    verbose_interval : int, optional
        The number of iterations after which to print training progress. Defaults to None.
    model_name : str, optional
        The name of the model, used for saving the model periodically. Defaults to None.

    Returns
    -------
    None
    """
    to_device(model)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                
    best_val_loss = float('inf')  # To keep track of the best validation loss
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        data_loader_train = dt.load_image_batch_by_batch(*data_settings_train)
        data_loader_validation = dt.load_image_batch_by_batch(*data_settings_validation)
        train_model(model, data_loader_train, optimizer, criterion, verbose_interval)

        # Validation for early stopping when improvement is less than 0.005 in loss
        val_loss, validation_samples = validate_model(model, data_loader_validation, criterion)
        if verbose:
            print(f" · Validation loss ({validation_samples} samples): {val_loss:.4f}")
        
        if val_loss < best_val_loss - patience_eps:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else: 
            epochs_without_improvement +=1 
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping: No improvement in validation loss for {patience} epochs.")
            break

    
        if scheduler is not None:
            scheduler.step()
            
        if model_name is not None:
            save_model(model, f"{model_name}_epoch_{epoch:0{4}d}")
        # end if
    # end for
    save_model(model, f"{model_name}_epoch_{epoch:0{4}d}_FINAL")
# end def

def train_model(model, data_loader_train, optimizer, criterion, verbose_interval):
    """
    Trains the model on the provided training data.

    This function handles the forward pass, loss computation, and backward pass (gradient update) 
    for the training process, printing the progress if necessary.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    data_loader_train : DataLoader
        The DataLoader object for the training data.
    optimizer : torch.optim.Optimizer
        The optimizer for updating the model's weights.
    criterion : Loss
        The loss function to compute the training loss.
    verbose_interval : int, optional
        The number of iterations after which to print the training progress. Defaults to None.

    Returns
    -------
    None
    """
    model.train()
    running_loss = 0.0
    running_samples = 0
    running_batches = 0
    
    print(f"  - Started training...")
    for batch_idx, (inputs, targets) in enumerate(data_loader_train):
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        
        outputs = model(inputs)
        # print(f"{outputs = }")
        # The ground truth labels have a channel dimension (NCHW).
        # We need to remove it before passing it into
        # CrossEntropyLoss so that it has shape (NHW) and each element
        # is a value representing the class of the pixel.
        # if isinstance(criterion, nn.CrossEntropyLoss):
        #      targets = targets.squeeze(dim=1)
        #      targets = targets.long()
        # end if
        loss = criterion(outputs, targets)
        # print(f"{loss.item() = :0.4f}")
        loss.backward()
        # count = 0
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"No gradient for {name}")
        #     else:
        #         print(f"Gradient for {name}: {param.grad.norm()}")      
        #     count+=1 
        #     if count >= 1: break
        # print("...")  
        
        optimizer.step()
    
        running_samples += len(targets)
        running_loss += loss.item()
        running_batches += 1
        
        if (verbose_interval is not None) and (running_samples % verbose_interval == 0):
            print(f"  - Training {running_samples} samples, Loss: { running_loss / (batch_idx+1):.4f}")
            
    # end for

    print(f" · Trained {running_samples} samples, Loss: { running_loss / running_batches:.4f}")
# end def
    
def validate_model(model, data_loader_validation, criterion):
    """
    Validates the model on the provided validation data.

    This function computes the validation loss over the validation dataset without performing gradient updates.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be validated.
    data_loader_validation : DataLoader
        The DataLoader object for the validation data.
    criterion : Loss
        The loss function to compute the validation loss.

    Returns
    -------
    tuple
        A tuple containing the average validation loss and the number of validation samples.
    """
    model.eval()
    
    loss = 0
    validation_samples = 0
    validation_batches = 0
    with torch.no_grad():  
        for (inputs, targets) in data_loader_validation:
            inputs = to_device(inputs)
            targets = to_device(targets)
            outputs = model(inputs)
            
            loss += criterion(outputs, targets).item()
            validation_samples += len(targets)
            validation_batches += 1
    
    assert validation_batches > 0, "ERROR: To validate the model at least one sample is needed"
    return loss/(validation_batches), validation_samples
    
    
