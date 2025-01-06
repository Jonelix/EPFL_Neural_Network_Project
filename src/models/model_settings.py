from dataclasses import dataclass, asdict, fields
from ..models.segmentation_segnet import Segmentation_Road_Segnet
from ..models.segmentation_transformer import VisionTransformerForSegmentation
from ..models.segmentation_unet import Segmentation_Road_UNet
  
@dataclass
class Model_Settings():
    LR: float = 0.001
    STEP_SIZE: int = 7
    GAMMA: float = 0.7
    THRESHOLD: float = 0.5
    EXC_TRANS: list = []
    def __str__(self):
        return(
            f" · Model Settings:\n"
            f"   - LR: {self.LR}\n"
            f"   - STEP_SIZE: {self.STEP_SIZE}\n"
            f"   - GAMMA: {self.GAMMA}\n"
            f"   - THRESHOLD: {self.THRESHOLD}\n"
            f"   - EXCLUDING TRANSFORMATIONS: {self.EXC_TRANS}\n"
            
        )
    
@dataclass
class Segnet(Model_Settings):
    # The best values that works for your task
    KS: int = 3 # Kernel Size
    PRETRAINED: bool = True
    def __str__(self):
        return(
            super().__str__() +
            f"   - KERNEL SIZE: {self.KS}\n"
            f"   - PRETRAINED: {self.PRETRAINED}\n"
        )
        
        
@dataclass
class UNet(Model_Settings):
    # The best values that works for your task
    PRETRAINED: bool = True
    def __str__(self):
        return (
            super().__str__() +
            f"   - PRETRAINED: {self.PRETRAINED}\n"
        )
        
@dataclass
class ViT(Model_Settings):
    image_size: int = 400
    patch_size: int = 20
    in_channels: int = 3
    out_channels: int = 1
    embed_size: int = 768
    num_blocks: int = 12
    num_heads: int = 8
    dropout: float = 0.2
    
    def __str__(self):
        return (
            super().__str__() +
            f"   - image_size: {self.image_size}\n"
            f"   - patch_size: {self.patch_size}\n"
            f"   - in_channels: {self.in_channels}\n"
            f"   - out_channels: {self.out_channels}\n"
            f"   - embed_size: {self.embed_size}\n"
            f"   - num_blocks: {self.num_blocks}\n"
            f"   - num_heads: {self.num_heads}\n"
            f"   - dropout: {self.dropout}\n"
        )
    def get_arg(self):
        # Get fields defined in the current subclass
        parent_fields = {field.name for field in fields(Model_Settings)}  # Parent fields
        all_fields = asdict(self)  # All fields in the subclass instance
        return {key: value for key, value in all_fields.items() if key not in parent_fields}

        
        
def get_model_from_settings(model_settings: Model_Settings):
    """
    Returns a model instance based on the given model settings. Depending on the type of `model_settings`,
    it returns an instance of `Segmentation_Road_Segnet`, `VisionTransformerForSegmentation`, or `Segmentation_Road_UNet`.

    Parameters
    ----------
    model_settings : Model_Settings
        The settings object containing model-specific configurations such as pretrained weights and kernel size.

    Returns
    -------
    torch.nn.Module
        A model instance of the appropriate type, either `Segmentation_Road_Segnet`, `VisionTransformerForSegmentation`, 
        or `Segmentation_Road_UNet`, based on the type of `model_settings`.
    
    Notes
    -----
    - If the `model_settings` is an instance of `Segnet`, the function returns a `Segmentation_Road_Segnet` model.
    - If the `model_settings` is an instance of `ViT`, the function returns a `VisionTransformerForSegmentation` model.
    - If neither, it defaults to returning a `Segmentation_Road_UNet` model with hardcoded parameters for the task.
    """
    if isinstance(model_settings, Segnet): 
        model_to_test = Segmentation_Road_Segnet(pre_trained=model_settings.PRETRAINED, kernel_size=model_settings.KS) 
    elif isinstance(model_settings, ViT): 
        model_to_test = VisionTransformerForSegmentation(**model_settings.get_arg())
    else: #if isinstance(model_settings, UNet): 
        model_to_test = Segmentation_Road_UNet(pre_trained=model_settings.PRETRAINED, n_channels=3, n_classes=1) # Harcoded paraneters because of the task
        
    return model_to_test
    