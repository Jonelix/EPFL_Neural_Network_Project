# Functions
from src.utils.settings import Settings
from src.models import model_training as m_train 
from src.models import model_testing as m_test 
from src.utils.settings import Settings
from src.utils.loss_functions import Loss, get_loss_name
from src.utils import utils
from src.data_processing import image_transformation as it
from src.scripts import pipelines as pipe
from data_processing import image_aumentation as ia
# Models
from src.models import model_settings as m_settings


NUM_EPOCHS = 15
PATIENCE_EPS = 0.025
PATIENCE = 10

def train_with_exclusion(model_to_test, settings, model_settings, model_name, transformation_to_exclude):
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"segnet_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Custom",
        verbose=False,
        exclude=transformation_to_exclude
    )
    utils.save_model(model_to_test, model_name)
    
    f_score = m_test.test_model_batch_by_batch(
        model_to_test, 
        settings, 
        model_settings, 
        origin="Custom", 
        print_images=False, 
    )[1] #fscore 
    print(f" - {f_score = :0.4f}")
    
    del model_to_test
    return f_score


all_model_settings = [
    ("SEGNET", m_settings.Segnet()),
    ("UNET", m_settings.Unet()),
    ("VIT", m_settings.ViT()),
]
settings = Settings(
    LOSS=Loss.DICE, 
    NUM_EPOCHS=NUM_EPOCHS, 
    PATIENCE_EPS=PATIENCE_EPS, 
    PATIENCE=PATIENCE,
    START_TEST_P=0.0,
    END_TEST_P=0.15 # Little data so it doesn't take tht much
)  
for (name, model_settings) in all_model_settings:
    print("#############################")
    print("VALIDATING " + name)
    print("#############################")
    
    model_to_test = m_settings.get_model_from_settings(model_settings)
    model_name = f"segnet_testing_ori_{get_loss_name(settings.LOSS)}_ex{transformation.value[0]}_FINAL"
    
    # Normal training to see how it performs with all the transformations
    f_score_base = train_with_exclusion(model_to_test, settings, model_settings, model_name, [None]) 
    exclude = []
    
    for transformation in it.Transformation:
        model_to_test = m_settings.get_model_from_settings(model_settings)
        model_name = f"segnet_testing_ori_{get_loss_name(settings.LOSS)}_ex{transformation.value[0]}_FINAL"
        
        f_score = train_with_exclusion(model_to_test, settings, model_settings, model_name, [transformation])
        
        if f_score > f_score_base:
            exclude.append(transformation.value[0])
        
    print(f"-----------------------")
    print(f"EXCLUDE THE FOLLOWING {exclude} FOR {name}")
    print(f"-----------------------")
        
        
        
        
        
    

