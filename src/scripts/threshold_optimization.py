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


thresholds = [i*1e-1 for i in range (1,10)]
best_thress = None
best_fscore = -float('inf')
for thres in thresholds:
    settings = Settings()

    # Select the type of model in the settings
    model_settings = m_settings.Segnet(THRESHOLD=thres)
    model_to_test = m_settings.get_model_from_settings(model_settings)
    utils.load_model(model_to_test, "segnet_regalito_custom_testing_epoch_0062")

    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori")[1]
    if f_score > best_fscore:
        best_fscore = f_score
        best_thress = thres
    
    del model_to_test
    
print(f"-----------------------")
print(f"Best thres for SEGNET loss: {best_thress} with {best_fscore = :0.4f}")
print(f"-----------------------")


thresholds = [i*1e-1 for i in range (1,10)]
best_thress = None
best_fscore = -float('inf')
for thres in thresholds:
    settings = Settings()

    # Select the type of model in the settings
    model_settings = m_settings.UNet(THRESHOLD=thres)
    model_to_test = m_settings.get_model_from_settings(model_settings)
    utils.load_model(model_to_test, "unet_ori_DICE_lr0.001_epoch_0025")

    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori")[1]
    if f_score > best_fscore:
        best_fscore = f_score
        best_thress = thres
    
    del model_to_test
    
print(f"-----------------------")
print(f"Best thres for UNet loss: {best_thress} with {best_fscore = :0.4f}")
print(f"-----------------------")


thresholds = [i*1e-1 for i in range (1,10)]
best_thress = None
best_fscore = -float('inf')
for thres in thresholds:
    settings = Settings()

    # Select the type of model in the settings
    model_settings = m_settings.ViT(THRESHOLD=thres)
    model_to_test = m_settings.get_model_from_settings(model_settings)
    utils.load_model(model_to_test, "vit_DICE_lr0.001_epoch_0004")

    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori")[1]
    if f_score > best_fscore:
        best_fscore = f_score
        best_thress = thres
    
    del model_to_test
    
print(f"-----------------------")
print(f"Best thres for ViT loss: {best_thress} with {best_fscore = :0.4f}")
print(f"-----------------------")
