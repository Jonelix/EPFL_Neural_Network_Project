# Functions
from src.utils.settings import Settings
from src.models import model_training as m_train 
from src.models import model_testing as m_test 
from src.utils.settings import Settings
from src.utils.loss_functions import Loss, get_loss_name
from src.utils import utils

# Models
from src.models import model_settings as m_settings

"""
BEST FOUND
    SEGNET: 
        - LOSS DICE
        - Lr 1e-3
        - ks 3
    UNET:
        - LOSS DICE
        - Lr 1e-3
        - ks 3
    VIT:
        - LOSS DICE
        - Lr 1e-3
        - image_size = 400
        - patch_size = 20
        - in_channels = 3
        - out_channels = 1
        - embed_size = 768
        - num_blocks = 12
        - num_heads = 8
        - dropout = 0.2
"""

NUM_EPOCHS = 15
PATIENCE_EPS = 0.025
PATIENCE = 10
###########################################
#              SEGNET
###########################################

lr = 1e-3
ks = 3
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]
ks_list = [3, 5, 7, 9]
best_loss = Loss.DICE
best_lr = 1e-3
#---------------------------------------------
#              LOSS
#---------------------------------------------


best_loss = None
best_fscore = -float('inf')

for loss, loss_name in Loss.list_training_options():
    print(f"\n\n {loss_name = }")
    settings = Settings(LOSS=loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.Segnet(LR=lr, KS=ks)    
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"segnet_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"segnet_{loss_name}_k{ks}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_loss = loss
    
    del model_to_test
    
print(f"-----------------------")
print(f"Best found loss: {best_loss} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              LR
#---------------------------------------------

best_lr = None
best_fscore = -float('inf')

for lr in lr_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.Segnet(LR=lr, KS=ks)
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"segnet_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"segnet_{get_loss_name(best_loss)}_k{ks}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_lr = lr
        
    del model_to_test

print(f"-----------------------")
print(f"Best found lr: {best_lr} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              KS
#---------------------------------------------

best_ks = None
best_fscore = -float('inf')

for ks in ks_list:
    print(f"\n\n {ks = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.Segnet(LR=lr, KS=ks, PRE_TRAINED=False)
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"segnet_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"segnet_{get_loss_name(best_loss)}_k{ks}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_ks = ks
    
    del model_to_test

print(f"Best found kr: {best_ks} with {best_fscore = :0.4f}")

print("#############################")
print(f"BEST FOR SEGNET: {get_loss_name(best_loss) = }, {best_ks = }, {best_lr = }")
print("#############################")


###########################################
#              UNET
###########################################

lr = 1e-3
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]

#---------------------------------------------
#              LOSS
#---------------------------------------------

best_loss = None
best_fscore = -float('inf')

for loss, loss_name in Loss.list_training_options():
    print(f"\n\n {loss_name = }")
    settings = Settings(LOSS=loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.UNet(LR=lr)
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"unet_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"unet_{loss_name}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_loss = loss
    
    del model_to_test

print(f"-----------------------")
print(f"Best found loss: {loss} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              LR
#---------------------------------------------

best_lr = None
best_fscore = -float('inf')

for lr in lr_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.UNet(LR=lr)
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"unet_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"unet_{get_loss_name(best_loss)}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_lr = lr

    del model_to_test
    
print(f"Best found lr: {best_lr} with {best_fscore = :0.4f}")

print("#############################")
print(f"BEST FOR UNET: {get_loss_name(best_loss) = }, {best_lr = }")
print("#############################")

###########################################
#              VIT
###########################################
lr = 1e-3
patch_size = 20
embed_size = 768
num_blocks = 8
num_heads = 8
dropout = 0.2

lr_list = [1e-4, 1e-3, 1e-2, 1e-1]
patch_size_list = [16, 20, 32]
embed_size_list = [512, 768, 1024]
num_blocks_list = [6, 8, 12]
num_heads_list = [4, 8, 16]
dropout_list =  [0.1, 0.2, 0.3]

#---------------------------------------------
#              LOSS
#---------------------------------------------
best_loss = None
best_fscore = -float('inf')

for loss, loss_name in Loss.list_training_options():
    print(f"\n\n {loss_name = }")
    settings = Settings(LOSS=loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.ViT(
        LR=lr, patch_size=patch_size, embed_size=embed_size, 
        num_blocks=num_blocks, num_heads=num_heads, dropout=dropout
    )
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"vit_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"vit_{loss_name}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_loss = loss
        
    del model_to_test

print(f"-----------------------")
print(f"Best found loss: {loss} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              LR
#---------------------------------------------
best_lr = None
best_fscore = -float('inf')

for lr in lr_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.ViT(
        LR=lr, patch_size=patch_size, embed_size=embed_size, 
        num_blocks=num_blocks, num_heads=num_heads, dropout=dropout
    )
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"vit_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"vit_{get_loss_name(best_loss)}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_lr = lr
        
    del model_to_test

print(f"-----------------------")
print(f"Best found lr: {best_lr} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              PATCH SIZE
#---------------------------------------------
best_patch_size = None
best_fscore = -float('inf')

for patch_size in patch_size_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.ViT(
        LR=lr, patch_size=patch_size, embed_size=embed_size, 
        num_blocks=num_blocks, num_heads=num_heads, dropout=dropout
    )
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"vit_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"vit_{get_loss_name(best_loss)}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_patch_size = patch_size
        
    del model_to_test

print(f"-----------------------")
print(f"Best found pathc size: {best_patch_size} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              EMBED SIZE
#---------------------------------------------
best_embed_size = None
best_fscore = -float('inf')

for embed_size in embed_size_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.ViT(
        LR=lr, patch_size=best_patch_size, embed_size=embed_size, 
        num_blocks=num_blocks, num_heads=num_heads, dropout=dropout
    )
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"vit_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"vit_{get_loss_name(best_loss)}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_embed_size = embed_size
        
    del model_to_test

print(f"-----------------------")
print(f"Best found embed_size: {best_embed_size} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              NUM BLOCK
#---------------------------------------------
best_num_blocks = None
best_fscore = -float('inf')

for num_blocks in num_blocks_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.ViT(
        LR=lr, patch_size=best_patch_size, embed_size=best_embed_size, 
        num_blocks=num_blocks, num_heads=num_heads, dropout=dropout
    )
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"vit_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"vit_{get_loss_name(best_loss)}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_num_blocks= num_blocks

    del model_to_test
    
print(f"-----------------------")
print(f"Best found num_blocks: {best_num_blocks} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              NUM HEADS
#---------------------------------------------
best_num_heads = None
best_fscore = -float('inf')

for num_heads in num_heads_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.ViT(
        LR=lr, patch_size=best_patch_size, embed_size=best_embed_size, 
        num_blocks=best_num_blocks, num_heads=num_heads, dropout=dropout
    )
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"vit_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"vit_{get_loss_name(best_loss)}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_num_heads= num_heads
        
    del model_to_test

print(f"-----------------------")
print(f"Best found num_heads: {best_num_heads} with {best_fscore = :0.4f}")
print(f"-----------------------")

#---------------------------------------------
#              DROPOUT
#---------------------------------------------
best_dropout = None
best_fscore = -float('inf')

for dropout in dropout_list:
    print(f"\n\n {lr = }")
    settings = Settings(LOSS=best_loss, NUM_EPOCHS=NUM_EPOCHS, PATIENCE_EPS=PATIENCE_EPS, PATIENCE=PATIENCE)
    model_settings = m_settings.ViT(
        LR=lr, patch_size=best_patch_size, embed_size=best_embed_size, 
        num_blocks=best_num_blocks, num_heads=best_num_heads, dropout=dropout
    )
    model_to_test = m_settings.get_model_from_settings(model_settings)
    
    m_train.start_training_settings(
        model=model_to_test, 
        # model_name=f"vit_{loss_name}_k{kr}_lr{lr}",
        settings=settings,
        model_settings=model_settings,
        origin="Ori",
        verbose=False
    )
    utils.save_model(model_to_test, f"vit_{get_loss_name(best_loss)}_lr{lr}_FINAL")
    
    f_score = m_test.test_model_batch_by_batch(model_to_test, settings, model_settings, origin="Ori", print_images=False)[1]
    print(f" - {f_score = :0.4f}")
    
    if f_score > best_fscore:
        best_fscore = f_score
        best_dropout = dropout
        
    del model_to_test

print(f"Best found dropout: {best_dropout} with {best_fscore = :0.4f}")

print("#############################")
print(
    f"BEST FOR VIT: {get_loss_name(best_loss) = }, {best_lr = }, {best_patch_size = }, "
    f"{best_embed_size = }, {best_num_blocks = }, {best_num_heads = }"
)
print("#############################")
