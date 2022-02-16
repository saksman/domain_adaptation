from . import lcfcn, csrnet, glance
from . import resnet, fcn8, inception, unet, fcn8_vgg16, discriminator



def get_model(model_name, exp_dict):
    if exp_dict['dataset'] == 'fish_loc':
        n_classes = 1
    elif exp_dict['dataset'] == 'gta5cityscapes':
        n_classes = 19
    else:
        n_classes = 2

    if model_name == "fcn8":
        model = fcn8.FCN8(n_classes)
    
    if model_name == "fcn8_vgg16":
        
        model = fcn8_vgg16.FCN8VGG16(n_classes=n_classes)

    if model_name == "unet":
        model = unet.UNET()

    if model_name == "resnet":
        model = resnet.ResNet(n_classes=1)

    if model_name == "inception":
        model = inception.inceptionresnetv2(num_classes=1)
    
    if model_name == "discriminator":
        model = discriminator.FCDiscriminator(num_classes=n_classes)

    return model
