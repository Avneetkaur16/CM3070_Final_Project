import tensorflow as tf
from src.models.vgg16CNN import create_vgg16_model
from src.models.resnet50CNN import create_resnet50_model
from src.models.densenet121CNN import create_densenet121_model
from src.models.mobilenetv2CNN import create_mobilenetv2_model
from src.models.efficientnetb0CNN import create_efficientnetb0_model

# Model selection function based on model name
def select_model(model_name):
    # VGG16
    if(model_name == 'vgg16'):
        return create_vgg16_model()
    #ResNet50
    elif(model_name == 'resnet50'):
        return create_resnet50_model()
    # DenseNet121
    elif(model_name == 'densenet121'):
        return create_densenet121_model()
    # MobileNetV2
    elif(model_name == 'mobilenetv2'):
        return create_mobilenetv2_model()
    # EfficientNet-B0
    elif(model_name == 'efficientnetb0'):
        return create_efficientnetb0_model()
    else:
        return None

# CNN model compiler function
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
    ])
    return model