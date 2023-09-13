from inference_rayserve.models import LprnetDeployment, StnDeployment, YoloDeployment
from inference_rayserve.plate_recognition import PlateRecognitionDeployment
from inference_rayserve.settings import settings
from nn.settings import settings as settings_nn

app = PlateRecognitionDeployment.bind(
    YoloDeployment.bind(settings_nn.YOLO.WEIGHTS, settings.DEVICE),
    StnDeployment.bind(settings_nn.STN.WEIGHTS, settings.DEVICE),
    LprnetDeployment.bind(settings_nn.LPRNET.WEIGHTS, settings.DEVICE),
)
