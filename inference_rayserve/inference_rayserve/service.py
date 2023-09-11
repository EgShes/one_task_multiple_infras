from inference_rayserve.models import LprnetDeployment, StnDeployment, YoloDeployment
from inference_rayserve.plate_recognition import PlateRecognitionDeployment
from nn.settings import settings as settings_nn

app = PlateRecognitionDeployment.bind(
    YoloDeployment.bind(settings_nn.YOLO.WEIGHTS),
    StnDeployment.bind(settings_nn.STN.WEIGHTS),
    LprnetDeployment.bind(settings_nn.LPRNET.WEIGHTS),
)
