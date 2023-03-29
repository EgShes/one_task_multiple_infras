from pydantic import BaseModel, confloat, conint

COORDINATE = conint(ge=0)
CONFIDENCE = confloat(ge=0, le=1)


class PlatePrediction(BaseModel):
    xmin: COORDINATE
    ymin: COORDINATE
    xmax: COORDINATE
    ymax: COORDINATE
    confidence: CONFIDENCE
