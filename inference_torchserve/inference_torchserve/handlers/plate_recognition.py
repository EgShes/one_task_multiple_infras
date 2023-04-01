import json


def aggregate(data, context):
    if data:
        yolo_result = json.loads(data[0].get("yolo"))
        lprnet_result = json.loads(data[0].get("lprnet"))
        response = {
            "coordinates": yolo_result["coordinates"],
            "texts": lprnet_result["data"],
        }
        return [response]
