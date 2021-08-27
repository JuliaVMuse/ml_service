from PIL import Image
import numpy as np

from typing import List, Dict

from models.utils import get_sub_img
from models.ocr import image2text
from models.ocr_postprocessing import postprocessing_entities
from cv_sberocr_inference.ocr import image2multilinestext


def image_ocr(doc_type: str,
              image: Image,
              field_coordinates: List[np.array],
              field_names: List[str]) -> List[Dict]:

    """
    Entities recognition

    :param doc_type: type of document
    :param image: doc's image
    :param field_coordinates: coordinates of fields
    :param field_names: names of fields
    :return: list_entities: list of entities
    """

    entities, confidences = {}, {}
    list_entities, reason_confidence = [], []
    thresh = 0
    reason_value = ""

    if "reason" in field_names:
        heights = [box[1][1] - box[2][1] for box in field_coordinates if (box[1][1] - box[2][1]) > 6]
        min_ = min(heights)
        thresh = min_ * 1.7

    for box, entity_name in zip(field_coordinates, field_names):
        if entity_name == "combined_mort_reason":
            continue

        try:
            field = get_sub_img(image, box)
            field = field.convert('RGB')
            confidence, entity_value = image2text(field)

            if doc_type == 'reason' and entity_name == 'reason':
                if field.height > thresh:
                    entity_value, confidence = image2multilinestext(np.array(field, dtype=np.uint8))
                    res = ""

                    if type(entity_value) is list:
                        for ent in entity_value:
                            res += ' '.join(ent)
                            res += " "

                    elif type(entity_value) is str:
                        res += entity_value
                        res += " "

                    entity_value = res
                    conf = []
                    if type(confidence) is list:
                        for c in confidence:
                            conf.append(np.mean(c))
                    else:
                        conf.append(np.mean(confidence))
                    confidence = np.mean(conf)
                else:
                    confidence, entity_value = image2text(field)
                reason_confidence.append(confidence)

                entity_value = entity_value.strip()
                entity_value = entity_value.replace(" 6) ", " б) ")
                if entity_value == "":
                    continue

                if "I.а)" in entity_value:
                    ind = entity_value.index('I.а)')
                    entity_value = entity_value[ind:]
                if entity_value == "":
                    continue

                reason_value += entity_value
                reason_value += " "

            if doc_type == 'cert' and entity_name == 'name':
                if len(entity_value.split()) > 1:
                    entities["patronymic"] = entity_value.split()[1]
                confidences["patronymic"] = confidence

                if len(entity_value.split()) > 0:
                    entity_value = entity_value.split()[0]

            if doc_type != 'reason' and entity_name != 'reason':
                entities[entity_name] = entity_value
                confidences[entity_name] = confidence
        except:
            continue

    if doc_type == 'reason':
        entities["combined_mort_reason"] = reason_value
        confidences["combined_mort_reason"] = np.mean(reason_confidence)

    if doc_type == 'note':
        entities = postprocessing_entities(entities)

    for entity_name in entities:
        conf = 0
        if type(confidences[entity_name]) is list:
            conf = min(confidences[entity_name])
        else:
            conf = confidences[entity_name]
        list_entities.append(
            {
                "entity_name": entity_name,
                "entity_value": entities[entity_name],
                "confidence": conf
            }
        )

    return list_entities
