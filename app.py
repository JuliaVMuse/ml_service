import os
import json
import time
from aiohttp import web, web_response

from models.classify_inference import classification_doc
from aiohttp_service_builder import response_error, build_aiohttp_service_app
from utils import preprocessing_module, ocr_module, nlp_classification_module, object_detection_module
from models import get_fields, get_reason_crop


os.environ['OMP_THREAD_LIMIT'] = '1'


def logic_for_service_app(request_obj, response_obj, array_images):
    doc_type, img_arr, doc_id, entities = None, None, None, None
    LIST_SEGMENTATION_MODELS = ['cert', 'note']
    response_data = {}

    if "request_data" in request_obj:
        doc_type = request_obj['request_data'].get('doc_type')
        doc_id = request_obj['request_data'].get('doc_id')
        if doc_id is not None:
            response_data["doc_id"] = doc_id

    if doc_type is None:
        #  CV-Классификация
        try:
            start = time.time()
            doc_type = classification_doc(array_images[0])
            print("Время CV-Классификации: {0:.2f}".format(time.time() - start))

        except Exception as e:
            return response_error({"ErrorCode": 3, "ErrorMessage": f"Ошибка CV-Классификации {doc_type}, {e}"})

    if doc_type == 'other':
        #  NLP-классификация
        try:
            start = time.time()
            doc_type = nlp_classification_module.nlp_classification_doc(array_images[0])
            print("Время NLP-Классификации: {0:.2f}".format(time.time() - start))
        except Exception as e:
            return response_error({"ErrorCode": 404, "ErrorMessage": f"Ошибка NLP-Классификации {doc_type}, {e}"})

    response_data["doc_name"] = doc_type
    reason_crop, det_coordinate_fields_, det_name_fields_ = None, None, None

    if doc_type in LIST_SEGMENTATION_MODELS:
        try:
            start = time.time()
            #  Preprocessing
            img_arr = preprocessing_module.preprocessing_image(array_images, doc_type)
            print("Время Препроцессинга: {0:.2f}".format(time.time() - start))

        except Exception as e:
            return response_error({"ErrorCode": 5, "ErrorMessage": f"Ошибка Препроцессинга {doc_type}, {e}"})

        try:
            start = time.time()
            #  Извлечение сущностей из свидетельтв о смерти и справок о смерти
            det_coordinate_fields, det_name_fields = get_fields(img_arr, doc_type)
            if doc_type == "note" and "combined_mort_reason" in det_name_fields:
                reason_crop = get_reason_crop(det_coordinate_fields, det_name_fields, img_arr)

                #  Извлечение сущностей из поля причина смерти справок о смерти
                doc_type = "reason"
                det_coordinate_fields_, det_name_fields_ = get_fields(reason_crop, doc_type)
                doc_type = "note"
            print("Время извлечения сущностей: {0:.2f}".format(time.time() - start))
        except Exception as e:
            return response_error({"ErrorCode": 403, "ErrorMessage": f"Ошибка сегментации {doc_type}, {e}"})

        try:
            start = time.time()
            entities = ocr_module.image_ocr(doc_type, img_arr, det_coordinate_fields, det_name_fields)
            if doc_type == "note":
                doc_type = "reason"
                entities_ = ocr_module.image_ocr(doc_type, reason_crop, det_coordinate_fields_, det_name_fields_)
                entities.extend(entities_)
                doc_type = "note"
            print("Время распознавания: {0:.2f}".format(time.time() - start))
        except Exception as e:
            return response_error({"ErrorCode": 405, "ErrorMessage": f"Ошибка OCR {doc_type}, {e}"})

    if entities is None:
        entities = []
        doc_type = "other"
    elif len(entities) == 0:
        doc_type = "other"

    response_data["entities"] = entities
    response_data["doc_name"] = doc_type

    try:
        start = time.time()
        entities_od = object_detection_module.object_detection_image(array_images)

        if entities is None:
            entities = entities_od
        else:
            entities.extend(entities_od)
        del entities_od

        response_data["entities"] = entities

        print("Время object detection: {0:.2f}".format(time.time() - start))
    except Exception as e:
        return response_error({"ErrorCode": 406, "ErrorMessage": f"Ошибка object detection {doc_type}, {e}"})

    response_obj['response_data'] = response_data

    # Основной вывод объекта с ответом, если все удачно :)
    if type(response_obj) == web_response.Response:
        return response_obj
    else:
        # Собираем ответ и отправляем пользователю
        try:
            # сериализуем обьект ответа в json
            response_str = json.dumps(response_obj, ensure_ascii=False)
            return web.Response(text=response_str, content_type='application/json')

        except Exception as e:
            return response_error({"ErrorCode": 999, "ErrorMessage": f"Не удалось сериализовать ответ {doc_type}, {e}"})


# Запуск сервиса aiohttp на порту 8080
if __name__ == '__main__':
    build_aiohttp_service_app(logic_for_service_app)
