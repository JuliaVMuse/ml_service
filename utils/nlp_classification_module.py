import fuzzysearch
from PIL import Image
import numpy as np

from cv_sberocr_inference.ocr import image2multilinestext


def nlp_classification_doc(image: np.ndarray) -> str:
    """
    NLP-классификация: если признаков заявления или приложения нет, то это неизвестный документ

    :param image: image of document, np.ndarray
    :return: doc's type, str
    """

    pil_image = Image.fromarray(image)
    del image

    if pil_image.height > 1750:
        h_resize = 1750
        w_resize = int(pil_image.width / (pil_image.height / 1750))
        pil_image = pil_image.resize((w_resize, h_resize))

        del h_resize, w_resize

    try:
        arr_img = np.array(pil_image)
        arr_img = arr_img[:pil_image.height // 3]   # оставляем верхнюю треть документа
        ent_list, _ = image2multilinestext(np.array(arr_img, dtype=np.uint8))
        del arr_img
    except:
        try:
            ent_list, _ = image2multilinestext(np.array(pil_image, dtype=np.uint8))
        except:
            ent_list = [""]
    del pil_image

    text = ""
    for ent in ent_list:
        if type(ent) is list:
            for s in ent:
                text += s
                text += " "
        elif type(ent) is str:
            text += ent
            text += " "
    text = text.strip()

    doc_types = {'reminder': ['Памятка при наступлении страхового события'],
                 'statement': ['Заявление о наступлении события, имеющего признаки страхового случая',
                               'Заявитель обращается', 'Сведения о событии', 'Сведения медицинском учреждении'],
                 'application_declaration': ['Декларация заявителя (продолжение Заявления)'],
                 'application_beneficiary': ['Выгодоприобретатель (продолжение Заявления)'],
                 'application_parent': ['Согласие второго родителя (продолжение Заявления)']}

    text = text.lower()

    for key in doc_types.keys():
        for type_doc in doc_types[key]:
            # if substring not in string: len return value = 0
            find_string = fuzzysearch.find_near_matches(subsequence=type_doc.lower(),
                                                        sequence=text,
                                                        max_l_dist=len(type_doc) // 6)

            if len(find_string) > 0:
                return key

    del doc_types, text

    return 'other'
