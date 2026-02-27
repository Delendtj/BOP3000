import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "False"
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

result = ocr.predict("../pictures/number20.png")

for res in result:
    res.print()
    print(res.text)