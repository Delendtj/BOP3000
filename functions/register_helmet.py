import cv2
import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

from paddleocr import PaddleOCR

# Threshold, vi upscaler alt under dette
UPSCALE_THRESH = 60

_ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def register_helmet(helmets, debug=False):
    """
    Process a list of helmet dicts (from BBExtractor) and return OCR results.

    Each dict in `helmets` must have keys: 'image', 'bbox', 'conf'.
    Returns a list of dicts with keys: 'bbox', 'helmet_number', 'ocr_conf'.
    """
    results = []

    for helmet in helmets:
        img   = helmet['image']   # numpy array (BGR crop)
        bbox  = helmet['bbox']
        conf  = helmet['conf']
        tid = helmet['track_id']

        processed_img = preprocess_image(img, debug=debug)

        if debug:
            print("Running OCR...")
            print("Input shape: ", img.shape)
            print("Input shape after processing: ", img.shape)

        raw = _ocr.predict(processed_img)

        # Collect recognized text and confidence, then keep digits only.
        number_str = ""
        ocr_conf   = 0.0
        valid_texts = []
        valid_confs = []

        for res in raw:
            if isinstance(res, dict):
                rec_texts = res.get("rec_texts") or []
                rec_scores = res.get("rec_scores") or []
            else:
                rec_texts = getattr(res, "rec_texts", []) or []
                rec_scores = getattr(res, "rec_scores", []) or []

            for text, score in zip(rec_texts, rec_scores):
                text = str(text).strip()
                if not text:
                    continue
                # Filter out non digit characters
                digits = "".join(ch for ch in text if ch.isdigit())
                if not digits:
                    continue
                valid_texts.append(digits)
                valid_confs.append(float(score))



        if valid_texts:
            number_str = "".join(valid_texts).strip()
            ocr_conf = (sum(valid_confs) / len(valid_confs)) * 100.0
            # Show image if it valid
            if debug:
                print("Number accepted was: ", number_str, " for track_id: ", tid)
                cv2.imshow("Valid Image", processed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if debug:
            print(f"PaddleOCR raw text: {number_str!r}  conf: {ocr_conf:.1f}%")

        results.append({
            'bbox':          bbox,
            'helmet_number': number_str,
            'ocr_conf':      ocr_conf,
            'track_id': tid
        })

    return results

# Preprocessing av input
def preprocess_image(image, debug=False):

    # Konverter til GrayScale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Prøv simple upscaling hvis bildet er lite.
    if image.shape[0] < UPSCALE_THRESH or image.shape[1] < UPSCALE_THRESH:
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


    # Lowkey kjønner ikke mye av dette
    # Men slik jeg tolket det så lager det en blurred bildet (gaussian) som blurrer hovedsakelig noise
    # Og så minuser vi det bort ved hjelp av addWeighted()  fra original bildet (gray)
    # gray (2.0) - gaussian blur (1.0) = sharpened bilde
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)

    # Hvis den fortsatt ikke funker kan man binarize bilde slik at vi får higlighted edges bedre
    # Her funket det bra noen ganger og noen ganger ikke så lar dette stå kommentert ut.
    # computed_thresh, th1 = cv2.threshold(gray, None, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #_, output = cv2.threshold(sharpened, None, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    output = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # Vis fram for debugging
    #if debug:
    #    cv2.imshow("image", output)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    return output
