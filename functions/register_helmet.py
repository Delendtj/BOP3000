import detectors
import timm
import cv2
import torch
from PIL import Image

# Model som brukes: https://huggingface.co/edadaltocg/resnet34_svhn

# Gjør klar modellen
model = timm.create_model("resnet34_svhn", pretrained=True)
model.eval() #Sett den fra .train() til eval() mode
model.zero_grad(set_to_none=True) # Skru av gardients for bedre memory optimization

config = timm.data.resolve_data_config(model=model)

# Transform bilde til det modellen forventer
# config er et dict som vi henter alle keywordsa som holde på settings
tr = timm.data.create_transform(**config)

# Pipline Plan:
# cropped_image > digit_splitter() > resnet34_svhn (OCR) > digit_concat > connect id to original bbox detection

# Så må vi koble alle sammen til et nummer til slutt.
# Når vi detecter må vi akseptere nummere med bare høy confidence me thinks
# Det gjør det også greit å croppe feil og for mye, så lenge
# det vi ender opp med har høy confidence.
def register_helmet(image):
    boxes = image_cropper(image)

    for i, (x, y, w, h) in enumerate(boxes):
        digit_crop = image[y:y + h, x:x + w]
        # image = cv2.cvtColor(digit_crop, cv2.COLOR_BGR2RGB)
        label, confidence = digit_ocr(digit_crop)

        print("Confidence: ", confidence)
        print("Label/Number: ", label)

        # Funker ganske dårlig ngl. Ting må tunes i image_cropper()
        # Hvis du vil se cropped bilder. uncomment
        # cv2.imwrite(f"cropped_img{i}.png", digit_crop)


def image_cropper(image):
    # Konverter til GrayScale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binarize
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contours (digit blobs)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # FILTERING DEL
    # Det er her vi må tune for å faktisk få de gode crops
    # Next er å få til verdier som er relativ til fatisk størrelsen på det
    # originale bildet. (Ikke bruke piksler)
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0

        # Exclude horizontal bars (aspect < 0.3) and tiny noise (area < 100)
        # Keep boxes that look digit-shaped
        if aspect_ratio > 0.6 and w * h > 400 and h > 20:
            digit_boxes.append((x, y, w, h))

    # Sort and take first 3
    digit_boxes.sort(key=lambda box: box[0])
    digit_boxes = digit_boxes[:3]

    return digit_boxes

def digit_ocr(img):
    # image = Image.open("digits_debug_2.jpg").convert("RGB")

    image = Image.fromarray(img)

    # Unsqueezer det siden vi har shape (3, W, H) men trenger (1, 3, W, H) der 1 er batch size
    img_normalized = tr(image).unsqueeze(0)
    print(img_normalized.shape)

    result = model(img_normalized)
    print("Result: ", result)
    print("Shape: ", result.shape)

    # Aner ikke hva denne faktisk gjør
    # Tensor stuff til probability score og labels
    probs = torch.nn.functional.softmax(result, dim=1)
    conf, pred = probs.max(dim=-1)

    label = pred.item()
    confidence = conf.item()

    return label, confidence

