import cv2
import torch
from PIL import Image
import pytesseract

#############################################################################
# OLD STUFF
# Model som brukes: https://huggingface.co/edadaltocg/resnet34_svhn

# Gjør klar modellen
#model = timm.create_model("resnet34_svhn", pretrained=True)
#model.eval() #Sett den fra .train() til eval() mode
#model.zero_grad(set_to_none=True) # Skru av gardients for bedre memory optimization

#config = timm.data.resolve_data_config(model=model)

# Transform bilde til det modellen forventer
# config er et dict som vi henter alle keywordsa som holde på settings
#tr = timm.data.create_transform(**config)
#############################################################################

# Threshold, vi upscaler alt under dette
UPSCALE_THRESH = 60

def register_helmet(image):
    processed_img =  preprocess_image(image)

    # psm = 7 (bilde blir håndtert som et single line tekst.
    # Whitelist: tall fra 0-9
    output = pytesseract.image_to_string(processed_img, lang="eng", config="--psm 7 -c tessedit_char_whitelist=0123456789")
    print("Tesseract output: ", output)



# Preprocessing av input
def preprocess_image(image):

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

    output = sharpened

    # Vis fram for debugging
    cv2.imshow("image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output


######[OLD]#####
# Alt under her er old, når vi prøvde digit cropping.
# Har det liggende her i tilfelle vi vil gå tilbake.
def show_cropped(image):
    boxes = image_cropper(image)

    for i, (x, y, w, h) in enumerate(boxes):
        # Hente ut cropped område fra bilde
        digit_crop = image[y:y + h, x:x + w]
        # image = cv2.cvtColor(digit_crop, cv2.COLOR_BGR2RGB)
        label, confidence = digit_ocr(digit_crop)

        print("Confidence: ", confidence)
        print("Label/Number: ", label)

        # Funker ganske dårlig ngl. Ting må tunes i image_cropper()
        # DEBUGGING: Hvis du vil se cropped bilder. uncomment
        # cv2.imwrite(f"cropped_img{i}.png", digit_crop)
        cv2.imshow("image", digit_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def image_cropper(image):
    # Konverter til GrayScale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binarize
    computed_thresh, thresh = cv2.threshold(gray, None, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                         cv2.THRESH_BINARY, 11, 2)

    # find contours (digit blobs)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # FILTERING DEL
    # Det er her vi må tune for å faktisk få de gode crops
    # Next er å få til verdier som er relativ til fatisk størrelsen på det
    # originale bildet. (Ikke bruke piksler)

    #print("thresh: ", computed_thresh)

    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0

        # Exclude horizontal bars (aspect < 0.3) and tiny noise (area < 100)
        # Keep boxes that look digit-shaped
        #if aspect_ratio > 0.6 and w * h > 400 and h > 20:
        #    digit_boxes.append((x, y, w, h))

    # Sort and take first 3
    digit_boxes.sort(key=lambda box: box[0])
    digit_boxes = digit_boxes[:3]

    return digit_boxes

def digit_ocr(img):
    # image = Image.open("digits_debug_2.jpg").convert("RGB")

    image = Image.fromarray(img)

    # Unsqueezer det siden vi har shape (3, W, H) men trenger (1, 3, W, H) der 1 er batch size
    img_normalized = tr(image).unsqueeze(0)
    print("img shape: ", img_normalized.shape)
    print("length: ", len(img_normalized))

    result = model(img_normalized)

    # Aner ikke hva denne faktisk gjør
    # Tensor stuff til probability score og labels
    probs = torch.nn.functional.softmax(result, dim=1)
    conf, pred = probs.max(dim=-1)

    label = pred.item()
    confidence = conf.item()

    return label, confidence

