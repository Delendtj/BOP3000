import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import cv2

# Notes:
# We can not cache since images will be different each time.
#

# Model: Qwen3-VL-2B-Instruct
# Using small model since we are simply doing OCR.
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

img = cv2.imread("164.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

instruction_short = "digits only"
instruction_long = "Extract the number from this image. If the number is unidentifiable return 'unknown' else, only respond with number and confidence score."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": instruction_long},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  # OCR = short output
        do_sample=False  # stability/controls randomness
    )

prompt_len = inputs["input_ids"].shape[-1]

result = processor.decode(
    outputs[0][prompt_len:],
    skip_special_tokens=True
).strip()

# result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(result)