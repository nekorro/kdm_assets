from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
import os
from torch import cuda
import sys

directory = sys.argv[1]
files = Path(directory).glob('*.png')
text_location_left = 0
text_location_top = 70
text_location_bottom = 160

device = "cuda:0" if cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-stage1')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-stage1')

for file in files:
    print(file)

    image = Image.open(file).convert("RGB")
    width, height = image.size
    text_location_right = width
    text_location = (text_location_left,text_location_top,text_location_right,text_location_bottom)
    image_textarea = image.crop(text_location)

    pixel_values = processor(images=image_textarea, return_tensors="pt").to(device).pixel_values
    generated_ids = model.to(device).generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    newfile = os.path.join(directory, generated_text + '.png')
    print(newfile)
    os.rename(file, newfile)