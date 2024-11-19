import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = "img_test.png"
image = Image.open(image_path)

grayscale_image = image.convert("L")
threshold = 128
binary_image = grayscale_image.point(lambda x: 255 if x > threshold else 0, '1')
pixel_array = np.logical_not(np.array(binary_image))

np.save("img_test", arr=pixel_array)