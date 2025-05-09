import easyocr  # type: ignore
from PIL import Image, ImageDraw, ImageOps  # type: ignore
import os
import numpy as np

# Initialize the EasyOCR reader for English
reader = easyocr.Reader(["en"], gpu=False)

# Load the image
image_path = r"C:\python_project.py\uploaded images\im2.png"
print(f"Input image path: {image_path}")
im = Image.open(image_path)

# Convert the image to grayscale for segmentation methods
im_gray = im.convert('L')

# Threshold the image to binary (black and white)
im_bin = im_gray.point(lambda p: p > 128 and 255)

# Convert the image to a NumPy array for manipulation
im_array = np.array(im_bin)

# Horizontal Projection Method (for line segmentation)
def horizontal_projection(image_array):
    # Sum the pixel values along the horizontal axis (rows)
    projection = np.sum(image_array, axis=1)
    return projection

# Vertical Projection Method (for word segmentation)
def vertical_projection(image_array):
    # Sum the pixel values along the vertical axis (columns)
    projection = np.sum(image_array, axis=0)
    return projection

# Connected Component Approach (for word segmentation)
def connected_components(image_array):
    from scipy.ndimage import label
    # Label connected components (assumes binary image with foreground as white)
    labeled_array, num_features = label(image_array)
    return labeled_array, num_features

# Event Window Technique (simple version for segmenting words)
def event_window_technique(projection, threshold=100):
    segments = []
    segment = []
    for i, value in enumerate(projection):
        if value > threshold:
            segment.append(i)
        else:
            if segment:
                segments.append(segment)
                segment = []
    if segment:  # Add last segment if it exists
        segments.append(segment)
    return segments

# Apply Horizontal Projection
horizontal_proj = horizontal_projection(im_array)
print(f"Horizontal Projection: {horizontal_proj}")

# Apply Vertical Projection
vertical_proj = vertical_projection(im_array)
print(f"Vertical Projection: {vertical_proj}")

# Apply Connected Components
labeled_image, num_features = connected_components(im_array)
print(f"Number of connected components: {num_features}")

# Apply Event Window Technique for word segmentation based on vertical projection
word_segments = event_window_technique(vertical_proj)
print(f"Word Segments: {word_segments}")

# Perform OCR and get bounding boxes (this will be used for drawing bounding boxes)
bounds = reader.readtext(image_path)
if bounds:
    print("Detected text and bounding boxes:", bounds)
else:
    print("No text detected in the image.")

# Function to draw bounding boxes
def draw_boxes(image, bounds, color="yellow", width=8):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

# Draw bounding boxes on the image
drawn_image = draw_boxes(im, bounds)

# Ensure output directory exists
output_dir = r"C:\python_project.py\output_images"
os.makedirs(output_dir, exist_ok=True)

# Save the output image
output_image_path = os.path.join(output_dir, "output_image.jpg")
try:
    drawn_image.save(output_image_path)
    print(f"Image successfully saved to {output_image_path}")
except Exception as e:
    print(f"Error saving image: {e}")

# Save the extracted text to a file
text_output_path = os.path.join(output_dir, "output_text.txt")
try:
    with open(text_output_path, mode="w", encoding="utf-8") as f:
        for bound in bounds:
            text = bound[1]
            print(text)  # Print the text to the console
            f.write(text + "\n")
    print(f"Extracted text successfully saved to {text_output_path}")
except Exception as e:
    print(f"Error saving text: {e}")
