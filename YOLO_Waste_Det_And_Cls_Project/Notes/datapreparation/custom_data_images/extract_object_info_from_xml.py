import os
from glob import glob
import pandas as pd
from xml.etree import ElementTree as ET
from pathlib import Path
from shutil import move
import shutil

# Function to replace backslashes with forward slashes
replace_slashes = lambda x: str(x).replace('\\', '/')

# Step-1: Get paths of each XML file
xml_dir = Path('../datapreparation/data_images')
xml_files = glob(str(xml_dir / '*.xml'))
xml_files = list(map(replace_slashes, xml_files))

# Step-2: Function to extract data from XML files
def extract_text(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    image_name = root.find('filename').text      # extract filename
    width = int(root.find('size').find('width').text)  # for img width
    height = int(root.find('size').find('height').text)   # for img height
    parser = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])

    return parser

# From all XML files and extract data
parser_all = [extract_text(file) for file in xml_files]

# Flatten the list of lists
data = [item for sublist in parser_all for item in sublist]

# Create a DataFrame from extracted data
columns = ['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax']
dtf = pd.DataFrame(data, columns=columns)

# Convert columns to integer type
cols_to_convert = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
dtf[cols_to_convert] = dtf[cols_to_convert].astype(int)

# Calculate center x, center y, width (w), and height (h)
dtf['center_x'] = ((dtf['xmax'] + dtf['xmin']) / 2) / dtf['width']
dtf['center_y'] = ((dtf['ymax'] + dtf['ymin']) / 2) / dtf['height']
dtf['w'] = (dtf['xmax'] - dtf['xmin']) / dtf['width']
dtf['h'] = (dtf['ymax'] - dtf['ymin']) / dtf['height']

# Display a sample of processed data
print(dtf.head())

# Split data into train and test sets
images = dtf['filename'].unique()
train = list(images[:int(len(images) * 0.75)])
test = list(images[int(len(images) * 0.75):])

train_dtf = dtf[dtf['filename'].isin(train)]
test_dtf = dtf[dtf['filename'].isin(test)]

# Assign ID number according to the object names
# Label encoding
labels = {'battery': 0, 'biological': 1, 'cardboard': 2, 'glass': 3, 'metal': 4, 'paper': 5, 'plastic': 6, 'trash': 7}
train_dtf['id'] = train_dtf['name'].apply(lambda x: labels[x])
test_dtf['id'] = test_dtf['name'].apply(lambda x: labels[x])

# Create train and test folders
train_folder_images = 'data_images/train/images'
train_folder_labels = 'data_images/train/labels'
test_folder_images = 'data_images/test/images'
test_folder_labels = 'data_images/test/labels'

os.makedirs(train_folder_images, exist_ok=True)
os.makedirs(train_folder_labels, exist_ok=True)
os.makedirs(test_folder_images, exist_ok=True)
os.makedirs(test_folder_labels, exist_ok=True)

# Function to save image and labels in respective folders
def save_data(filename, folder_path_images, folder_path_labels, dtf):
    # Move image to the destination folder
    src = os.path.join('data_images', filename)
    dst_image = os.path.join(folder_path_images, os.path.basename(filename))
    move(src, dst_image)

    # Save the labels with .txt files
    text_filename = os.path.join(folder_path_labels, os.path.splitext(os.path.basename(filename))[0] + '.txt')
    data_subset = dtf[dtf['filename'] == filename]
    
    with open(text_filename, 'w') as f:
        for index, row in data_subset.iterrows():
            class_id = row['id']
            center_x = row['center_x']
            center_y = row['center_y']
            width = row['w']
            height = row['h']
            f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

# Save data for train and test sets
for filename in train_dtf['filename'].unique():
    save_data(filename, train_folder_images, train_folder_labels, train_dtf)

for filename in test_dtf['filename'].unique():
    save_data(filename, test_folder_images, test_folder_labels, test_dtf)

# Move all .xml files to the annotations folder
annotations_folder = './data_images/annotation'
os.makedirs(annotations_folder, exist_ok=True)

for xml_file in xml_files:
    shutil.move(xml_file, os.path.join(annotations_folder, os.path.basename(xml_file)))

# Display confirmation messages
print(f"Train data successfully created in {train_folder_images} and {train_folder_labels}")
print(f"Test data successfully created in {test_folder_images} and {test_folder_labels}")
print(f"All XML files successfully moved to the annotations folder.")
