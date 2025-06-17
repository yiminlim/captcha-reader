# Captcha Reader

## Project Overview

This project is a Python-based CAPTCHA reader designed to automatically read and decode CAPTCHA images containing alphanumeric characters. The objective was to develop a solution that can process CAPTCHA images and accurately extract the embedded code using a simple yet effective approach to achieve reliable results.

## Approach and Solution

### Problem Understanding

The task was to build a system that can accurately extract and recognize the characters from CAPTCHA images, given a set of sample images and their corresponding ground truth outputs. The solution needed to process new CAPTCHA images and output the decoded string.

### Solution Overview

My approach was to use a straightforward, template-matching method for character recognition, prioritizing simplicity and reliability. The main steps of the algorithm are:

1. **Image Loading**  
   - Load the input CAPTCHA image directly as a 3D NumPy array using the Pillow library.

2. **Image Cropping**  
   - Crop the image to the bounding box containing all black pixels, isolating the CAPTCHA code from any background.

3. **Character Segmentation**  
   - Segment the cropped image into individual character images by detecting columns containing black pixels.

4. **Template Creation**  
   - For each sample image, segment and resize the characters, and map them to their ground truth labels to build a template library.

5. **Character Recognition**  
   - For each character in a new CAPTCHA image, resize and flatten the image, then compare it against all templates using correlation.
   - Select the character with the highest similarity score as the recognized character.

6. **Output Generation**  
   - Combine the recognized characters into a string and save the result to an output file.


### Rationale

The template-matching approach was chosen for its simplicity, transparency, and effectiveness given the constraints of the problem (fixed font, consistent character size, and limited character set). This method avoids the complexity of machine learning models, making it easy to understand, maintain, and audit.

### Possible Improvements

If more time and resources were available, the following enhancements could be considered:

- **Code Structure**: Refactor the codebase for better modularity, documentation, and test coverage.
- **Robust Segmentation**: Use more advanced image processing techniques (e.g., connected component analysis, morphological operations) to improve character segmentation, especially for noisy or distorted images.
- **Machine Learning Models**: Train a lightweight convolutional neural network (CNN) for character recognition, which could improve accuracy and robustness to variations in font or noise.
- **Data Augmentation**: Generate synthetic variations of the training data to make the system more resilient to different CAPTCHA styles.
- **Performance Optimization**: Parallelize processing steps and optimize I/O for faster inference on large batches of images.

## Features

- Simple and effective template-matching CAPTCHA recognition
- Modular code for easy extension and maintenance
- Outputs decoded CAPTCHA strings to a text file

## Installation

```bash
git clone https://github.com/yiminlim/captcha-reader.git
cd captcha-reader
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python src/captcha_reader/main.py
```

You will be prompted to enter the input file path, i.e. <path/to/input_file.jpg>. The decoded CAPTCHA will be saved to `result/output.txt`.

## Configuration

- Ensure sample input CAPTCHA (.jpg) images are in the `data/input/` directory.
- Ensure sample output text (.txt) files are in the `data/output/` directory.
- Update file paths in the code if your directory structure differs.


## Contact

For questions, contact project owner [Lim Yi-Min](mailto:lim_yi_min@hotmail.com).