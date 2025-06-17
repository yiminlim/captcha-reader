import numpy as np
from PIL import Image
import cv2 

class Captcha(object):
    def __init__(self):
        pass

    def load_image_as_array(self, image_path):
        """
        Loads an image file directly into a 3D NumPy array.

        This function replaces the need to first save an image to a custom 
        text file and then parse it back. It's more direct and efficient.

        Args:
            image_path (str): The path to the image file (e.g., .jpg, .png).

        Returns:
            np.array: A 3D NumPy array representing the image (height, width, RGB),
                    or None if the image cannot be opened.
        """
        try:
            # 1. Load the image using Pillow 
            img = Image.open(image_path).convert("RGB")
            
            # 2. Convert the Pillow Image object directly into a NumPy array
            image_array = np.array(img, dtype=np.uint8)
            
            return image_array

        except FileNotFoundError:
            print(f"Error: The file could not be found at: {image_path}")
            return None
        except UnidentifiedImageError:
            print(f"Error: Cannot identify image file at: {image_path}. It might be corrupt.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    # Function to load image and convert to vector file in the sample input format
    def load_data(self, image_path):
        # Load image and ensure RGB mode
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        pixels = list(img.getdata())

        # Convert flat list to rows
        rows = [pixels[i * width : (i + 1) * width] for i in range(height)]
        image_array_path = image_path.replace(".jpg", ".txt")
        with open(image_array_path, "w") as f:
            # Write dimensions: height first, then width
            f.write(f"{height} {width}\n")
            for row in rows:
                # Join each pixel as R,G,B
                line = " ".join([f"{r},{g},{b}" for (r, g, b) in row])
                f.write(line + "\n")
        return image_array_path
    
    # Function to parse vector file in the sample input format and return a 3D numpy array
    def parse_vector_file(self, image_path):
        with open(image_path, 'r') as f:
            lines = f.readlines()
        height, width = map(int, lines[0].strip().split())
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        for row_idx, line in enumerate(lines[1:]):
            pixels = line.strip().split()
            for col_idx, pixel in enumerate(pixels):
                r, g, b = map(int, pixel.split(','))
                image_array[row_idx, col_idx] = [r, g, b]
        return image_array
        
    # Function to crop the image to the bounding box of black pixels, output 3D numpy array of the cropped captcha code 
    def crop_to_black_code(self, image_array, black_thresh=0):
        """
        Finds the smallest bounding box containing all black pixels (code)
        and returns the cropped image array.
        
        Args:
            image_array: 3D numpy array (height, width, 3) from vector file
            black_thresh: Maximum RGB value to consider as "black" (0-255)
            
        Returns:
            Cropped 3D numpy array containing only the code region
        """
        # Create mask where all RGB channels are <= threshold (black pixels)
        mask = np.all(image_array <= black_thresh, axis=2)
        
        # Get coordinates of all black pixels
        coords = np.argwhere(mask)
        
        if coords.size == 0:
            return image_array  # No black pixels found
        
        # Find bounding box boundaries
        y_min, x_min = coords.min(axis=0)  # Top-left corner
        y_max, x_max = coords.max(axis=0)  # Bottom-right corner
        
        # Add 1 to include last pixel
        cropped = image_array[y_min:y_max+1, x_min:x_max+1]
        
        return cropped
    
    # Function to segment the CAPTCHA image into 5 individual characters, output list of 3D numpy arrays for each characterS
    def segment_characters(self, image_array, num_chars=5, black_thresh=30):
        """
        Segments CAPTCHA image into individual characters using black pixel boundaries.
        
        Args:
            image_array: 3D numpy array (height, width, 3) in RGB format
            num_chars: Number of characters to extract (default 5)
            black_thresh: Maximum RGB value to consider as black (0-255)
            
        Returns:
            List of 3D numpy arrays for each character
        """
        height, width, _ = image_array.shape
        segments = []
        
        # Create mask where True = black pixel (all channels <= black_thresh)
        black_mask = np.all(image_array <= black_thresh, axis=2)
        
        x_start = 0
        
        for char_num in range(num_chars):
            # Find left boundary (x1)
            x1 = None
            for x in range(x_start, width):
                if np.any(black_mask[:, x]):
                    x1 = x
                    break
            if x1 is None:
                break  # No more characters found
            
            # Find right boundary (x2)
            x2 = x1
            for x in range(x1, width):
                if not np.any(black_mask[:, x]):
                    break
                x2 = x
            
            # Extract character (include all rows)
            char_img = image_array[:, x1:x2+1, :]
            segments.append(char_img)
            
            # Set next search start after current character
            x_start = x2 + 1
        
        return segments

    def resize_image(self, image_array, new_size=(10, 10)):
        return cv2.resize(image_array, new_size, interpolation=cv2.INTER_AREA)
        
    def save_image_array_as_jpg(self, image_array, filename):
        """
        Saves a 3D numpy array (height, width, 3) as a JPG image.
        
        Args:
            image_array: numpy array of shape (height, width, 3), dtype=uint8
            filename: output file path, e.g., 'output.jpg'
        """
        img = Image.fromarray(image_array)
        img.save(filename)
        return f"Image saved as {filename}"

    # Function to read text from a file and return it as a string
    def read_text_from_file(self, file_path):
        """
        Opens a text file, reads its content, and returns it as a string.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The content of the file as a string, or None if an error occurs.
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                return content
        except FileNotFoundError:
            #print(f"Error: The file at {file_path} was not found.")
            return None
        except Exception as e:
            #print(f"An error occurred: {e}")
            return None
    
    def get_chars_from_image(self, image_path):
        """
        Reads an image file, crops it to the bounding box of black pixels, segments it into characters,
        and returns a list of character images as 3D numpy arrays.
        
        Args:
            image_path (str): Path to the input image file.
        
        Returns:
            List of 3D numpy arrays representing each character image.
        """
        input_array = self.load_image_as_array(image_path)
        cropped_input = self.crop_to_black_code(input_array)
        input_characters = self.segment_characters(cropped_input, num_chars=5)
        return input_characters


    # Function to create templates of each character given sample images and the ground truth outputs
    def create_character_templates(self):
        """
        Creates a template library of characters from a given path.
        
        """
        # Initialise input and output paths for sample files
        input_path = "data/input/"
        output_path = "data/output/"

        # Initialize an empty dictionary to hold character templates
        char_templates = {}    
        #Enumerate through each sample input and output file
        for i in range(0, 26):
            if i < 10:
                num = "0" + str(i)
            else:
                num = str(i)
            # Construct the input file path
            input_file = input_path + "input" + num + ".jpg"
            
            # Construct the output file path
            output_file = output_path + "output" + num + ".txt"
            
            # Read the input vector file to get the image array, then crop it to the bounding box of black pixels, then segment it into characters
            input_characters = self.get_chars_from_image(input_file)
            for i, char_img in enumerate(input_characters):
                # Resize each character image to a fixed size (e.g., 10x10)
                input_characters[i] = self.resize_image(char_img)
            
            # Read the ground truth string output from the output file
            output_string = self.read_text_from_file(output_file)

            # Map each output character to its image array and save the output
            for i, char_img in enumerate(input_characters):
                if output_string is not None:
                    char = output_string[i]
                    char_templates.setdefault(char, [])
                    char_templates[char].append(char_img)

        return char_templates
        pass

    def recognize_character(self, char_features, templates, possible_chars):
        """Recognize character using template matching"""
        best_match_char = None
        overall_best_score = -1  # The highest score found across all possible characters
        
        resized_char_features = self.resize_image(char_features)
        flat_char_features = resized_char_features.flatten()

        # Loop through each possible character
        for char in possible_chars:
            if char in templates:
                # Best score for the current character after checking all of its available templates
                best_score_for_this_char = -1

                # Iterate through each template image for the current character
                for template_img in templates[char]:
                    # Flatten the current template image to a 1D vector
                    flat_template_features = template_img.flatten()

                    # Compare the input character with this one specific template
                    score = np.corrcoef(flat_char_features, flat_template_features)[0, 1]
                
                    # If this template is a better match, update the best score for this character
                    if score > best_score_for_this_char:
                        best_score_for_this_char = score

                # After checking all templates for 'char', see if it's the best overall match so far
                if best_score_for_this_char > overall_best_score:
                    overall_best_score = best_score_for_this_char
                    best_match_char = char
        
        return best_match_char, overall_best_score

    def read_captcha(self, im_path, save_path, char_templates):
        # Initialise output string
        output_string = ""
        # Initialize possible characters
        possible_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        # Get input characters from the image
        input_characters = self.get_chars_from_image(im_path)
        # For each character in the new input image, compare it to the character templates and find the best match
        for char_img in input_characters:
            # Resize the character image to a fixed size (e.g., 10x10)
            resized_char_img = self.resize_image(char_img)
            # Recognise the character using the templates
            result, _ = self.recognize_character(resized_char_img, char_templates, possible_chars)
            # Append the recognised character to the output string
            output_string += result

        # Write the output string to the save path
        with open(save_path, "w") as f:
            f.write(output_string+"\n")

    # Main function to run the algorithm for inference
    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        # Create character templates from the sample input and output files 
        char_templates = self.create_character_templates()
        # print(f"Character templates created: {len(char_templates)} unique characters")
        # for char, imgs in char_templates.items():
        #     print(f"Character '{char}': {len(imgs)} images")

        
        # Read the input image and extract the CATPCHA code by comparing against character templates, then save the output to a file
        self.read_captcha(im_path, save_path, char_templates)

        pass