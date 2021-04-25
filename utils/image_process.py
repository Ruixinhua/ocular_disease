import cv2
import numpy as np
import os
import logging
import PIL
from PIL import Image
"""
The class ImageCrop and ImageResize reference from 
https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning
"""


class ImageCrop:
    """
    Image crop only keep coloured pixels.
    """
    def __init__(self, source_folder, destination_folder, file_name):
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.file_name = file_name

    def remove_black_pixels(self):
        # read the image file
        file = os.path.join(self.source_folder, self.file_name)
        image = cv2.imread(file)

        # Mask of coloured pixels.
        mask = image > 0

        # Coordinates of coloured pixels.
        coordinates = np.argwhere(mask)

        # Binding box of non-black pixels.
        x0, y0, s0 = coordinates.min(axis=0)
        x1, y1, s1 = coordinates.max(axis=0) + 1  # slices are exclusive at the top

        # Get the contents of the bounding box.
        cropped = image[x0:x1, y0:y1]
        # overwrite the same file
        file_cropped = os.path.join(self.destination_folder, self.file_name)
        cv2.imwrite(file_cropped, cropped)


class ImageResizer:
    """
    This class allows you to resize and mirror an image of the dataset according to specific rules
    """
    def __init__(self, image_width, quality, source_folder, destination_folder, file_name, keep_aspect_ratio):
        self.logger = logging.getLogger("resize")
        self.image_width = image_width
        self.quality = quality
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.file_name = file_name
        self.keep_aspect_ration = keep_aspect_ratio

    def run(self):
        """ Runs the image library using the constructor arguments.
        Args:
          No arguments are required.
        Returns:
          Saves the treated image into a separate folder.
        """
        # We load the original file, we resize it to a smaller width and correspondent height and
        # also mirror the image when we find a right eye image so they are all left eyes

        file = os.path.join(self.source_folder, self.file_name)
        img = Image.open(file)
        if self.keep_aspect_ration:
            # it will have the exact same width-to-height ratio as the original photo
            width_percentage = (self.image_width / float(img.size[0]))
            height_size = int((float(img.size[1]) * float(width_percentage)))
            img = img.resize((self.image_width, height_size), PIL.Image.ANTIALIAS)
        else:
            # This will force the image to be square
            img = img.resize((self.image_width, self.image_width), PIL.Image.ANTIALIAS)
        if "right" in self.file_name:
            self.logger.debug("Right eye image found. Flipping it")
            img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(self.destination_folder, self.file_name),
                                                      optimize=True, quality=self.quality)
        else:
            img.save(os.path.join(self.destination_folder, self.file_name), optimize=True, quality=self.quality)
        self.logger.debug("Image saved")
