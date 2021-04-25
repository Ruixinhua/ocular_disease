import logging.config
import argparse
from os import listdir, makedirs
from os.path import isfile, join

from utils.image_process import ImageCrop, ImageResizer


def crop_images(source_path, destination_path):
    """
    Crop images with according source path, and save images to destination path
    """
    files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    for file in files:
        logger.debug("Processing: " + file)
        ImageCrop(source_path, destination_path, file).remove_black_pixels()


def resize_images(source_path, destination_path):
    """
    Resize images with according source path, and save images to destination path
    """
    files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    for file in files:
        logger.debug('Processing image: ' + file)
        ImageResizer(args.image_width, args.quality, source_path, destination_path, file, args.keep_aspect_ratio).run()


if __name__ == "__main__":
    """
    """
    # set argument parser, and pass argument needed for data processing
    args = argparse.ArgumentParser(description="Data pre-processing for training and testing")
    args.add_argument("-s", "--source_path", default="dataset/ODIR-5K/ODIR-5K/Testing Images", type=str,
                      help="path to the images folder")
    args.add_argument("-d", "--destination_path", default="dataset/test_crop", type=str,
                      help="path to processed folder")
    # argument about how to resize image
    args.add_argument("-w", "--image_width", default=512, type=int, help="width of image")
    args.add_argument("-q", "--quality", default=100, type=int, help="quality of image after resize")
    args.add_argument("-k", "--keep_aspect_ratio", default=False, type=bool, help="whether keep original image ratio")
    args.add_argument("-t", "--task", default="crop", type=str, help="specific task apply on the images")
    args = args.parse_args()
    # create logger
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filename="saved/log/data_process.log")
    logger = logging.getLogger("crop images")
    # make sure the destination folder exist
    makedirs(args.destination_path, exist_ok=True)
    if "crop" in args.task:
        crop_images(args.source_path, args.destination_path)
    if "resize" in args.task:
        resize_images(args.source_path, args.destination_path)
