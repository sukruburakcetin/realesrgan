import sys
import os
import cv2
import numpy as np
import warnings
from subprocess import call

from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

global down_width
global down_height


def load_images_from_folder(folder):
    images = []
    image_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            image_names.append(filename)
    return images, image_names


def sharpen_chips(OUTPUT_DIR, SHARPENED_DIR):
    print("sharpening is started")
    images, image_names = load_images_from_folder(OUTPUT_DIR)
    for x in tqdm(range(0, len(images))):
        img_sharpened = cv2.filter2D(images[x], -1, kernel)
        cv2.imwrite(SHARPENED_DIR + "\\" + str(
            os.path.splitext(image_names[x])[0] + os.path.splitext(image_names[x])[1]), img_sharpened)

    images.clear()
    image_names.clear()
    print("sharpening is done")
    print("---------------------------")


def resize_chips(SHARPENED_DIR, RESIZED_DIR, width=512, height=512):
    print("rescaling is started")
    images, image_names = load_images_from_folder(SHARPENED_DIR)

    # let's downscale the image using new  width and height
    down_width = width
    down_height = height

    down_points = (down_width, down_height)
    for x in tqdm(range(0, len(images))):
        resized_down = cv2.resize(images[x], down_points, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(RESIZED_DIR + "\\" + str(
            os.path.splitext(image_names[x])[0] + os.path.splitext(image_names[x])[1]), resized_down)

    images.clear()
    image_names.clear()
    print("rescaling is done")
    print("---------------------------")


def convert_chips(RESIZED_DIR, CONVERTED_DIR):
    print("converting is started")
    filelist = []
    for file in os.listdir(RESIZED_DIR):
        if file.endswith(".png"):
            filelist.append(
                os.path.join(RESIZED_DIR, file))

    for i in tqdm(range(0, len(filelist))):
        img = Image.open(filelist[i])
        img.save(CONVERTED_DIR + "\\" +
                 os.path.splitext(os.path.basename(filelist[i]))[0] + ".tif")  # or 'test.tif'

    print("converting is done")


def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def inference(INPUT_DIR, OUTPUT_DIR, mode, anti_alias):
    filelist = []
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".png"):
            filelist.append(os.path.join(INPUT_DIR, file))
    print("Input images are being uploaded for processing...")
    if anti_alias.lower() == "y":
        for i in tqdm(range(0, len(filelist))):
            img = Image.open(filelist[i])
            basewidth = 256
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))

            # for pillow versions lower than Pillow 10.0.0(cpu pytorch) use ANTIALIAS:
            # img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            # for pillow versions higher than Pillow 10.0.0(gpu pytorch(overrides)) use Resampling.LANCZOS:
            img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
            img.save(INPUT_DIR + "\\" + os.path.basename(filelist[i]))
    else:
        pass
    if mode == "base":
        run_cmd("python inference_realesrgan.py -n RealESRGAN_x4plus -i " + INPUT_DIR + " -o " + OUTPUT_DIR)

    else:
        os.system("python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i " + INPUT_DIR + " -o " + OUTPUT_DIR)


def validate_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist.")
            return False
    return True


def validate_mode(mode):
    if mode not in ["base", "anime"]:
        print("Error: Invalid mode. Please enter 'base' or 'anime'.")
        return False
    return True


def main():
    mode = input("Enter the mode (base or anime): ").strip()
    anti_alias = input("Do you want to apply anti aliasing? (Y/N): ").strip()

    if not validate_mode(mode):
        return

    image_path = input("Enter the path to the input images: ").strip()
    enhanced_path = input("Enter the path to the output(real-esrgan enhanced) images: ").strip()

    if mode == "base":
        sharpened_path = input("Enter the path to the sharpened images (optional, press Enter to skip): ").strip()
        resized_path = input("Enter the path to the resized images (optional, press Enter to skip): ").strip()
        resize_width = int(input("Enter the resize width: ").strip())
        resize_height = int(input("Enter the resize height: ").strip())
        converted_path = input("Enter the path to the converted images (optional, press Enter to skip): ").strip()

        paths = [image_path, enhanced_path]
        if sharpened_path:
            paths.append(sharpened_path)
        if resized_path:
            paths.append(resized_path)
        if converted_path:
            paths.append(converted_path)
    else:
        sharpened_path = input("Enter the path to the sharpened images (optional, press Enter to skip): ").strip()
        resized_path = input("Enter the path to the resized images (optional, press Enter to skip): ").strip()
        resize_width = int(input("Enter the resize width: ").strip())
        resize_height = int(input("Enter the resize height: ").strip())
        converted_path = input("Enter the path to the converted images (optional, press Enter to skip): ").strip()

        paths = [image_path, enhanced_path]
        if sharpened_path:
            paths.append(sharpened_path)
        if resized_path:
            paths.append(resized_path)
        if converted_path:
            paths.append(converted_path)

    if not all(paths):
        print("Error: Paths cannot be empty.")
        return

    if not validate_paths(paths):
        return

    if len(set(paths)) != len(paths):
        print("Error: Input and output paths must be unique.")
        return

    print("Initializing...")
    inference(image_path, enhanced_path, mode, anti_alias)
    if mode == "base":
        if sharpened_path and resized_path and converted_path:
            sharpen_chips(enhanced_path, sharpened_path)
            resize_chips(sharpened_path, resized_path, resize_width, resize_height)
            convert_chips(resized_path, converted_path)
        elif sharpened_path and resized_path:
            sharpen_chips(enhanced_path, sharpened_path)
            resize_chips(sharpened_path, resized_path, resize_width, resize_height)
        elif resized_path and converted_path:
            resize_chips(enhanced_path, resized_path, resize_width, resize_height)
            convert_chips(resized_path, converted_path)
        elif sharpened_path and converted_path:
            sharpen_chips(enhanced_path, sharpened_path)
            convert_chips(sharpened_path, converted_path)
        elif sharpened_path:
            sharpen_chips(enhanced_path, sharpened_path)
        elif resized_path:
            resize_chips(enhanced_path, resized_path, resize_width, resize_height)
        elif converted_path:
            convert_chips(resized_path, converted_path)
    else:
        if sharpened_path and resized_path and converted_path:
            sharpen_chips(enhanced_path, sharpened_path)
            resize_chips(sharpened_path, resized_path, resize_width, resize_height)
            convert_chips(resized_path, converted_path)
        elif sharpened_path and resized_path:
            sharpen_chips(enhanced_path, sharpened_path)
            resize_chips(sharpened_path, resized_path, resize_width, resize_height)
        elif resized_path and converted_path:
            resize_chips(enhanced_path, resized_path, resize_width, resize_height)
            convert_chips(resized_path, converted_path)
        elif sharpened_path and converted_path:
            sharpen_chips(enhanced_path, sharpened_path)
            convert_chips(sharpened_path, converted_path)
        elif sharpened_path:
            sharpen_chips(enhanced_path, sharpened_path)
        elif resized_path:
            resize_chips(enhanced_path, resized_path, resize_width, resize_height)
        elif converted_path:
            convert_chips(resized_path, converted_path)

    if len(set(paths)) != len(paths):
        print("Error: Input and output paths must be unique.")
        return

    print("Given paths:")
    for p in paths:
        if p == image_path:
            print("Input images path:", p)
        elif p == enhanced_path:
            print("Output(real-esrgan enhanced) images path:", p)
        elif p == sharpened_path:
            print("Sharpened images path:", p)
        elif p == resized_path:
            print("Resized images path:", p)
        elif p == converted_path:
            print("Converted images path:", p)


if __name__ == "__main__":
    main()
