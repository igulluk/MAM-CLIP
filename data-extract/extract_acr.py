import fitz
import io
import os
import pytesseract
import numpy as np
from PIL import Image
import json
import argparse


def extract(args):
    file = args.pdf_file_path
    pdf_file = fitz.open(file)
    img_main_path = args.img_main_path

    image_caption_dict = []

    for page_index in range(len(pdf_file)):

        if page_index < 35:
            continue

        page = pdf_file[page_index]
        image_list = page.get_images()

        for image_index, img in enumerate(image_list, start=1):

            xref = img[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            image_name = f"page_{page_index}_img_{image_index}.jpg"

            captions = pytesseract.image_to_string(image)
            if os.path.isfile(os.path.join(img_main_path, image_name)):
                continue

            if not ("~" in captions or "-" in captions):
                continue
            if "~" in captions:
                start_idx = captions.find("~")
            else:
                start_idx = captions.find("-")
            captions = captions[start_idx + 1 :]
            captions = captions.replace("\n", " ")

            image = np.array(image)
            min_y = None
            for iy in range(image.shape[0] - 1):
                if np.sum(255 - image[iy, :, :]) > 20 * np.sum(
                    255 - image[iy + 1, :, :]
                ):
                    min_y = iy
                    break
            if min_y is None:
                continue
            image = image[: min_y + 1, :, :]
            image = Image.fromarray(np.uint8(image))

            os.makedirs(img_main_path, exist_ok=True)
            image.save(os.path.join(img_main_path, image_name))

            new_dict = {"image": image_name, "caption": captions}
            image_caption_dict.append(new_dict)

    with open(args.json_file_path, "w") as json_file:
        json.dump(image_caption_dict, json_file)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pdf_file_path", type=str, help="Path to the PDF of the book")
    parser.add_argument("--json_file_path", type=str, help="Path to the generated JSON file")
    parser.add_argument("--img_main_path", type=str, help="Path to the target image directory")
    args = parser.parse_args()

    extract(args)


if __name__ == "__main__":
    main()
