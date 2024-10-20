import fitz
import io
from PIL import Image
import os
import json
import argparse


def extract_caption(txt):
    if "Figure" in txt and ("IMPRESSION" in txt or "MAMMOGRAPHY" in txt):
        occurrences = []
        word_to_find = "Figure"
        index = txt.find(word_to_find)

        while index != -1:
            occurrences.append(index)
            index = txt.find(word_to_find, index + 1)
        occurrences.append(len(txt))
        captions = []
        while len(occurrences) >= 2:
            fig_idx = occurrences[0]

            if "NOTE" in txt[fig_idx:]:
                word = "NOTE"
            elif "HISTOPATHOLOGY" in txt[fig_idx:]:
                word = "HISTOPATHOLOGY"
            elif "IMPRESSION" in txt[fig_idx:]:
                word = "IMPRESSION"
            else:
                word = "MAMMOGRAPHY"

            next_idx = txt.find(word, fig_idx + 1)
            if next_idx > occurrences[1]:
                occurrences = occurrences[1:]
                continue
            next_idx = txt.find(".", next_idx + 1)

            hist_idx = txt.find("HISTORY", fig_idx + 1)
            if hist_idx == -1 or hist_idx > occurrences[1]:
                hist_idx = 0
            caption = txt[hist_idx:next_idx]
            caption = caption.strip().replace("\n", " ")
            caption = caption.replace("- ", "")
            caption = caption.replace("\ufb01", "fi")
            captions.append(caption)
            occurrences = occurrences[1:]
        return captions
    return []


def extract(args):
    file = args.pdf_file_path
    images_path = args.img_main_path
    os.makedirs(images_path, exist_ok=True)
    # open the file
    pdf_file = fitz.open(file)

    page_texts = []

    image_caption_dict = []

    for page_index in range(len(pdf_file)):

        page = pdf_file[page_index]
        page_text = page.get_text()
        page_texts.append(page_text)

        image_list = page.get_images()

        page_captions = extract_caption(page_text)

        if len(page_captions) == 0:
            continue

        m = (len(image_list) + 1) // 2

        for image_index, img in enumerate(image_list, start=1):

            xref = img[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image = Image.open(io.BytesIO(image_bytes))
            image_name = f"page_{page_index}_img_{image_index}.jpg"
            image.save(os.path.join(images_path, image_name))
            new_dict = {
                "image": image_name,
                "caption": page_captions[
                    0 if image_index <= m or len(page_captions) == 1 else 1
                ],
            }
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
