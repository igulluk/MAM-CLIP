# Integrating Image Interpretation and Textual Context for Improved Breast Imaging Classification



## Folder Structure

The `data-extract/` folder contains the following Python scripts:

- `extract_mam.py`: Script to extract images and captions from the **Atlas of Mammography**.
- `extract_acr.py`: Script to extract images and captions from the **BI-RADS® Atlas**.

The `train/` folder contains the following Python scripts:

- `cfg.yaml`: Config file. Convnext small/tiny option can be changed under vision_model.name 
- `mam_clip_load.ipynb`: Jupyter notebook examples of loading pretrained clip model as well as fine tuned vision models.
- `loaders.py`: Dataloaders file.
- `main.py`: Code for starting pretraining CLIP model on mammography images.
- `files/model.py`: Includes implementation of vision language model and pytorch lightning training model.
- `files/dataset.py`: Image-Text dataset file.

## Image and Caption Extraction from Medical Books

We extract image-caption pairs from the following books that are used in the pretraining.

1. **Atlas of Mammography, 3rd Edition**  
   [Link to Book](https://www.amazon.com/Atlas-Mammography-Ellen-Shaw-deParedes/dp/0781764335)

2. **BI-RADS® Atlas, 5th Edition e-book**  
   [Link to Book](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Bi-Rads)


### Running the Extraction Scripts

Each script requires three arguments:

1. **`pdf_file_path`**: Path to the PDF file of the book.
2. **`json_file_path`**: Path to save the output JSON file, which contains the image paths and extracted captions.
3. **`img_main_path`**: Directory to save the extracted images.

### Example Commands:

#### 1. Extracting Images and Captions from the Atlas of Mammography:

```bash
python data-extract/extract_mam.py --pdf_file_path /path/to/atlas_of_mammography.pdf --json_file_path /path/to/output/atlas_mammography_data.json --img_main_path /path/to/save/images/atlas_of_mammography_images/

python data-extract/extract_acr.py --pdf_file_path /path/to/birads_atlas.pdf --json_file_path /path/to/output/birads_atlas_data.json --img_main_path /path/to/save/images/birads_atlas_images/
```

## Model Weights
   [Model Weights](https://drive.google.com/drive/folders/1faHwdyjpDB5yQX0acgBlT8A3KgsVI2MM?usp=drive_link) are available.
   1. `pretrain_convnext_tiny.ckpt` Vision Language Model weights after pretraining where the image encoder is Convnext tiny model. 
   2. `pretrain_convnext_small.ckpt`  Vision Language Model weights after pretraining where the image encoder is Convnext small model. 
   3. `convnext_tiny_ftune.pth`  Finetuned version convnext tine using the mammography classification dataset.
   4. `convnext_small_ftune.pth`  Finetuned version convnext small using the mammography classification dataset.

   After downloading checkpoints, model weights can be loaded as shown in `train/mam_clip_load.ipynb` notebook.
   
## Acknowledgement
   1. [PMC-CLIP](https://github.com/WeixiongLin/PMC-CLIP/tree/main)
   2. [OpenCLIP](https://github.com/mlfoundations/open_clip)

Our vision language model code is based on  [PMC-CLIP](https://github.com/WeixiongLin/PMC-CLIP/tree/main)
 and [OpenCLIP](https://github.com/mlfoundations/open_clip). We followed the preprocessing pipeline of (https://github.com/dangnh0611/kaggle_rsna_breast_cancer) using YOLOX model to crop the breast region in mammography images. 


## Citation
```bash
@inproceedings{
gulluk2024integrating,
title={Integrating Image Interpretation and Textual Context for Improved Breast Imaging Classification},
author={Halil Ibrahim Gulluk and Olivier Gevaert},
booktitle={Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond},
year={2024},
url={https://openreview.net/forum?id=Sqom7PZBxe}
}
```

## Contact
If you have any questions, feel free to reach out at gulluk@stanford.edu.