# srGAN-MI

This project aims to create a generative AI model for artificial image quality enhancement, specifically focusing on improving radiological images. The main objective is to address the issue of quality loss in medical images due to various reasons, aiming to obtain more standardized images in terms of quality. This will not only provide better images for medical personnel to read but also add value as data for future applications in model development. This project is developed in Python using the PyTorch deep learning framework. Image preprocessing and manipulation were carried out with the OpenCV computer vision framework, and the implemented metrics were monitored using the TensorBoard framework.

The project is organized as follows:
```{sh}
/data 
/docs
 |----HowToTest.md
 |----HowToTrain.md
 |----project.docx
/models
/notebooks
 |----01-SRGAN.ipynb
 |----02-VGG-19.ipynb
 |----03-Adversarial-loss.ipynb
 |----04-Grad-CAM.ipynb
 |----05-PSNR.ipynb
 |----06-dataset.ipynb
 |----07-cycleGAN-Consistency.ipynb
 |----08-tensorboard.ipynb
 |----09-ImageProcessing.ipynb
/references
 |----srgan.bib 
/scripts
 |----test.sh
 |----train.sh 
/source
  |--/srGAN
     |----losses.py
     |----main.py
     |----mode.py
     |----ops.py
     |----srgan_model.py
     |----vgg19.py
  |--/real-ESR-GAN
     |----losses.py
     |----main.py
     |----mode.py
     |----real-esr-gan-model.py
     |----vgg19.py
     |---- ops.py
 /test
```

## Directories Description

- **Data**: This directory stores the datasets for training, validation, and testing the model. Docs: Contains project documentation, including how to use this repository, theoretical foundation, and motivation for the project. 
- **Notebooks**: Includes various notebooks that explain key elements implemented in the code, providing clarity on why they are used and how they can be modified and reproduced in the future. 
- **References**: All bibliographic material used for the development of this project. 
- **Scripts**: Python files with all the requirements for the computational implementation of the model. 
- **Test**: Stores the results of the tests conducted with the model.