# 🚀 Road Seqmentation Task With A Segmentation Dataset Available!
About
=====
What is Image Segmentation? 

At its core, image segmentation is the process of partitioning a digital image into multiple segments, or "regions of interest" (ROIs). The goal is to simplify the image's representation into something more meaningful and easier to analyze.
This project is based on these main resources:
1) <b>U-Net<b> : [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
2) Modern computer vision with Pytorch book:[Amazon Website](https://www.amazon.com/Modern-Computer-Vision-PyTorch-comprehensive/dp/1803231335).
3) <b>Vgg16<b>:[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).

# Let's dive into U-Net and VGG16 architecture (a quick summary)
<img src="Pics/model architecture/unet.png">

# ⚙️ Key Architecture Features
## 1. U-Shaped Design
- **Encoder**(Contracting Path): Left side of the "U" that captures context 
- **Decoder**(Expanding Path): Right side of the "U" that enables precise localization   
- **Bottleneck**: The bottom layer connecting encoder and decoder
## 2. Skip Connections
- The most important feature - connects corresponding layers from encoder to decoder
- Preserves spatial information lost during downsampling
- Combines high-level features with fine-grained details
## 3. Fully Convolutional
- No dense layers at the end
- Can handle input images of any size
- Outputs segmentation maps of the same spatial dimensions as input

# 💡 Why U-Net is So Effective?
## 1. Excellent for Small Datasets
- Originally designed for medical imaging where labeled data is scarce
- Data augmentation techniques work well with its architecture
## 2. Precise Boundary Detection
- Skip connections preserve spatial details
- Perfect for tasks requiring pixel-level accuracy
## 3. Computationally Efficient
- Relatively simple architecture compared to later models
- Fast training and inference

<img src="Pics/model architecture/vgg16.png">

# ⚙️ Key Architecture Features
## 1. Simplicity and Uniformity
- Uses only 3×3 convolutional layers throughout the entire network
- Uses only 2×2 max pooling layers for downsampling
-Simple, repetitive structure makes it easy to understand and implement
## 2. 16-Layer Depth
- 13 convolutional layers + 3 fully connected layers = 16 weight layers
- One of the deepest networks of its time (2014)
## 3. Increasing Filter Depth
- Filter depth doubles after each max pooling: 64 → 128 → 256 → 512 → 512

# 💡 Key Innovations
## 1. Small Receptive Fields
- Stacking multiple 3×3 conv layers has same receptive field as larger kernels But with fewer parameters and more non-linearities

## 2. Depth Over Complexity
- Proved that network depth is crucial for performance
- Inspired even deeper architectures (ResNet, etc.)

# Theory is enogh,lets go for code

## 🗺️ *The path to be taken to build this project*
```
Load dataset and apply mask annotations on them and create mask images
↓
Preproccess and visualize images and masks and put them in dataloader
↓
Define model architecture and use vgg16 as a backbone for feature extraction
↓
Define two metrics,IoU(Intersection over Union) and Dice score for evaluation
↓
Train model to learn patterns and decrease loss and increase dice score
↓
Use Fast API and use for prediction
↓
Project has been completed!
```

## *Let see what is a road segmentation task*
