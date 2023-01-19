# Lane-Extraction-from-Grass-Surfaces

A Tensorflow and Keras implementation of Semantic Segmentation to extract white lanes from surfaces with grass and dirt.
This model can be useful for autonomous robots to detect lanes in non-conventianal roads, such as the IGVC autonomous course.

## Netwok Architecture
This network uses an encoder-decoder architecture to create a mask from the image. 
<br>The encoder consists of several consecutive residual blocks. This model mainly uses two types of residual blocks:
1. Convolutional Block: Here the skipped layer undergoes convolution before getting merged with the main flow.

![alt text](/models/convolutional_block.png)

2. Identity Block: Here the skipped layer does not undergo any function before merging with the main flow.

![alt text](/models/identity_block.png)

To reduce dimentions, I took some inspiration from the Deeplab-v3 model and used depthwise seperable convolutions instead of maxpooling. I also applied several dilated convolutions with rates of (3,3) to give an effect of a (7,7) filter.

The decoder consists of several initial atrous convolutions with rates of (3,3), (5,5) and (7,7), followed by a convolution function. Further, transposed convolutions are applied to increase the dimentions back to the required shape. Also, a layer from the encoder is concatenated with a layer of the same shape of the decoder to ensure the model does not deviate from what it has to generate.

The basic architecture is as follows:

![alt text](/models/model_architechture.png)

For a more detailed model architecture check out [this image.](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/models/model_summary.png)

## Requirements
The model needs the following libraries and packages:
* Tensorflow-gpu
* OpenCV
* Numpy
* Matplotlib

## Test the Model
To check the performance of the model on your own images, you first need to download the weights from [here.](https://drive.google.com/file/d/14EkHsn-_x4Ss1uwLKWcEEjBFXjQ-DEDB/view?usp=sharing) The model is approximately 500 MB large and is trained on approximately 600 images for nearly 25 epochs. If you want to explore these images, then their tfrecords can be found [here.](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/tree/master/Dataset/tfrecords)

You can test the model on a single image by typing:
```python
python3 predict.py ./path/to/input_image ./path/to/saved/model.h5
```
## Results
The current model gave the following predictions:

**Image** | **Annotation** | **Prediction**
----------|----------------|---------------
![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/Dataset/Samples/image_5.jpg) | ![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/Dataset/Samples/label_5.png) | ![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/results/pred_5.png)
![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/Dataset/Samples/image_4.jpg) | ![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/Dataset/Samples/label_4.png) | ![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/results/pred_4.png)
![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/Dataset/Samples/image_3.jpg) | ![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/Dataset/Samples/label_3.png) | ![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/results/pred_3.png)

## Test Your Own Model

### Prepare Your dataset
This model expects the data to be in a tfrecord format. You can use the help of tfrecord_utils.py to convert your dataset into tfrecord files by providing the necessry arguments.

**Example:** Let ```filename_pairs``` be a list containing image_path and annotation_path pairs in a tuple. So the list will look like: ```filename_pairs = [(image1, label1), (image2, label2) .......]```

Let ```tfrecord_path``` be a string containing the name of the target tfrecord file.

Now, to generate a tfrecord file you have to include the following snippet in your code:
```python
import tfrecord_utils
...
tfrecord_utils.write_image_annotation_pairs_to_tfrecord(filename_pairs, tfrecord_path)
```
This utils file can also be used to parse images from the tfrecords for debugging.

### Start Training
To train the model, you have to run the LaneExtraction.ipynb notebook. Make sure you edit the lines specifying the name of your dataset (tfrecord file names).

After training the model for about 25 epochs on my dataset, it gave a training accuracy of 99.53 %, validation accuracy of 98.87 % and test accuracy of 98.80 %. The accuracy and loss throughout the process varied as follows:

![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/models/accuracy.png)

![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/models/loss.png)

## References
Most of the images for the dataset were captured by my own camera and annotated using the labelme module. Some images were also taken from the youtube video of the IGVC Course track which can be found from the following link: https://www.youtube.com/watch?v=A9BVr7kltl8

The model takes several inspirations from the Deeplab-V3 model which can be found from the following link:
https://github.com/tensorflow/models/tree/master/research/deeplab

## TODO

- [ ] Improve the model performance by training it on a bigger dataset.
- [ ] Simplify the process for generating datasets and training the model.



