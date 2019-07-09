# Lane-Extraction-from-Grass-Surfaces

A Tensorflow and Keras implementation of Semantic Segmentation to extract while lanes from surfaces with grass and dirt.
This model can be useful for autonomous robots to detect lanes in non-conventianal roads, such as the IGVC autonomous course.

## Netwok Architechture
This network uses an encoder-decoder architechture to create a mask from the image. The encoder consists of several residual blocks along with depthwise seperable convolutions to reduce dimentions. The decoder consists of atrous convolutions followed by transposed convolutions to increase the dimentions back to the required shape.
The basic architecture is as follows:

![alt text](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/models/model_architechture.png "hello")

For a detailed model architechture check out [this image.](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/blob/master/models/model.png)

## Requirements
The model needs the following libraries and packages:
* Tensorflow-gpu
* OpenCV
* Numpy
* Matplotlib

## Test Model
To check the performance of the model on your own custom images, you first need to download it from [here.](https://drive.google.com/file/d/14EkHsn-_x4Ss1uwLKWcEEjBFXjQ-DEDB/view?usp=sharing) The model is approximately 500 MB large and is trained on approximately 600 images for nearly 25 epochs. If you want to explore these images, then their tfrecords can be found [here.](https://github.com/Chinnu1103/Lane-Extraction-from-Grass-Surfaces/tree/master/Dataset/tfrecords)

You can test the model on a single image by typing:
```python
python3 predict.py ./path/to/input_image ./path/to/saved/model.h5
```
## Results

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
