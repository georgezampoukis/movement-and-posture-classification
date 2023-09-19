# movement-and-posture-classification
Accurate monitoring of 24-hour real-world movement behavior in people with cerebral palsy is possible using multiple wearable sensors and deep learning.


Abstract: Monitoring and quantifying movement behavior is crucial for improving the health of
individuals with Cerebral Palsy (CP). We have modeled and trained an image-based Convolutional
Neural Network (CNN) to recognize specific movement classifiers relevant to individuals with CP.
This study evaluates the CNNs performance and determine the feasibility of 24-hour recordings. 7
sensors provided accelerometer and gyroscope data from 14 typically developed adults during
videotaped physical activity. The performance of the CNN was assessed against test data and
human video annotation. For feasibility testing, one typically developed adult and one adult with
CP wore sensors for 24 hours. The CNN demonstrated exceptional performance against test data,
with a mean accuracy of 99.7%. Its general true positives (TP) and true negatives (TN) were 1.00.
Against human annotators, performance was high, with mean accuracy at 83.4%, TP 0.84, and TN
0.83. 24-hour recordings were successful without data loss or adverse events. Participants wore
sensors for the full wear time, and the data output was credible. We conclude that monitoring real-
world movement behavior in individuals with CP is possible with multiple wearable sensors and
CNN. This is of great value for identifying functional decline and informing new interventions,
leading to improved outcomes.

Keywords: Cerebral palsy, movement behavior; wearable sensors; deep learning; monitoring


------------- NETWORK ARCHITECTURE -------------

Overview

The network is a deep convolutional neural network (CNN) designed for image classification tasks. It is characterized by its use of residual blocks, skip connections, and instance normalization to enhance the learning process. It is a custom architecture that combines elements and ideas from well-known and established networks such as ResNet, VGG and InceptionV3.

Key Components

1.1 Strided Conv2D

Strided Conv2D is a custom 2D convolutional layer with stride of 2, followed by instance normalization and LeakyReLU activation function. We perform this operation with a stride of 2 (except the first one) to gradually reduce spatial dimensions while increasing the feature pool. Kernel sizes are gradually descending, from 9 to 3, to enable the network to focus more on high frequency details as the feature pool increases. After the convolution, instance normalization and LeakyReLU activation with a negative slope of 0.2 are applied to the output feature maps to improve convergence and counteract vanishing gradients.

1.2 SkipConv2D

SkipConv2D is a special 2D convolutional layer that incorporates skip connections. These connections combine the feature maps from before and after the use of residual blocks, which can help maintain gradient flow and promote faster convergence.

1.3 Residual Block

Residual Block is a building block consisting of two Conv2D layers. The input is added to the output of the second Conv2D layer, creating a skip connection that helps maintain gradient flow through the network. This design choice enables the network to learn more complex features and improves its ability to generalize.

1.4 Dropout2D

Dropout2D is employed to reduce overfitting and improve the generalization capability of the architecture. The dropout rate is a configurable parameter, with a default value of 0.5.
1.5 InstanceNorm2D

For normalization layers, InstanceNorm2D was chosen as it best fits the encoded 2-channel image, allowing for faster convergence and optimal weight normalization.


1.6 FCN Approach

The network follows a fully convolutional approach (FCN), being consisted only by convolutional layers and no linear/dense layers. While most classification networks rely on dense layers for the classification part, we found, after experimentation, that a fully convolutional approach yielded slightly better results. This approach is highly beneficial as it preserves spatial information throughout the whole network while also significantly improving model efficiency by drastically decreasing the number of trainable parameters.



Implementation & Training

The model and the training logic were implemented in python 3.11.0 using the Pytorch framework. An initial learning rate of 1e-4 was used, gradually decayed to 5e-6 to achieve the lowest possible loss (Binary Cross Entropy). We make use of the Adam optimizer with β1=0.9, β2=0.999. Batch size was set to 16 and the train - validation split was set to 80 - 20 respectively. The model achieved best loss after 50 training epochs, or 25.000 total steps for a dataset of 10000 samples.


Data Processing & Encoding

The data from each Mbient sensor is stored in a CSV format which represents a time series of x, y, z data from each of the 2 sub-sensors (Accelerometer & Gyroscope). We make use of a 1-second timeframe as our classification window, and we record data from the sensors with a sampling rate of 50Hz. As the 2D convolution network was designed to accept 2-channel images (3D Arrays) as input, the data from the sensors must be encoded in such an image, that would effectively represent 1 second worth of data from 7 sensors. To achieve this, we first slice 50 samples (1 second of data) from each component (x, y, z) from each sub-sensor (Accelerometer & Gyroscope) from each of the 7 sensors used for the recording. These samples are then concatenated vertically, as demonstrated in Fig. 1, with all the accelerometer data in channel 1 and all the gyroscope data in channel 2, producing final image of dimension: [2, 21, 50]  – [Channels, Height, Width]. The channels effectively represent the sub-sensors (Accelerometer & Gyroscope -> 2), the height represents the 7 sensors (7 sensors x 3 components x, y, z -> 21) and finally, the width represents the samples (1 second of samples at 50Hz sampling rate -> 50 samples). Lastly, the encoded 3D array is resized to 2x64x64 using nearest neighbour interpolation which produces the final encoded 2-channel image.


Dataset

To train the network, a custom dataset was created by timestamped scripted movements consisting of 10000 1-second samples. During this process, a python script was used that would indicate a range of movements to a performer, covering different label combinations every time so that a balanced and diverse dataset can be created. Since the indicated movements were timestamped in conjunction with the sensor recordings, we eliminated the need of manually annotating data, which allowed for a very dense dataset to be created in a very short amount of time. This automated approach of producing data also allowed for quick prototyping of ideas, since no time is wasted in manual annotation of data. Since the data that the sensors record is contained within a specific range (± 10) and is already represented as float32 values, no data normalization was applied.

