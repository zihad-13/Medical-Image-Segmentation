# %%
"""
<img src="https://www.med.upenn.edu/cbica/assets/user-content/images/BraTS/brats-tumor-subregions.jpg" alt="rats official annotations" style="width: 1500px;"/>
"""

# %%
"""
# Setup

## Import the libraries and define constants
"""

# %%
import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  


# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
#import nifti2gif.core as nifti2gif


# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# %%
# DEFINE seg-areas  
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 32 
VOLUME_START_AT = 22 # first slice of volume that we will include

# %%
"""
# Image data descriptions

All BraTS multimodal scans are available as  NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings 
1. **T1**: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
2. **T1c**: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
3. **T2**: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
4. **FLAIR**: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

Data were acquired with different clinical protocols and various scanners from multiple (n=19) institutions.

All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.


"""

# %%
TRAIN_DATASET_PATH = 'Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = 'Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
ax5.set_title('Mask')


# %%
"""
**Show whole nifti data -> print each slice from 3d data**
"""

# %%
# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_image_t1[50:-50,:,:]), 90, resize=True), cmap ='gray')

# %%
"""
**Show segment of tumor for each above slice**
"""

# %%
# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_mask[60:-60,:,:]), 90, resize=True), cmap ='gray')

# %%
shutil.copy2(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii', './test_gif_BraTS20_Training_001_flair.nii')
#nifti2gif.write_gif_normal('./test_gif_BraTS20_Training_001_flair.nii')

# %%
"""
**Gif representation of slices in 3D volume**
<img src="https://media1.tenor.com/images/15427ffc1399afc3334f12fd27549a95/tenor.gif?itemid=20554734">
"""

# %%
"""
**Show segments of tumor using different effects**
"""

# %%
niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
nimask = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


nlplt.plot_anat(niimg,
                title='BraTS20_Training_001_flair.nii plot_anat',
                axes=axes[0])

nlplt.plot_epi(niimg,
               title='BraTS20_Training_001_flair.nii plot_epi',
               axes=axes[1])

nlplt.plot_img(niimg,
               title='BraTS20_Training_001_flair.nii plot_img',
               axes=axes[2])

nlplt.plot_roi(nimask, 
               title='BraTS20_Training_001_flair.nii with mask plot_roi',
               bg_img=niimg, 
               axes=axes[3], cmap='Paired')

plt.show()

# %%
"""
# Create model || U-Net: Convolutional Networks for Biomedical Image Segmentation

U-net is a convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin
[more on](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
![official definiton](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

"""

# %%
"""
# Loss function
**Dice coefficient**
, which is essentially a measure of overlap between two samples. This measure ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap. The Dice coefficient was originally developed for binary data, and can be calculated as:

![dice loss](https://wikimedia.org/api/rest_v1/media/math/render/svg/a80a97215e1afc0b222e604af1b2099dc9363d3b)

**As matrices**
![dice loss](https://www.jeremyjordan.me/content/images/2018/05/intersection-1.png)

[Implementation, (images above) and explanation can be found here](https://www.jeremyjordan.me/semantic-segmentation/)
"""

# %%
# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
#    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss


 
# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def balanced_accuracy(y_true, y_pred):
    return (sensitivity(y_true, y_pred) + specificity(y_true, y_pred)) / 2

# %%
IMG_SIZE=128
# %%

from model import Patches,PatchEncoder,mlp
from tensorflow.keras import layers

input_shape=(8,8,512)
patch_size=8
num_patches =patch_size**2#(8 // patch_size) ** 2
projection_dim = 512
transformer_layers = 3
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]

mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
def VIT(inputs):
    #inputs = tf.keras.Input(shape=[ 8, 8, 256])#layers.Input(shape=input_shape)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    #import pdb;pdb.set_trace()
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    encoded_patches=tf.reshape(encoded_patches,(-1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1,x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    return encoded_patches

# %%
# source https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a

def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool)
    conv = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv3)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)
    
    #import pdb;pdb.set_trace()
    ###### ########### ViT
    transformer_input=drop5
    transformer_output = VIT(transformer_input)
    transformer_output = Dropout(0.1)(transformer_output)   
    
    
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(transformer_output)) # drop5
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv9)
    
    up = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv9))
    merge = concatenate([conv1,up], axis = 3)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(merge)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = ker_init)(conv)
    
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv)
    
    return Model(inputs = inputs, outputs = conv10)

input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

model = build_unet(input_layer, 'he_normal', 0.2)
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity,balanced_accuracy, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )

# %%
"""
**model architecture** <br>
If you are about to use U-NET, I suggest to check out this awesome library that I found later, after manual implementation of U-NET [keras-unet-collection](https://pypi.org/project/keras-unet-collection/), which also contains implementation of dice loss, tversky loss and many more!
"""

# %%
plot_model(model, 
           show_shapes = True,
           show_dtype=False,
           show_layer_names = True, 
           rankdir = 'TB', 
           expand_nested = False, 
           dpi = 70)

# %%
"""
# Load data
Loading all data into memory is not a good idea since the data are too big to fit in.
So we will create dataGenerators - load data on the fly as explained [here](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
"""

# %%
# lists of directories with studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

# file BraTS20_Training_355 has ill formatted name for for seg.nii file
train_and_val_directories.remove(TRAIN_DATASET_PATH+'BraTS20_Training_355')


def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories); 

    
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 

# %%
"""
**Override Keras sequence DataGenerator class**
"""

# %%
def mbobhe(image):
    # Ensure the image is of depth CV_8U (8-bit unsigned)
    if image.dtype != np.uint8:
        # Scale the image to 8-bit depth if it's not already
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Check if the image is 4-channel, if so, process each channel separately
    if len(image.shape) == 3 and image.shape[2] == 2:
        # Create an empty array to store the processed channels
        processed_channels = []

        # Process each channel separately
        for channel in cv2.split(image):
            # Apply histogram equalization to the channel
            equalized_channel = cv2.equalizeHist(channel)
            processed_channels.append(equalized_channel)

        # Recombine the processed channels into a 4-channel image
        equalized_image = cv2.merge(processed_channels)

    else:
        # If the image is not 4-channel, convert it to grayscale and apply histogram equalization
        equalized_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(equalized_image)

    return equalized_image

def mphe(image):
    equalized_channels = []
    
    for channel in range(image.shape[-1]):
        # Extract the individual channel
        channel_data = image[:, :, channel]
        
        # Compute the histogram of the channel
        hist, bins = np.histogram(channel_data.flatten(), 256, [0, 256])

        # Calculate the cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        # Apply multiplicative equalization to the channel
        equalized_channel = np.interp(channel_data, bins[:-1], cdf_normalized)

        # Convert back to uint8 (0-255 range)
        equalized_channel = (255 * (equalized_channel / equalized_channel.max())).astype(np.uint8)

        equalized_channels.append(equalized_channel)
    
    # Stack the equalized channels to form the equalized image
    equalized_image = np.stack(equalized_channels, axis=-1)

    return equalized_image
def clahe(image):
    image = (image * 255).astype(np.uint8)

    # Split the image into its 3 channels
    b, g, = cv2.split(image)
    #import pdb;pdb.set_trace()
    # Create CLAHE objects for each channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to each channel separately
    #b=cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    # r_clahe = clahe.apply(r)
    # a_clahe = clahe.apply(a)
    # Merge the CLAHE-enhanced channels back into a 3-channel image
    clahe_image = cv2.merge([b_clahe, g_clahe])

    return clahe_image
# %%
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True, fe=''):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        
        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(data_path).get_fdata()    

            data_path = os.path.join(case_path, f'{i}_t1ce.nii');
            ce = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(data_path).get_fdata()
        
            for j in range(VOLUME_SLICES):
                 X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                 X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                 y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];
        #import pdb;pdb.set_trace()
        if fe=='clahe':
            for i in range(X.shape[0]):
                X[i]=clahe(X[i])
        elif fe=='mphe':
            for i in range(X.shape[0]):
                X[i]=mphe(X[i])
        elif fe=='mbobhe':
            for i in range(X.shape[0]):
                X[i]=mbobhe(X[i])
        
        else: X=X
        
        
        # Generate masks
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X/np.max(X), Y
fe='mbobhe'        
training_generator = DataGenerator(train_ids,fe=fe)
valid_generator = DataGenerator(val_ids,fe=fe)
test_generator = DataGenerator(test_ids,fe=fe)

# %%
"""
**Number of data used**
for training / testing / validation
"""

# %%
# show number of data for each dir 
def showDataLayout():
    plt.bar(["Train","Valid","Test"],
    [len(train_ids), len(val_ids), len(test_ids)], align='center',color=[ 'green','red', 'blue'])
    plt.legend()

    plt.ylabel('Number of images')
    plt.title('Data distribution')

    plt.show()
    
showDataLayout()

# %%
"""
**Add callback for training process**
"""

# %%
csv_logger = CSVLogger('training.log', separator=',', append=False)


callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
#  keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
        csv_logger
    ]



# %%
K.clear_session()

history =  model.fit(training_generator,
                    epochs=20,
                    steps_per_epoch=len(train_ids),
                    callbacks= callbacks,
                    validation_data = valid_generator
                    )  
#model.save("model_x1_1.h5")


model_json = model.to_json()
with open("model_brats_%s.json"%fe, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_brats_%s.h5"%fe)
print("Saved model to disk")


# %%
"""
**Visualize the training process**
"""

# %%
############ load trained model ################
# model = keras.models.load_model('../input/modelperclasseval/model_per_class.h5', 
#                                    custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
#                                                    "dice_coef": dice_coef,
#                                                    "precision": precision,
#                                                    "sensitivity":sensitivity,
#                                                    "specificity":specificity,
#                                                    "dice_coef_necrotic": dice_coef_necrotic,
#                                                    "dice_coef_edema": dice_coef_edema,
#                                                    "dice_coef_enhancing": dice_coef_enhancing
#                                                   }, compile=False)

# history = pd.read_csv('../input/modelperclasseval/training_per_class.log', sep=',', engine='python')

# hist=history

# ############### ########## ####### #######

hist=history.history

acc=hist['accuracy']
val_acc=hist['val_accuracy']

epoch=range(len(acc))

loss=hist['loss']
val_loss=hist['val_loss']

train_dice=hist['dice_coef']
val_dice=hist['val_dice_coef']

# f,ax=plt.subplots(1,4,figsize=(16,8))

# ax[0].plot(epoch,acc,'b',label='Training Accuracy')
# ax[0].plot(epoch,val_acc,'r',label='Validation Accuracy')
# ax[0].legend()

# ax[1].plot(epoch,loss,'b',label='Training Loss')
# ax[1].plot(epoch,val_loss,'r',label='Validation Loss')
# ax[1].legend()

# ax[2].plot(epoch,train_dice,'b',label='Training dice coef')
# ax[2].plot(epoch,val_dice,'r',label='Validation dice coef')
# ax[2].legend()

# ax[3].plot(epoch,hist['mean_io_u'],'b',label='Training mean IOU')
# ax[3].plot(epoch,hist['val_mean_io_u'],'r',label='Validation mean IOU')
# ax[3].legend()

# Plot loss and metrics
plt.figure(figsize=(16, 12))
plt.subplot(5,1, 1)
plt.plot(range(len(history.history['loss'])), history.history['loss'], label='Loss', color='b')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'], label='Accuracy', color='g')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(range(len(history.history['specificity'])), history.history['specificity'], label='Specificity', color='r')
plt.title('Specificity Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Specificity')
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(range(len(history.history['sensitivity'])), history.history['sensitivity'], label='Sensitivity', color='m')
plt.title('Sensitivity Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Sensitivity')
plt.legend()

plt.subplot(5, 1, 5)
plt.plot(range(len(history.history['balanced_accuracy'])), history.history['balanced_accuracy'], label='Balanced Accuracy', color='c')
plt.title('Balanced Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.legend()

plt.show()

# %%
"""
# Prediction examples 
"""

# %%
# mri type must one of 1) flair 2) t1 3) t1ce 4) t2 ------- or even 5) seg
# returns volume of specified study at `path`
def imageLoader(path):
    image = nib.load(path).get_fdata()
    X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
    for j in range(VOLUME_SLICES):
        X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(image[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
        X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

        y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];
    return np.array(image)


# load nifti file at `path`
# and load each slice with mask from volume
# choose the mri type & resize to `IMG_SIZE`
def loadDataFromDir(path, list_of_files, mriType, n_images):
    scans = []
    masks = []
    for i in list_of_files[:n_images]:
        fullPath = glob.glob( i + '/*'+ mriType +'*')[0]
        currentScanVolume = imageLoader(fullPath)
        currentMaskVolume = imageLoader( glob.glob( i + '/*seg*')[0] ) 
        # for each slice in 3D volume, find also it's mask
        for j in range(0, currentScanVolume.shape[2]):
            scan_img = cv2.resize(currentScanVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
            mask_img = cv2.resize(currentMaskVolume[:,:,j], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA).astype('uint8')
            scans.append(scan_img[..., np.newaxis])
            masks.append(mask_img[..., np.newaxis])
    return np.array(scans, dtype='float32'), np.array(masks, dtype='float32')
        
#brains_list_test, masks_list_test = loadDataFromDir(VALIDATION_DATASET_PATH, test_directories, "flair", 5)


# %%
def predictByPath(case_path,case):
    files = next(os.walk(case_path))[2]
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
  #  y = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE))
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
    flair=nib.load(vol_path).get_fdata()
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
    ce=nib.load(vol_path).get_fdata() 
    
 #   vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_seg.nii');
 #   seg=nib.load(vol_path).get_fdata()  

    
    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
 #       y[j,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        
  #  model.evaluate(x=X,y=y[:,:,:,0], callbacks= callbacks)
    return model.predict(X/np.max(X), verbose=1)


def showPredictsById(case, start_slice = 5):
    path = f"Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByPath(path,case)
    #import pdb;pdb.set_trace()
    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(18, 20))
    f, axarr = plt.subplots(1,6, figsize = (22, 4)) 

    for i in range(6): # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
    
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('all classes')
    axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    f.suptitle("Visualization result for Brats 2020 dataset and Feature Enhanchement %s"%fe)
    plt.show()
    
    
showPredictsById(case=test_ids[0][-3:])
showPredictsById(case=test_ids[1][-3:])
showPredictsById(case=test_ids[2][-3:])
showPredictsById(case=test_ids[3][-3:])
showPredictsById(case=test_ids[4][-3:])
showPredictsById(case=test_ids[5][-3:])
showPredictsById(case=test_ids[6][-3:])


# mask = np.zeros((10,10))
# mask[3:-3, 3:-3] = 1 # white square in black background
# im = mask + np.random.randn(10,10) * 0.01 # random image
# masked = np.ma.masked_where(mask == 0, mask)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(im, 'gray', interpolation='none')
# plt.subplot(1,2,2)
# plt.imshow(im, 'gray', interpolation='none')
# plt.imshow(masked, 'jet', interpolation='none', alpha=0.7)
# plt.show()

# %%
"""
# Evaluation
"""

# %%

from tensorflow.keras.models import model_from_json
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, jaccard_score, roc_auc_score

# load json and create model
json_file = open("model_brats_%s.json"%fe, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json,custom_objects={"Patches": Patches,'PatchEncoder':PatchEncoder})
# load weights into new model
model.load_weights("model_brats_%s.h5"%fe)
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
              metrics=['accuracy', specificity, sensitivity, balanced_accuracy])
# Evaluate the model on the test data after training
test_loss, test_accuracy, test_specificity, test_sensitivity, test_balanced_accuracy = model.evaluate(
    test_generator, verbose=0)
print("Final Performance Metrics:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Specificity: {test_specificity:.4f}")
print(f"Test Sensitivity: {test_sensitivity:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")

# Calculate and print other metrics like F1, Cohen's Kappa, Precision, Recall, Jaccard Index, ROC AUC, etc.
predictions = model.predict(test_generator[0][0])
predictions_binary = (predictions > 0.6).astype(int)
masks_test_binary = (np.array(test_generator[0][1]) > 0.9).astype(int)

f1 = f1_score(masks_test_binary.flatten(), predictions_binary.flatten())
kappa = cohen_kappa_score(masks_test_binary.flatten(), predictions_binary.flatten())
precision = precision_score(masks_test_binary.flatten(), predictions_binary.flatten())
recall = recall_score(masks_test_binary.flatten(), predictions_binary.flatten())
jaccard = jaccard_score(masks_test_binary.flatten(), predictions_binary.flatten())
roc_auc = roc_auc_score(masks_test_binary.flatten(), predictions_binary.flatten())

print(f"F1 Score: {f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Jaccard Index: {jaccard:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
metrics=['Accuracy','Specificity','Sensitivity','Balanced Accuracy','F1 Score',"Cohen's Kappa","Precision","Recall","Jaccard Index","ROC AUC"]

MBOBHE=[test_accuracy, test_specificity, test_sensitivity, test_balanced_accuracy,f1,kappa,precision,recall,jaccard,roc_auc]


# %%
# %% For CLAHE
fe='clahe'
# load json and create model
json_file = open("model_brats_%s.json"%fe, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json,custom_objects={"Patches": Patches,'PatchEncoder':PatchEncoder})
# load weights into new model
model.load_weights("model_brats_%s.h5"%fe)
print("Loaded model from disk")

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
              metrics=['accuracy', specificity, sensitivity, balanced_accuracy])
# Evaluate the model on the test data after training
test_loss, test_accuracy, test_specificity, test_sensitivity, test_balanced_accuracy = model.evaluate(
    test_generator, verbose=0)

print("Final Performance Metrics:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Specificity: {test_specificity:.4f}")
print(f"Test Sensitivity: {test_sensitivity:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")

# Calculate and print other metrics like F1, Cohen's Kappa, Precision, Recall, Jaccard Index, ROC AUC, etc.
predictions = model.predict(test_generator[0][0])
predictions_binary = (predictions > 0.6).astype(int)
masks_test_binary = (np.array(test_generator[0][1]) > 0.9).astype(int)

f1 = f1_score(masks_test_binary.flatten(), predictions_binary.flatten())
kappa = cohen_kappa_score(masks_test_binary.flatten(), predictions_binary.flatten())
precision = precision_score(masks_test_binary.flatten(), predictions_binary.flatten())
recall = recall_score(masks_test_binary.flatten(), predictions_binary.flatten())
jaccard = jaccard_score(masks_test_binary.flatten(), predictions_binary.flatten())
roc_auc = roc_auc_score(masks_test_binary.flatten(), predictions_binary.flatten())

print(f"F1 Score: {f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Jaccard Index: {jaccard:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
metrics=['Accuracy','Specificity','Sensitivity','Balanced Accuracy','F1 Score',"Cohen's Kappa","Precision","Recall","Jaccard Index","ROC AUC"]

CLAHE=[test_accuracy, test_specificity, test_sensitivity, test_balanced_accuracy,f1,kappa,precision,recall,jaccard,roc_auc]

# %% For MPHE
fe='mphe'
# load json and create model
json_file = open("model_brats_%s.json"%fe, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json,custom_objects={"Patches": Patches,'PatchEncoder':PatchEncoder})
# load weights into new model
model.load_weights("model_brats_%s.h5"%fe)
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
              metrics=['accuracy', specificity, sensitivity, balanced_accuracy])
# Evaluate the model on the test data after training
test_loss, test_accuracy, test_specificity, test_sensitivity, test_balanced_accuracy = model.evaluate(
    test_generator, verbose=0)

print("Final Performance Metrics:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Specificity: {test_specificity:.4f}")
print(f"Test Sensitivity: {test_sensitivity:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")

# Calculate and print other metrics like F1, Cohen's Kappa, Precision, Recall, Jaccard Index, ROC AUC, etc.
predictions = model.predict(test_generator[0][0])
predictions_binary = (predictions > 0.6).astype(int)
masks_test_binary = (np.array(test_generator[0][1]) > 0.9).astype(int)

f1 = f1_score(masks_test_binary.flatten(), predictions_binary.flatten())
kappa = cohen_kappa_score(masks_test_binary.flatten(), predictions_binary.flatten())
precision = precision_score(masks_test_binary.flatten(), predictions_binary.flatten())
recall = recall_score(masks_test_binary.flatten(), predictions_binary.flatten())
jaccard = jaccard_score(masks_test_binary.flatten(), predictions_binary.flatten())
roc_auc = roc_auc_score(masks_test_binary.flatten(), predictions_binary.flatten())

print(f"F1 Score: {f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Jaccard Index: {jaccard:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
metrics=['Accuracy','Specificity','Sensitivity','Balanced Accuracy','F1 Score',"Cohen's Kappa","Precision","Recall","Jaccard Index","ROC AUC"]

MPHE=[test_accuracy, test_specificity, test_sensitivity, test_balanced_accuracy,f1,kappa,precision,recall,jaccard,roc_auc]

# %%

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate('%.2f' % height,
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', rotation=45)
def plotting(model1,model2,model3,modelnames,metrics,title='Performance analysis',N = 3):
    import numpy as np 
    import matplotlib.pyplot as plt 
     # Set the figure size
    plt.figure(figsize=(22, 18))
    
    ind = np.arange(N)  
    width = 0.25
      
    xvals = model1
    bar1 = plt.bar(ind, xvals, width, color = 'skyblue') 
    autolabel(bar1)  
    yvals = model2
    bar2 = plt.bar(ind+width, yvals, width, color='lightblue') 
    autolabel(bar2)  
    zvals =model3
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'olive') 
    autolabel(bar3)  
    plt.xlabel("Metrics") 
    plt.ylabel('Scores') 
    plt.title(title) 
    

    # Set x-axis ticks with 45-degree rotation
    plt.xticks(ind + width, metrics, rotation=45, ha ='right')  
    #plt.legend((bar1, bar2, bar3), modelnames, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.legend((bar1, bar2, bar3), modelnames,  ncol=3)
    #plt.legend( (bar1, bar2, bar3), modelnames ) # modelnames

    plt.show()
# %%
modelnames=['FE-CLAHE',"FE-MBOBHE","FE-MPHE"]
plotting(CLAHE,MBOBHE,MPHE,modelnames,metrics,title='Performance analysis for Brats2020 Dataset',N = len(CLAHE))
