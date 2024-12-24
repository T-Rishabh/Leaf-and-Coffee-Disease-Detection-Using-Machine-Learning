{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c5740791",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\KIIT\\anaconda3\\lib\\site-packages\\pandas\\core\\arrays\\masked.py:60: UserWarning: Pandas requires version '1.3.6' or newer of 'bottleneck' (version '1.3.5' currently installed).\n",
      "  from pandas.core import (\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "783113e3",
   "metadata": {},
   "source": [
    "# DATA PREPROCESSING"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36b5c14b",
   "metadata": {},
   "source": [
    "### Training Image Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "7e188520",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Images have been moved to separate folders based on annotations.\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import shutil\n",
    "\n",
    "# Directory containing images and annotation text file\n",
    "directory = 'train'\n",
    "\n",
    "# Read the annotation text file\n",
    "with open(os.path.join(directory, '_annotations.txt'), 'r') as file:\n",
    "    lines = file.readlines()\n",
    "\n",
    "# Process each line in the annotation text file\n",
    "for line in lines:\n",
    "    parts = line.strip().split()\n",
    "    image_name = parts[0]\n",
    "    annotations = parts[1:]\n",
    "\n",
    "    # Determine the class ID from the first annotation (assuming all annotations for an image have the same class ID)\n",
    "    class_id = annotations[0].split(',')[-1]\n",
    "\n",
    "    # Create a directory for the class if it doesn't exist\n",
    "    class_dir = os.path.join(directory, f'class_{class_id}')\n",
    "    if not os.path.exists(class_dir):\n",
    "        os.makedirs(class_dir)\n",
    "\n",
    "    # Move the image to the class directory\n",
    "    image_path = os.path.join(directory, image_name)\n",
    "    new_image_path = os.path.join(class_dir, image_name)\n",
    "    shutil.move(image_path, new_image_path)\n",
    "\n",
    "print('Images have been moved to separate folders based on annotations.')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b198c6c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 1236 files belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "training_set = tf.keras.utils.image_dataset_from_directory(\n",
    "    './train/',\n",
    "    labels=\"inferred\",\n",
    "    label_mode=\"categorical\",\n",
    "    class_names=None,\n",
    "    color_mode=\"rgb\",\n",
    "    batch_size=32,\n",
    "    image_size=(128, 128),\n",
    "    shuffle=True,\n",
    "    seed=None,\n",
    "    validation_split=None,\n",
    "    subset=None,\n",
    "    interpolation=\"bilinear\",\n",
    "    follow_links=False,\n",
    "    crop_to_aspect_ratio=False,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1cfab49c",
   "metadata": {},
   "source": [
    "### VALIDATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "6bcd4277",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Images have been moved to separate folders based on annotations.\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import shutil\n",
    "\n",
    "# Directory containing images and annotation text file\n",
    "directory = 'valid'\n",
    "\n",
    "# Read the annotation text file\n",
    "with open(os.path.join(directory, '_annotations.txt'), 'r') as file:\n",
    "    lines = file.readlines()\n",
    "\n",
    "# Process each line in the annotation text file\n",
    "for line in lines:\n",
    "    parts = line.strip().split()\n",
    "    image_name = parts[0]\n",
    "    annotations = parts[1:]\n",
    "\n",
    "    # Determine the class ID from the first annotation (assuming all annotations for an image have the same class ID)\n",
    "    class_id = annotations[0].split(',')[-1]\n",
    "\n",
    "    # Create a directory for the class if it doesn't exist\n",
    "    class_dir = os.path.join(directory, f'class_{class_id}')\n",
    "    if not os.path.exists(class_dir):\n",
    "        os.makedirs(class_dir)\n",
    "\n",
    "    # Move the image to the class directory\n",
    "    image_path = os.path.join(directory, image_name)\n",
    "    new_image_path = os.path.join(class_dir, image_name)\n",
    "    shutil.move(image_path, new_image_path)\n",
    "\n",
    "print('Images have been moved to separate folders based on annotations.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b916e177",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 200 files belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "validation_set = tf.keras.utils.image_dataset_from_directory(\n",
    "    './valid/',\n",
    "    labels=\"inferred\",\n",
    "    label_mode=\"categorical\",\n",
    "    class_names=None,\n",
    "    color_mode=\"rgb\",\n",
    "    batch_size=32,\n",
    "    image_size=(128, 128),\n",
    "    shuffle=True,\n",
    "    seed=None,\n",
    "    validation_split=None,\n",
    "    subset=None,\n",
    "    interpolation=\"bilinear\",\n",
    "    follow_links=False,\n",
    "    crop_to_aspect_ratio=False,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5a429f49",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<_PrefetchDataset element_spec=(TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 3), dtype=tf.float32, name=None))>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "25818139",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tf.Tensor(\n",
      "[[[[209. 207. 218.]\n",
      "   [219. 217. 228.]\n",
      "   [132. 130. 141.]\n",
      "   ...\n",
      "   [154. 154. 152.]\n",
      "   [147. 146. 152.]\n",
      "   [ 54.  53.  49.]]\n",
      "\n",
      "  [[103. 102. 110.]\n",
      "   [141. 140. 148.]\n",
      "   [136. 135. 143.]\n",
      "   ...\n",
      "   [ 75.  76.  70.]\n",
      "   [144. 157. 137.]\n",
      "   [172. 196. 170.]]\n",
      "\n",
      "  [[ 94.  93.  99.]\n",
      "   [133. 132. 138.]\n",
      "   [133. 132. 138.]\n",
      "   ...\n",
      "   [157. 184. 153.]\n",
      "   [172. 211. 158.]\n",
      "   [195. 239. 204.]]\n",
      "\n",
      "  ...\n",
      "\n",
      "  [[122. 121. 116.]\n",
      "   [ 74.  73.  69.]\n",
      "   [226. 225. 223.]\n",
      "   ...\n",
      "   [180. 197. 155.]\n",
      "   [223. 255. 201.]\n",
      "   [224. 247. 201.]]\n",
      "\n",
      "  [[ 14.  10.   9.]\n",
      "   [ 19.  15.  16.]\n",
      "   [169. 164. 170.]\n",
      "   ...\n",
      "   [214. 239. 197.]\n",
      "   [204. 247. 201.]\n",
      "   [207. 246. 202.]]\n",
      "\n",
      "  [[  0.  10.   0.]\n",
      "   [ 60.  73.  63.]\n",
      "   [185. 198. 191.]\n",
      "   ...\n",
      "   [184. 195. 161.]\n",
      "   [220. 253. 224.]\n",
      "   [222. 253. 222.]]]\n",
      "\n",
      "\n",
      " [[[210. 211. 206.]\n",
      "   [128. 129. 124.]\n",
      "   [210. 211. 206.]\n",
      "   ...\n",
      "   [  1.   0.   2.]\n",
      "   [  3.  34.   3.]\n",
      "   [ 51. 115.  54.]]\n",
      "\n",
      "  [[216. 217. 212.]\n",
      "   [214. 215. 210.]\n",
      "   [210. 211. 206.]\n",
      "   ...\n",
      "   [  7.   0.   1.]\n",
      "   [ 63.  78.  59.]\n",
      "   [ 71. 112.  70.]]\n",
      "\n",
      "  [[197. 198. 193.]\n",
      "   [215. 216. 211.]\n",
      "   [212. 213. 208.]\n",
      "   ...\n",
      "   [ 16.  12.  11.]\n",
      "   [ 95. 105.  94.]\n",
      "   [ 99. 124. 102.]]\n",
      "\n",
      "  ...\n",
      "\n",
      "  [[ 93.  89.  86.]\n",
      "   [173. 169. 166.]\n",
      "   [ 75.  71.  68.]\n",
      "   ...\n",
      "   [  0.   0.   0.]\n",
      "   [  0.   0.   0.]\n",
      "   [  0.   0.   0.]]\n",
      "\n",
      "  [[148. 144. 141.]\n",
      "   [143. 139. 136.]\n",
      "   [ 64.  60.  57.]\n",
      "   ...\n",
      "   [  0.   0.   0.]\n",
      "   [  0.   0.   0.]\n",
      "   [  0.   0.   0.]]\n",
      "\n",
      "  [[166. 162. 159.]\n",
      "   [140. 136. 133.]\n",
      "   [163. 159. 156.]\n",
      "   ...\n",
      "   [  0.   0.   0.]\n",
      "   [  0.   0.   0.]\n",
      "   [  0.   0.   0.]]]\n",
      "\n",
      "\n",
      " [[[194. 226. 163.]\n",
      "   [194. 247. 175.]\n",
      "   [175. 219. 132.]\n",
      "   ...\n",
      "   [170. 222. 160.]\n",
      "   [174. 222. 162.]\n",
      "   [151. 196. 137.]]\n",
      "\n",
      "  [[161. 204. 124.]\n",
      "   [187. 255. 154.]\n",
      "   [189. 254. 138.]\n",
      "   ...\n",
      "   [167. 213. 151.]\n",
      "   [159. 200. 140.]\n",
      "   [183. 223. 163.]]\n",
      "\n",
      "  [[214. 242. 183.]\n",
      "   [178. 237. 145.]\n",
      "   [181. 240. 132.]\n",
      "   ...\n",
      "   [180. 231. 165.]\n",
      "   [177. 226. 161.]\n",
      "   [163. 209. 145.]]\n",
      "\n",
      "  ...\n",
      "\n",
      "  [[ 56.  56.  56.]\n",
      "   [ 56.  56.  56.]\n",
      "   [ 56.  56.  56.]\n",
      "   ...\n",
      "   [153. 155. 141.]\n",
      "   [123. 125. 111.]\n",
      "   [137. 139. 125.]]\n",
      "\n",
      "  [[ 56.  56.  56.]\n",
      "   [ 56.  56.  56.]\n",
      "   [ 56.  56.  56.]\n",
      "   ...\n",
      "   [145. 146. 140.]\n",
      "   [ 87.  88.  82.]\n",
      "   [ 90.  91.  85.]]\n",
      "\n",
      "  [[ 56.  56.  56.]\n",
      "   [ 56.  56.  56.]\n",
      "   [ 56.  56.  56.]\n",
      "   ...\n",
      "   [ 52.  52.  50.]\n",
      "   [ 54.  54.  52.]\n",
      "   [ 56.  56.  54.]]]\n",
      "\n",
      "\n",
      " ...\n",
      "\n",
      "\n",
      " [[[137. 137. 127.]\n",
      "   [ 99.  99.  91.]\n",
      "   [ 58.  58.  48.]\n",
      "   ...\n",
      "   [240. 255. 172.]\n",
      "   [251. 246. 214.]\n",
      "   [251. 255. 233.]]\n",
      "\n",
      "  [[169. 171. 157.]\n",
      "   [112. 114. 101.]\n",
      "   [ 41.  41.  41.]\n",
      "   ...\n",
      "   [152. 185.  96.]\n",
      "   [247. 255. 196.]\n",
      "   [245. 251. 205.]]\n",
      "\n",
      "  [[131. 138. 122.]\n",
      "   [131. 140. 111.]\n",
      "   [ 60.  67.  51.]\n",
      "   ...\n",
      "   [197. 228. 152.]\n",
      "   [225. 255. 169.]\n",
      "   [230. 251. 184.]]\n",
      "\n",
      "  ...\n",
      "\n",
      "  [[ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   ...\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]]\n",
      "\n",
      "  [[ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   ...\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]]\n",
      "\n",
      "  [[ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   ...\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]\n",
      "   [ 26.  26.  26.]]]\n",
      "\n",
      "\n",
      " [[[ 33.  44.  10.]\n",
      "   [ 57.  65.  41.]\n",
      "   [ 28.  30.  17.]\n",
      "   ...\n",
      "   [ 81. 155.  30.]\n",
      "   [ 75. 144.  28.]\n",
      "   [ 71. 136.  16.]]\n",
      "\n",
      "  [[ 59.  67.  52.]\n",
      "   [ 23.  29.  19.]\n",
      "   [ 49.  49.  47.]\n",
      "   ...\n",
      "   [ 84. 156.  28.]\n",
      "   [ 72. 141.  24.]\n",
      "   [ 66. 136.  14.]]\n",
      "\n",
      "  [[ 63.  70.  62.]\n",
      "   [ 43.  48.  42.]\n",
      "   [ 50.  50.  48.]\n",
      "   ...\n",
      "   [ 74. 139.  23.]\n",
      "   [ 83. 155.  30.]\n",
      "   [102. 179.  37.]]\n",
      "\n",
      "  ...\n",
      "\n",
      "  [[101. 156.  91.]\n",
      "   [ 80. 114.  63.]\n",
      "   [ 43.  87.  34.]\n",
      "   ...\n",
      "   [151. 187. 123.]\n",
      "   [162. 182. 133.]\n",
      "   [150. 170. 119.]]\n",
      "\n",
      "  [[118. 205.  90.]\n",
      "   [107. 159.  93.]\n",
      "   [164. 233. 144.]\n",
      "   ...\n",
      "   [189. 221. 156.]\n",
      "   [147. 189. 117.]\n",
      "   [132. 178. 103.]]\n",
      "\n",
      "  [[134. 220. 109.]\n",
      "   [151. 241. 127.]\n",
      "   [ 58. 106.  46.]\n",
      "   ...\n",
      "   [138. 169. 112.]\n",
      "   [142. 186. 109.]\n",
      "   [129. 170.  92.]]]\n",
      "\n",
      "\n",
      " [[[152. 187. 105.]\n",
      "   [136. 167.  99.]\n",
      "   [114. 136.  87.]\n",
      "   ...\n",
      "   [209. 222. 194.]\n",
      "   [157. 167. 143.]\n",
      "   [151. 185. 111.]]\n",
      "\n",
      "  [[117. 176.  70.]\n",
      "   [106. 147.  71.]\n",
      "   [ 89. 110.  67.]\n",
      "   ...\n",
      "   [242. 255. 216.]\n",
      "   [207. 242. 162.]\n",
      "   [215. 255. 152.]]\n",
      "\n",
      "  [[ 86. 154.  55.]\n",
      "   [120. 159.  92.]\n",
      "   [ 51.  58.  25.]\n",
      "   ...\n",
      "   [239. 240. 209.]\n",
      "   [183. 230. 126.]\n",
      "   [209. 255. 156.]]\n",
      "\n",
      "  ...\n",
      "\n",
      "  [[143. 144.  86.]\n",
      "   [201. 204. 151.]\n",
      "   [255. 255. 233.]\n",
      "   ...\n",
      "   [ 43.  43.  43.]\n",
      "   [ 43.  43.  43.]\n",
      "   [ 43.  43.  43.]]\n",
      "\n",
      "  [[161. 166. 102.]\n",
      "   [255. 255. 240.]\n",
      "   [255. 255. 250.]\n",
      "   ...\n",
      "   [ 43.  43.  43.]\n",
      "   [ 43.  43.  43.]\n",
      "   [ 43.  43.  43.]]\n",
      "\n",
      "  [[255. 255. 230.]\n",
      "   [253. 252. 250.]\n",
      "   [254. 255. 248.]\n",
      "   ...\n",
      "   [ 43.  43.  43.]\n",
      "   [ 43.  43.  43.]\n",
      "   [ 43.  43.  43.]]]], shape=(32, 128, 128, 3), dtype=float32) (32, 128, 128, 3)\n",
      "tf.Tensor(\n",
      "[[0. 0. 1.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]\n",
      " [0. 1. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]\n",
      " [1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]\n",
      " [1. 0. 0.]\n",
      " [1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]\n",
      " [0. 0. 1.]\n",
      " [1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]\n",
      " [0. 0. 1.]\n",
      " [1. 0. 0.]\n",
      " [1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]\n",
      " [0. 0. 1.]\n",
      " [0. 0. 1.]\n",
      " [1. 0. 0.]\n",
      " [1. 0. 0.]\n",
      " [0. 1. 0.]], shape=(32, 3), dtype=float32) (32, 3)\n"
     ]
    }
   ],
   "source": [
    "for x,y in training_set:\n",
    "    print(x,x.shape)\n",
    "    print(y,y.shape)\n",
    "    break"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e4b6ba4",
   "metadata": {},
   "source": [
    "# Building Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8f6a3ab6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout\n",
    "from tensorflow.keras.models import Sequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "42f723e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c1612209",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Bulding Cobvolution Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5c510a89",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\KIIT\\anaconda3\\lib\\site-packages\\keras\\src\\layers\\convolutional\\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))\n",
    "model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=2,strides=2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4bb3842b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))\n",
    "model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=2,strides=2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "08e298a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))\n",
    "model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=2,strides=2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c2ade540",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))\n",
    "model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=2,strides=2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "76d99c19",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))\n",
    "model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=2,strides=2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6bc403b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dropout(0.25))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f3c404de",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "acc8e115",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.regularizers import l2\n",
    "\n",
    "\n",
    "model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "86e86de6",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dropout(0.4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "aca6c922",
   "metadata": {},
   "outputs": [],
   "source": [
    "#output layer\n",
    "model.add(Dense(units=3,activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2b7fab91",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.environ['TF_USE_LEGACY_KERAS'] = 'True'\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e73cfe2c",
   "metadata": {},
   "source": [
    "# Compilling Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "ea02f7bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),\n",
    "              loss='categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "e4ca155a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)        │             <span style=\"color: #00af00; text-decoration-color: #00af00\">896</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">126</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">126</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)        │           <span style=\"color: #00af00; text-decoration-color: #00af00\">9,248</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">63</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │          <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">61</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">61</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │          <span style=\"color: #00af00; text-decoration-color: #00af00\">36,928</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">30</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)         │          <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_5 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)         │         <span style=\"color: #00af00; text-decoration-color: #00af00\">147,584</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)         │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_6 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">14</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)         │         <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_7 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">12</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">12</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)         │         <span style=\"color: #00af00; text-decoration-color: #00af00\">590,080</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)           │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_8 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)           │       <span style=\"color: #00af00; text-decoration-color: #00af00\">1,180,160</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_9 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)           │       <span style=\"color: #00af00; text-decoration-color: #00af00\">2,359,808</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)           │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)           │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2048</span>)                │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)                 │         <span style=\"color: #00af00; text-decoration-color: #00af00\">524,544</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)                 │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)                   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">771</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m, \u001b[38;5;34m128\u001b[0m, \u001b[38;5;34m32\u001b[0m)        │             \u001b[38;5;34m896\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m126\u001b[0m, \u001b[38;5;34m126\u001b[0m, \u001b[38;5;34m32\u001b[0m)        │           \u001b[38;5;34m9,248\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m32\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m63\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │          \u001b[38;5;34m18,496\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_3 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m61\u001b[0m, \u001b[38;5;34m61\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │          \u001b[38;5;34m36,928\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_4 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m30\u001b[0m, \u001b[38;5;34m128\u001b[0m)         │          \u001b[38;5;34m73,856\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_5 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m28\u001b[0m, \u001b[38;5;34m128\u001b[0m)         │         \u001b[38;5;34m147,584\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m128\u001b[0m)         │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_6 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m14\u001b[0m, \u001b[38;5;34m256\u001b[0m)         │         \u001b[38;5;34m295,168\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_7 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m12\u001b[0m, \u001b[38;5;34m12\u001b[0m, \u001b[38;5;34m256\u001b[0m)         │         \u001b[38;5;34m590,080\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_3 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m256\u001b[0m)           │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_8 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m6\u001b[0m, \u001b[38;5;34m512\u001b[0m)           │       \u001b[38;5;34m1,180,160\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_9 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m512\u001b[0m)           │       \u001b[38;5;34m2,359,808\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_4 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m512\u001b[0m)           │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout (\u001b[38;5;33mDropout\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m512\u001b[0m)           │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2048\u001b[0m)                │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m)                 │         \u001b[38;5;34m524,544\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m)                 │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m3\u001b[0m)                   │             \u001b[38;5;34m771\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">5,237,539</span> (19.98 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m5,237,539\u001b[0m (19.98 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">5,237,539</span> (19.98 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m5,237,539\u001b[0m (19.98 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "087347aa",
   "metadata": {},
   "source": [
    "## MODEL TRAINING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "ee289a85",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m42s\u001b[0m 998ms/step - accuracy: 0.4032 - loss: 5.8742 - val_accuracy: 0.2750 - val_loss: 5.1589\n",
      "Epoch 2/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m46s\u001b[0m 1s/step - accuracy: 0.4235 - loss: 5.0421 - val_accuracy: 0.5100 - val_loss: 4.6415\n",
      "Epoch 3/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m46s\u001b[0m 1s/step - accuracy: 0.4803 - loss: 4.6121 - val_accuracy: 0.5450 - val_loss: 4.2576\n",
      "Epoch 4/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 1s/step - accuracy: 0.4454 - loss: 4.2250 - val_accuracy: 0.4100 - val_loss: 3.9762\n",
      "Epoch 5/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 1s/step - accuracy: 0.4833 - loss: 3.8939 - val_accuracy: 0.2900 - val_loss: 3.7695\n",
      "Epoch 6/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 1s/step - accuracy: 0.4777 - loss: 3.5970 - val_accuracy: 0.4350 - val_loss: 3.3542\n",
      "Epoch 7/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 1s/step - accuracy: 0.4877 - loss: 3.3305 - val_accuracy: 0.4200 - val_loss: 3.1969\n",
      "Epoch 8/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 1s/step - accuracy: 0.5172 - loss: 3.0764 - val_accuracy: 0.4400 - val_loss: 2.9180\n",
      "Epoch 9/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 1s/step - accuracy: 0.5058 - loss: 2.8837 - val_accuracy: 0.3000 - val_loss: 3.0200\n",
      "Epoch 10/10\n",
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 1s/step - accuracy: 0.5104 - loss: 2.7411 - val_accuracy: 0.3550 - val_loss: 2.6718\n"
     ]
    }
   ],
   "source": [
    "training_history=model.fit(x=training_set,validation_data=validation_set,epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7632cfd4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "e91b210e",
   "metadata": {},
   "source": [
    "## MODEL EVALUATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f939076e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m39/39\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 223ms/step - accuracy: 0.5535 - loss: 2.5545\n",
      "Training accuracy: 0.5542071461677551\n"
     ]
    }
   ],
   "source": [
    "#Training set Accuracy\n",
    "train_loss, train_acc = model.evaluate(training_set)\n",
    "print('Training accuracy:', train_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "86bc6143",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m7/7\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 188ms/step - accuracy: 0.3212 - loss: 2.6946\n",
      "Validation accuracy: 0.35499998927116394\n"
     ]
    }
   ],
   "source": [
    "#Validation set Accuracy\n",
    "val_loss, val_acc = model.evaluate(validation_set)\n",
    "print('Validation accuracy:', val_acc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf236615",
   "metadata": {},
   "source": [
    "# SAVING MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "7fe2b0c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('trained_plant_disease_model.keras')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "828b8f0b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'accuracy': [0.3923948109149933,\n",
       "  0.4279935359954834,\n",
       "  0.45388349890708923,\n",
       "  0.4563106894493103,\n",
       "  0.48543688654899597,\n",
       "  0.47653722763061523,\n",
       "  0.491909384727478,\n",
       "  0.5064724683761597,\n",
       "  0.5105177760124207,\n",
       "  0.5177993774414062],\n",
       " 'loss': [5.550051689147949,\n",
       "  4.94712495803833,\n",
       "  4.528512477874756,\n",
       "  4.154115200042725,\n",
       "  3.8225958347320557,\n",
       "  3.537806749343872,\n",
       "  3.274664878845215,\n",
       "  3.0392541885375977,\n",
       "  2.8498735427856445,\n",
       "  2.686023235321045],\n",
       " 'val_accuracy': [0.2750000059604645,\n",
       "  0.5099999904632568,\n",
       "  0.5450000166893005,\n",
       "  0.4099999964237213,\n",
       "  0.28999999165534973,\n",
       "  0.4350000023841858,\n",
       "  0.41999998688697815,\n",
       "  0.4399999976158142,\n",
       "  0.30000001192092896,\n",
       "  0.35499998927116394],\n",
       " 'val_loss': [5.158854961395264,\n",
       "  4.641493797302246,\n",
       "  4.257646560668945,\n",
       "  3.976235866546631,\n",
       "  3.7694733142852783,\n",
       "  3.354166269302368,\n",
       "  3.1968741416931152,\n",
       "  2.918027400970459,\n",
       "  3.0200390815734863,\n",
       "  2.671832323074341]}"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_history.history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "b0c8d01d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Recording History in json\n",
    "import json\n",
    "with open('training_hist.json','w') as f:\n",
    "  json.dump(training_history.history,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "fa205ee2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss'])\n"
     ]
    }
   ],
   "source": [
    "print(training_history.history.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "79d25894",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.3923948109149933,\n",
       " 0.4279935359954834,\n",
       " 0.45388349890708923,\n",
       " 0.4563106894493103,\n",
       " 0.48543688654899597,\n",
       " 0.47653722763061523,\n",
       " 0.491909384727478,\n",
       " 0.5064724683761597,\n",
       " 0.5105177760124207,\n",
       " 0.5177993774414062]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_history.history['accuracy']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80bdef5f",
   "metadata": {},
   "source": [
    "#  Accuracy Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "efb35e41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "epochs = range(1, min(len(training_history.history['accuracy']) + 1,\n",
    "                     len(training_history.history['val_accuracy']) + 1))\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(epochs, training_history.history['accuracy'][:len(epochs)], \n",
    "         color='red', label='Training Accuracy')\n",
    "plt.plot(epochs, training_history.history['val_accuracy'][:len(epochs)], \n",
    "         color='blue', label='Validation Accuracy')\n",
    "plt.xlabel('No. of Epochs')\n",
    "plt.title('Visualization of Accuracy Result')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4e7d137",
   "metadata": {},
   "source": [
    "## Some other metrics for model evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "ef9ac29f",
   "metadata": {},
   "outputs": [],
   "source": [
    "class_name = validation_set.class_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "72447a09",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 200 files belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "test_set = tf.keras.utils.image_dataset_from_directory(\n",
    "    'valid',\n",
    "    labels=\"inferred\",\n",
    "    label_mode=\"categorical\",\n",
    "    class_names=None,\n",
    "    color_mode=\"rgb\",\n",
    "    batch_size=1,\n",
    "    image_size=(128, 128),\n",
    "    shuffle=False,\n",
    "    seed=None,\n",
    "    validation_split=None,\n",
    "    subset=None,\n",
    "    interpolation=\"bilinear\",\n",
    "    follow_links=False,\n",
    "    crop_to_aspect_ratio=False\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "7ebcefc3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m200/200\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 16ms/step\n"
     ]
    }
   ],
   "source": [
    "y_pred = model.predict(test_set)\n",
    "predicted_categories = tf.argmax(y_pred, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "778a9e96",
   "metadata": {},
   "outputs": [],
   "source": [
    "true_categories = tf.concat([y for x, y in test_set], axis=0)\n",
    "Y_true = tf.argmax(true_categories, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d7b68ad9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(200,), dtype=int64, numpy=\n",
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2], dtype=int64)>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_true"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "8e0f31f4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(200,), dtype=int64, numpy=\n",
       "array([2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 0, 1, 2, 2, 2, 2, 0, 2, 2,\n",
       "       0, 2, 2, 2, 1, 0, 2, 0, 2, 1, 2, 2, 0, 0, 2, 2, 0, 0, 0, 2, 0, 2,\n",
       "       0, 0, 2, 0, 2, 2, 2, 2, 1, 2, 0, 0, 2, 0, 2, 2, 2, 0, 2, 0, 1, 2,\n",
       "       2, 1, 0, 2, 2, 2, 2, 0, 2, 2, 1, 2, 1, 2, 0, 2, 2, 1, 2, 2, 2, 2,\n",
       "       2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 0, 1, 0, 2, 0,\n",
       "       2, 2, 2, 0, 2, 1, 2, 0, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1,\n",
       "       2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 0, 0, 2, 2,\n",
       "       2, 2, 1, 0, 1, 1, 2, 1, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 0, 2, 1, 1,\n",
       "       0, 1, 2, 2, 1, 0, 1, 2, 2, 0, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,\n",
       "       1, 1], dtype=int64)>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predicted_categories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "676909df",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix,classification_report\n",
    "cm = confusion_matrix(Y_true,predicted_categories)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "54568d9f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "     class_0       0.79      0.27      0.40       121\n",
      "     class_1       0.27      0.42      0.33        26\n",
      "     class_2       0.23      0.51      0.32        53\n",
      "\n",
      "    accuracy                           0.35       200\n",
      "   macro avg       0.43      0.40      0.35       200\n",
      "weighted avg       0.57      0.35      0.37       200\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Precision Recall Fscore\n",
    "print(classification_report(Y_true,predicted_categories,target_names=class_name))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d67be1d8",
   "metadata": {},
   "source": [
    "# Confusion Matrix Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "b4d49bcd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 300x300 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(3, 3))\n",
    "sns.heatmap(cm,annot=True,annot_kws={\"size\": 10})\n",
    "\n",
    "plt.xlabel('Predicted Class',fontsize = 10)\n",
    "plt.ylabel('Actual Class',fontsize = 10)\n",
    "plt.title('Cofee Disease Prediction Confusion Matrix',fontsize = 15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "e4b0027f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['class_0', 'class_1', 'class_2']"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class_name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af7bdae7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9dc57de",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d219fadf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
