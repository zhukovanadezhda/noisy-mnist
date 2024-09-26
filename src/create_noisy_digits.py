import keras
import numpy as np

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train = 0

if train:
    images = train_images
    labels = train_labels
    img_num = 60000
else:
    images = test_images
    labels = test_labels
    img_num = 10000

images = images.astype('float32')
images = images / 255
images = images.reshape(img_num,28,28,1)

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomTranslation(0.2, 0.2),
        keras.layers.RandomZoom((0.2, 0.2)),
        keras.layers.RandomBrightness((-0.2, 0.2),[0.0, 1.0])
    ]
)

augmented_images = data_augmentation(images)

if 0:
    plt.figure(figsize=[10,10])
    plt.subplot(1, 2, 1)
    plimage = images[5]*255
    plimage = plimage.astype("uint8")
    plimage = plimage.reshape(28,28,1)
    plt.imshow(plimage, cmap=plt.cm.binary)
    plt.subplot(1, 2, 2)
    plimage = augmented_images[5]*255
    plimage = plimage.numpy().astype("uint8")
    plimage = plimage.reshape(28,28,1)
    plt.imshow(plimage, cmap=plt.cm.binary)
    plt.show()

augmented_images = augmented_images.reshape(img_num,28,28) #??
augmented_images = augmented_images*255
augmented_images = augmented_images.numpy().astype("uint8")

if train:
    np.save("train_images", augmented_images)
    np.save("train_labels", labels)
else:
    np.save("test_images", augmented_images)
    np.save("test_labels", labels)

#
#
#
import numpy as np
train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")
