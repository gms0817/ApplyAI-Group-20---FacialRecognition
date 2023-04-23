import os
import random
import tkinter as tk
import numpy as np
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
from keras.utils import load_img, img_to_array
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import rotate
from skimage import exposure
from skimage.util import img_as_ubyte
from numpy import fliplr, flipud
import tensorflow as tf
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Config tkinter window
        self.title('LFW - Facial Recognition')
        self.resizable(False, False)

        self.img_file = None
        self.filepath = ''

        self.image_frame = ttk.Frame(self, width=300, height=300)
        self.image_frame.pack(side="top", padx=10, pady=10)

        self.placeholder_im = Image.open('placeholder.jpg')
        self.placeholder_im = self.placeholder_im.resize((300, 300), Image.LANCZOS)
        self.placeholder_im = ImageTk.PhotoImage(self.placeholder_im)

        self.image_label = tk.Label(self.image_frame, image=self.placeholder_im)
        self.image_label.pack()

        self.prediction_label = ttk.Label(self, text="Select an image to upload:")
        self.prediction_label.pack(side="top", padx=10, pady=10)

        self.browse_button = ttk.Button(self, text="Browse", command=self.load_image)
        self.browse_button.pack(side="top", padx=10, pady=10)

        self.submit_button = ttk.Button(self, text="Submit", command=self.submit_image)
        self.submit_button.pack(side="bottom", padx=10, pady=10)

    def load_image(self):
        self.file_path = filedialog.askopenfilename(title="Select an image", filetypes=(
            ("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))

        # Load the image file
        im = Image.open(self.file_path)
        im = im.resize((300, 300), Image.LANCZOS)
        self.img_file = ImageTk.PhotoImage(im)

        self.image_label.config(image=self.img_file)

    def submit_image(self):
        print('Image Submitted!')

        # Format img input to match model
        pred_im_file = load_img(self.file_path)
        pred_im_file = pred_im_file.resize((128, 128), Image.LANCZOS)
        pred_im_file = img_to_array(pred_im_file)
        pred_im_file = pred_im_file.reshape(-1, 128, 128, 3)

        # Make prediction
        pred_prob = fr.model.predict(pred_im_file)
        pred_class = pred_prob.argmax(axis=1)[0]
        print(pred_class)
        pred = fr.classesList[pred_class]

        self.prediction_label.config(text=pred)


class FacialRecognition:
    def __init__(self):
        self.dataset_path = './lfw_dataset/'
        self.photo_cap = 100
        self.num_of_pics = 0  # Track pics across single directory
        self.total_pics = 0  # Track total pics across all directories
        self.color_mode = 'rgb'
        self.classesList = self.load_classes()
        self.model = self.load_model()

    def load_dataset(self):
        print('Reached process_images()')
        print('Starting Image Pre-Processing...')

        # Load images recursively from directory
        for subdir, dirs, files in os.walk(self.dataset_path):
            self.num_of_pics = len(files)  # get initial number of files

            # Increment total num of pics
            self.total_pics = self.total_pics + self.num_of_pics

            for file in files:
                filetype = file[-4:]
                if filetype == '.jpg':
                    im = None
                    if self.color_mode == 'grayscale':
                        # Read the image from the file as grayscale
                        im = io.imread(os.path.join(subdir, file), as_gray=True)

                        # Remove old version of file
                        os.remove(os.path.join(subdir, file))
                        self.num_of_pics = self.num_of_pics - 1

                        # extract root filename from subdir
                        root_filename = subdir[subdir.rindex('/') + 1:]

                        # construct new filepath
                        filepath = f'{subdir}/{root_filename}-{self.num_of_pics}.jpg'

                        # Convert file to ubyte format
                        im = img_as_ubyte(im)

                        # save new version of file
                        io.imsave(filepath, im)
                        self.num_of_pics = self.num_of_pics + 1

                    elif self.color_mode == 'rgb':
                        # Read the image from the file as grayscale
                        im = io.imread(os.path.join(subdir, file), as_gray=False)

                        # Remove old version of file
                        os.remove(os.path.join(subdir, file))
                        self.num_of_pics = self.num_of_pics - 1

                        # extract root filename from subdir
                        root_filename = subdir[subdir.rindex('/') + 1:]

                        # construct new filepath
                        filepath = f'{subdir}/{root_filename}-{self.num_of_pics}.jpg'

                        # Convert file to ubyte format
                        im = img_as_ubyte(im)

                        # save new version of file
                        io.imsave(filepath, im)
                        self.num_of_pics = self.num_of_pics + 1

                    else:
                        print('Invalid color mode. Set color mode to "grayscale" or "rgb"')

                    # augment pics as needed
                    while self.num_of_pics < self.photo_cap:
                        # Augment the image
                        self.augment_image(im, subdir)

            print(f'Number of Pics in {subdir}: {self.num_of_pics}')

        # Print final total number of pics
        print(f'New Dataset Size: {self.total_pics}')
        print('Finished Image Pre-Processing.')

    def augment_image(self, im, subdir, ):
        # extract root filename from subdir
        root_filename = subdir[subdir.rindex('/') + 1:]

        # construct new filepath
        filepath = f'{subdir}/{root_filename}-{self.num_of_pics}.jpg'

        print('Reached augment_images()')
        # Randomly rotate image
        random_angle = random.randint(-90, 90)
        if self.num_of_pics < self.photo_cap:
            im_rotated = rotate(im, angle=random_angle, resize=False)

            if not os.path.exists(filepath):
                # increment pic count
                self.num_of_pics = self.num_of_pics + 1
                self.total_pics = self.total_pics + 1

                # Convert file to ubyte format
                im_rotated = img_as_ubyte(im_rotated)

                # Save the image
                io.imsave(filepath, im_rotated)

            print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, filepath)}')

        # Adjust brightness based on random gamma value
        if self.num_of_pics < self.photo_cap:
            gamma_val = random.uniform(0.5, 1.5, )
            ex_im = exposure.adjust_gamma(im, gamma=gamma_val, gain=1)

            if not os.path.exists(filepath):
                # increment pic count
                self.num_of_pics = self.num_of_pics + 1
                self.total_pics = self.total_pics + 1

                # Convert file to ubyte format
                ex_im = img_as_ubyte(ex_im)

                # save the vertically flipped im
                io.imsave(filepath, ex_im)

            print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, filepath)}')

        # Flip image horizontally
        if self.num_of_pics < self.photo_cap:
            hor_im = fliplr(im)  # Flip horizontally

            # Check for existing file
            if not os.path.exists(filepath):
                # increment pic count
                self.num_of_pics = self.num_of_pics + 1
                self.total_pics = self.total_pics + 1

                # Convert file to ubyte format
                hor_im = img_as_ubyte(hor_im)

                # Save the horizontally flipped im
                io.imsave(filepath, hor_im)

            print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, filepath)}')

        # Flip image vertically
        if self.num_of_pics < self.photo_cap:
            ver_im = flipud(im)  # Flip vertically

            # Check for existing file
            if not os.path.exists(filepath):
                # increment pic count
                self.num_of_pics = self.num_of_pics + 1
                self.total_pics = self.total_pics + 1

                # Convert file to ubyte format
                ver_im = img_as_ubyte(ver_im)

                # Save the vertically flipped im
                io.imsave(filepath, ver_im)

            print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, filepath)}')

    def load_model(self):
        try:
            print('Attempting to load model...')
            self.model = load_model('lfw_model.h5')
            print('Model loaded.')
        except OSError:
            print('Model not found. Training new model...')
            # Create model if model isn't found
            # Load test and training data
            train_path = './lfw_dataset/lfw_train'
            test_path = './lfw_dataset/lfw_test'

            # CNN Parameters
            num_of_classes = 1000
            batch_size = 4

            train_batches = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
                directory=train_path, target_size=(128, 128),
                class_mode='categorical', batch_size=batch_size, shuffle=True)
            test_batches = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
                directory=test_path, target_size=(128, 128),
                class_mode='categorical', batch_size=batch_size, shuffle=True)

            # Design the CNN
            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
            model.add(MaxPool2D(pool_size=(2, 2), strides=2))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=2))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
            model.add(MaxPool2D(pool_size=(2, 2), strides=2))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.3))
            model.add(Dense(num_of_classes, activation="softmax"))

            # Compile the model
            epochs = 40
            model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy',
                          metrics=['accuracy'])
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00001)
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

            history = model.fit(train_batches, epochs=epochs, callbacks=[reduce_lr, early_stop],
                                validation_data=test_batches)

            # goto next batch of imgs
            imgs, labels = next(test_batches)

            # Evaluate the model
            scores = model.evaluate(imgs, labels, verbose=0)
            print('Succesfully trained model.')
            print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

            # Save the trained model
            print('Saving model...')
            model.save('lfw_model.h5')
            print('Model saved. Testing Model...')

            # Plot the model performance over epochs
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(epochs)

            plt.figure(figsize=(15, 15))
            plt.subplot(2, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

            print('LFW Recognition Model created and saved.')

            self.model = model
            return model

    def load_classes(self):
        classesList = []
        numberOfClasses = 0

        # Load filepaths recursively
        for subdir, dirs, files in os.walk(self.dataset_path):
            numberOfClasses = numberOfClasses + 1
            try:
                classesList.append(subdir[subdir.rindex("\\") + 1:])
            except ValueError:
                classesList.append(subdir)

        return classesList


if __name__ == '__main__':
    # Load model
    fr = FacialRecognition()
    fr.load_model()
    # fr.load_dataset() only use to augment data

    # Build GUI
    print('Launching program.')
    app = App()
    app.mainloop()
    print('Program closed.')
