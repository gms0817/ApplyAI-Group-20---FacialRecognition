import os
import random
from skimage import io
from skimage.transform import rotate
from numpy import fliplr, flipud

class FacialRecognition:
    def __init__(self):
        self.dataset_path = './test_set/'
        self.seen_files = []
        self.photo_cap = 20
        self.num_of_pics = 0

    def load_dataset(self):
        print('Reached process_images()')
        print('Starting Image Pre-Processing...')

        # Load images recursively from directory
        for subdir, dirs, files in os.walk(self.dataset_path):  # Iterate through folders
            self.num_of_pics = 0  # Set counter variable to 0 and track how many pics each person has
            for file in files:  # iterate through files in folders
                file_type = file[-4:]  # Extract the file type
                aug_count = 0
                while self.num_of_pics < self.photo_cap:
                    aug_count = aug_count + 1  # Increment augmentation count
                    self.seen_files.clear()  # Reset seen list as needed to continue augmentation
                    if file_type == '.jpg':
                        if self.num_of_pics < self.photo_cap and file not in self.seen_files:  # Only augment pictures if under photo cap
                            self.num_of_pics = self.num_of_pics + 1
                            self.seen_files.append(file)  # Add current file to seen files list

                            # Provide update to terminal
                            print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, file)}')

                            # Read the image from the file as grayscale
                            im = io.imread(os.path.join(subdir, file), as_gray=True)

                            # Create augmented photos based on need
                            self.augment_image(im, subdir, file, aug_count)

            print('Finished Image Pre-Processing.')

    def augment_image(self, im, subdir, file, aug_count):
        print('Reached augment_images()')
        # Randomly rotate image
        random_angle = random.randint(-60, 60)
        if self.num_of_pics < self.photo_cap:
            self.num_of_pics = self.num_of_pics + 1
            im_rotated = rotate(im, angle=random_angle, resize=False)  # Rotate the image

            filepath = os.path.join(subdir, file[:4] + f'-rotated_-{random_angle}degrees-{aug_count}.jpg')  # Configure new filepath
            if not os.path.exists(filepath):
                io.imsave(filepath, im_rotated)  # Save the image
            self.seen_files.append(filepath)  # Update the seen files list
            print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, filepath)}')

            # Flip image horizontally and vertically
            if self.num_of_pics < self.photo_cap:
                self.num_of_pics = self.num_of_pics + 2
                hor_im = fliplr(im_rotated)  # Flip horizontally
                ver_im = flipud(im_rotated)  # Flip verticaly

                hor_filepath = os.path.join(subdir, file[:4] + f'-horizontal-flip-{aug_count}.jpg')  # Configure new filepath
                ver_filepath = os.path.join(subdir, file[:4] + f'-vertical-flip-{aug_count}.jpg')  # Configure new filepath

                if not os.path.exists(hor_filepath):
                    io.imsave(hor_filepath, hor_im)  # Save the horizontally flipped im
                if not os.path.exists(ver_filepath):
                    io.imsave(ver_filepath, ver_im)  # Save the vertically flipped im

                self.seen_files.append(hor_filepath)  # Update the seen files list
                print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, hor_filepath)}')

                self.seen_files.append(ver_filepath)  # Update the seen files list
                print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, ver_filepath)}')
                print(f'num_of_pics: {self.num_of_pics} - photo_cap: {self.photo_cap}')





def main():
    print('Reached main()')


if __name__ == '__main__':
    print('Launching program.')
    app = FacialRecognition()

    # Load dataset
    app.load_dataset()
