import os
import random
from skimage import io
from skimage.transform import rotate
from skimage import exposure
from skimage.util import img_as_ubyte
from numpy import fliplr, flipud


class FacialRecognition:
    def __init__(self):
        self.dataset_path = './test_dataset/'
        self.photo_cap = 100
        self.num_of_pics = 0  # Track pics across single directory
        self.total_pics = 0  # Track total pics across all directories
        self.color_mode = 'grayscale'

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
                        os.remove(file)
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

    def augment_image(self, im, subdir,):
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


def main():
    print('Reached main()')


if __name__ == '__main__':
    print('Launching program.')
    app = FacialRecognition()

    # Load dataset
    app.load_dataset()