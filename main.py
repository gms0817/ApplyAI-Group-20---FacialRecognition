import os
from skimage import io
from skimage.transform import rotate

class FacialRecognition:
    def __init__(self):
        self.dataset_path = './test_set/'
        self.seen_files = []
        self.photo_cap = 10
        self.num_of_pics = 0

    def load_dataset(self):
        print('Reached process_images()')
        print('Starting Image Pre-Processing...')

        # Load images recursively from directory
        for subdir, dirs, files in os.walk(self.dataset_path):  # Iterate through folders
            self.num_of_pics = 0  # Set counter variable to 0 and track how many pics each person has
            for file in files:  # iterate through files in folders
                file_type = file[-4:]  # Extract the file type
                if file_type == '.jpg':
                    if self.num_of_pics <= self.photo_cap and file not in self.seen_files:  # Only augment pictures if under 10 photos for person
                        self.num_of_pics = self.num_of_pics + 1
                        self.seen_files.append(file)  # Add current file to seen files list

                        print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, file)}')

                        # Read the image from the file as grayscale
                        im = io.imread(os.path.join(subdir, file), as_gray=True)

                        # Create augmented photos based on need - aim for 10 pics per person
                        self.augment_image(im, subdir, file)

                        # (TBD) Isolate the face and overwrite previous image
        print('Finished Image Pre-Processing.')

    def augment_image(self, im, subdir, file):
        print('Reached augment_images()')
        if self.num_of_pics <= self.photo_cap:
            # Rotate image -5 degrees
            self.num_of_pics = self.num_of_pics + 1
            im_rotated = rotate(im, angle=-5, resize=False)  # Rotate the image

            filepath = os.path.join(subdir, file[:4] + '-rotated_-5.jpg')  # Configure new filepath
            io.imsave(filepath, im_rotated)  # Save the image
            self.seen_files.append(filepath)  # Update the seen files list

            print(f'Number of Pictures: {self.num_of_pics} - {os.path.join(subdir, file)}')

            # Rotate image -15 degrees
            self.num_of_pics = self.num_of_pics + 1
            im_rotated = rotate(im, angle=-15, resize=False)  # Rotate the image

            filepath = os.path.join(subdir, file[:4] + '-rotated_-15.jpg')  # Configure new filepath
            io.imsave(filepath, im_rotated)  # Save the image
            self.seen_files.append(filepath)  # Update the seen files list

            # Rotate image 5 degrees
            self.num_of_pics = self.num_of_pics + 1
            im_rotated = rotate(im, angle=5, resize=False)  # Rotate the image

            filepath = os.path.join(subdir, file[:4] + '-rotated_5.jpg')  # Configure new filepath
            io.imsave(filepath, im_rotated)  # Save the image
            self.seen_files.append(filepath)  # Update the seen files list

            # Rotate image 15 degrees
            self.num_of_pics = self.num_of_pics + 1
            im_rotated = rotate(im, angle=15, resize=False)  # Rotate the image

            filepath = os.path.join(subdir, file[:4] + '-rotated_15.jpg')  # Configure new filepath
            io.imsave(filepath, im_rotated)  # Save the image
            self.seen_files.append(filepath)  # Update the seen files list
            # Flip image horizontally

            # Flip image vertically



def main():
    print('Reached main()')


if __name__ == '__main__':
    print('Launching program.')
    app = FacialRecognition()

    # Load dataset
    app.load_dataset()
