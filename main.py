import os


class FacialRecognition:
    def __init__(self):
        self.dataset_path = './lfw_dataset/'

        #

    def load_dataset(self):
        print('Reached process_images()')
        print('Starting Image Pre-Processing...')

        # Load images recursively from directory
        for subdir, dirs, files in os.walk(self.dataset_path):  # Iterate through folders
            num_of_pics = 0  # Set counter variable to 0 and track how many pics each person has
            for file in files:  # iterate through files in folders
                num_of_pics = num_of_pics + 1
                print(f'Number of Pictures: {num_of_pics} - {os.path.join(subdir, file)}')

                # Create augmented photos based on need - aim for 10 pics per person

                # (TBD) Isolate the face and overwrite previous image

        print('Finished Image Pre-Processing.')

    def augment_images(self):
        print('Reached augment_images()')


def main():
    print('Reached main()')


if __name__ == '__main__':
    print('Launching program.')
    app = FacialRecognition()

    # Load dataset
    app.load_dataset()
