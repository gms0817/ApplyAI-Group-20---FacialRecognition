import os


def main():
    total_pics = 0
    target_perc = 1.0 - 0.8
    path = "./lfw_dataset/lfw_train/"

    # Load images recursively from directory
    for subdir, dirs, files in os.walk(path):
        num_deleted = 0
        num_of_pics = len(files)  # get initial number of files
        num_to_delete = int(num_of_pics * target_perc)

        # Increment total num of pics
        total_pics = total_pics + num_of_pics

        # Delete files based on target percentage given
        for file in files:
            path = subdir + '/' + file
            if num_to_delete > num_deleted:
                os.remove(path)
                num_deleted = num_deleted + 1
                total_pics = total_pics - 1
            else:
                break
            print(f'Subdir: {subdir} | To-Delete: {num_to_delete} | Deleted: {num_deleted} | Total Pics: {total_pics}')


if __name__ == '__main__':
    print('Launching program.')
    main()
