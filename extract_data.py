import sys
import cv2
import os
import random
from skimage import io
import pandas as pd
from shutil import copyfile
from multiprocessing import Pool
import urllib

def download_jog(url, path):
    u = urllib.urlopen(url)
    f = open(path, 'wb')
    f.write(u.read())
    f.close()

def extract_data_openImages(base_path):
    """Extract K classes and N number of images
    from Open Images v4. Link to data files is
    given in the repo page.

    Download the images from URL, create train
    and test folders, copy the files in them
    create train_annotaions.txt and
    text_annotations.txt files.
    """
    images_boxable_fname = os.path.join(base_path,
                                        'train-images-boxable.csv')
    annotations_bbox_fname = os.path.join(base_path,
                                          'train-annotations-bbox.csv')
    class_description_fname = os.path.join(base_path,
                                           'class-descriptions-boxable.csv')

    print("Reading CSV files...")
    images_boxable = pd.read_csv(images_boxable_fname)
    annotations_bbox = pd.read_csv(annotations_bbox_fname)
    class_descriptions = pd.read_csv(class_description_fname,
                                     names=['name', 'class'])
    print("Done reading cvs.")
    # Find the label_name for 'Person', 'Mobile Phone' and 'Car' classes
    our_class = ['Person', 'Mobile phone', 'Car']

    keys = {}
    label_names = []
    bbox_keys = {}
    keys_img_url = {}
    sub_img_ids = {}
    N = 1000
    random.seed(10)

    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    for cls in our_class:
        keys[cls] = class_descriptions[
            class_descriptions['class'] == cls]
        label = keys[cls]['name'].values[0]
        print(label)
        bbox_keys[cls] = annotations_bbox[
            annotations_bbox['LabelName'] == label]
        sub_img_ids[cls] = bbox_keys[cls]['ImageID']\
                               .sample(frac=1)[:N]\
                                .map(lambda x: x+'.jpg')
        keys_img_url[cls] = images_boxable[images_boxable.image_name\
                                        .isin(sub_img_ids[cls])]
        urls = keys_img_url[cls].image_url.values

        # Download images
        print("Downloading images, this will take some time,"
              "Hold on tight...")
        # Create the directory
        saved_dir = os.path.join(base_path, cls)
        # pool = Pool()
        # pool.starmap(download_jog, [[url, saved_dir] for url in urls])
        if not os.path.isdir(saved_dir):
            os.mkdir(saved_dir)
        for url in urls:
            img = io.imread(url)
            saved_path = os.path.join(saved_dir,
                                      url[-20:])
            io.imsave(saved_path, img)
        print("Dowload completed!")

        print("Splitting data into train and test,"
              "moving them into folders")
        all_imgs = os.listdir(saved_dir)
        all_imgs = [f for f in all_imgs
                    if not f.startswith('.')]
        random.seed(1)
        random.shuffle(all_imgs)

        train_imgs = all_imgs[:800]
        test_imgs = all_imgs[800:]

        # Copy each classes' images to train directory
        for j in range(len(train_imgs)):
            original_path = os.path.join(saved_dir, train_imgs[j])
            new_path = os.path.join(train_path, train_imgs[j])
            copyfile(original_path, new_path)

        # Copy each classes' images to test directory
        for j in range(len(test_imgs)):
            original_path = os.path.join(saved_dir, test_imgs[j])
            new_path = os.path.join(test_path, test_imgs[j])
            copyfile(original_path, new_path)

        print("Done moving files")
        print('number of training images: ', len(os.listdir(train_path)))
        print('number of test images: ', len(os.listdir(test_path)))
        label_names.append(label)

    s = class_descriptions[class_descriptions['class'].isin(our_class)]\
            .set_index('name').T.to_dict()

    columns = ['FileName', 'XMin', 'XMax',
               'YMin', 'YMax',
               'ClassName']
    train_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(columns=columns)
    # Find boxes in each image and put them in a data frame
    train_imgs = os.listdir(train_path)
    train_imgs = [name[0:16] for name in train_imgs \
                  if not name.startswith('.')]

    df = annotations_bbox[annotations_bbox.ImageID \
        .isin(train_imgs)]
    df = df[df.LabelName.isin(label_names)]
    dd = df.groupby("ImageID")
    for ii in train_imgs:
        cc = dd.get_group(ii)
        for index, row in cc.iterrows():
            train_df = train_df.append({'FileName': '{}.jpg' \
                                       .format(ii),
                                        'XMin': row['XMin'],
                                        'XMax': row['XMax'],
                                        'YMin': row['YMin'],
                                        'YMax': row['YMax'],
                                        'ClassName': \
                                            s[row['LabelName']]['class']},
                                       ignore_index=True)

    test_imgs = os.listdir(test_path)
    test_imgs = [name[0:16] for name in test_imgs \
                  if not name.startswith('.')]

    df = annotations_bbox[annotations_bbox.ImageID \
        .isin(test_imgs)]
    df = df[df.LabelName.isin(label_names)]
    dd = df.groupby("ImageID")
    for ii in test_imgs:
        cc = dd.get_group(ii)
        for index, row in cc.iterrows():
            test_df = test_df.append({'FileName': '{}.jpg' \
                                       .format(ii),
                                        'XMin': row['XMin'],
                                        'XMax': row['XMax'],
                                        'YMin': row['YMin'],
                                        'YMax': row['YMax'],
                                        'ClassName': \
                                            s[row['LabelName']]['class']},
                                       ignore_index=True)

    # For training
    f = open(base_path + "/train_annotation.txt", "w+")
    for idx, row in train_df.iterrows():
        img = cv2.imread((base_path + '/train/' + row['FileName']))
        height, width = img.shape[:2]
        x1 = int(row['XMin'] * width)
        x2 = int(row['XMax'] * width)
        y1 = int(row['YMin'] * height)
        y2 = int(row['YMax'] * height)

        file_path = 'train'
        fileName = os.path.join(file_path,
                                row['FileName'])
        className = row['ClassName']
        f.write(fileName + ','+
                str(x1) + ',' +
                str(y1) + ',' +
                str(x2) + ',' +
                str(y2) + ',' +
                className + '\n')
    f.close()

    # For test
    f = open(base_path + "/test_annotation.txt", "w+")
    for idx, row in test_df.iterrows():
        img = cv2.imread((base_path + '/test/' + row['FileName']))
        height, width = img.shape[:2]
        x1 = int(row['XMin'] * width)
        x2 = int(row['XMax'] * width)
        y1 = int(row['YMin'] * height)
        y2 = int(row['YMax'] * height)

        file_path = 'test'
        fileName = os.path.join(file_path, row['FileName'])
        className = row['ClassName']
        f.write(fileName + ','+
                str(x1) + ',' +
                str(y1) + ',' +
                str(x2) + ',' +
                str(y2) + ',' +
                className + '\n')
    f.close()
    print("Done, Exiting now!! Enjoy.")


def get_data(input_path, base_path, mode='train'):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    i = 1
    with open(input_path, 'r') as f:
        print('Parsing annotation files')
        for line in f:
            # Print process
            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1
            line_split = line.strip().split(',')
            # Make sure the info saved in annotation file
            # matching the format (path_filename, x1, y1, x2, y2, class_name)
            # Note:
            #	One path_filename might has several classes (class_name)
            #	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
            #	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
            #   x1,y1-------------------
            #	|						|
            #	|						|
            #	|						|
            #	|						|
            #	---------------------x2,y2

            (filename, x1, y1, x2, y2, class_name) = line_split
            filename = filename.split("/")[-1]
            filename = os.path.join(base_path, mode, filename)
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. '
                          'Will be treated as a background region '
                          '(this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}
                # txt file doesn't contain width and height,
                # ideally, it should have this info
                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []

            all_imgs[filename]['bboxes'].append(
                {'class': class_name,
                 'x1': int(x1), 'x2': int(x2),
                 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys()
                                 if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


if __name__ == "__main__":
    extract_data_openImages(base_path="data")