import os
import shutil
import random


source_dir = '/home/wahabu/data/construction/MepDetection/zd_dataset/'
source_images = source_dir + 'images/'
source_lables = source_dir + 'labels/'
testing_classes_indxes = [1,5]  # 1: Cable Tray fitting, 5: Valve
testing_classes_indxes.sort()

des_dir = '/home/wahabu/data/construction/MepDetection/zd_CTF_V2/'
des_train = des_dir + 'train/'
des_valid = des_dir + 'valid/'
des_test = des_dir + 'test/'

des_train_images = des_train + 'images/'
des_train_labels = des_train + 'labels/'

des_valid_images = des_valid + 'images/'
des_valid_labels = des_valid + 'labels/'

des_test_images = des_test + 'images/'
des_test_labels = des_test + 'labels/'

os.makedirs(des_dir, exist_ok=True)
os.makedirs(des_train, exist_ok=True)
os.makedirs(des_test, exist_ok=True)

os.makedirs(des_train_images, exist_ok=True)
os.makedirs(des_train_labels, exist_ok=True)

os.makedirs(des_valid_images, exist_ok=True)
os.makedirs(des_valid_labels, exist_ok=True)

os.makedirs(des_test_images, exist_ok=True)
os.makedirs(des_test_labels, exist_ok=True)

for img, label in zip(os.listdir(source_images), os.listdir(source_lables)):
    # fix the label
    with open(source_lables + label, 'r') as f:
        lines = f.readlines()

    import os.path
    if not os.path.isfile(source_lables + label):
        print("file not found", source_lables + label)
    
    is_testing = False
    des_lines = []
    test_des_lines = []
    for line in lines:
        line = line.split()
        class_id = int(line[0])
        if class_id in testing_classes_indxes:
            is_testing = True
            class_id = testing_classes_indxes.index(class_id)
            line[0] = str(class_id)              
            test_des_lines.append(' '.join(line)+'\n')
        else:
            if class_id > testing_classes_indxes[0] and class_id < testing_classes_indxes[1]:
                class_id -= 1
            elif class_id > testing_classes_indxes[1]:
                class_id -= 2
            #class_id = class_id if class_id < testing_classes_indxes[0] else class_id - 1 if class_id < testing_classes_indxes[1] else class_id -2
        line[0] = str(class_id)              
        des_lines.append(' '.join(line)+'\n')
        #print("des_lines", des_lines)
    

    if is_testing:
        img_dest = des_test_images
        label_dest = des_test_labels
        des_lines = test_des_lines
    else:
        if random.random() <= 0.2:
            img_dest = des_valid_images
            label_dest = des_valid_labels
        else:
            img_dest = des_train_images
            label_dest = des_train_labels
    
    # copy the image
    shutil.copy(source_images + img, img_dest)

    # copy the label
    with(open(label_dest + label, 'w'))  as f:
        f.writelines(des_lines)



