import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate, ElasticTransform
from sklearn.model_selection import KFold

sep = "\\"
master_path = "C:"+sep+"Baseline_Dataset"+sep+"dataset_fix"+sep
H = 1024
W = 1024

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    """ Loading the images and masks """
#     X = sorted(glob(os.path.join(path, "images", "*.jpg")))
#     Y = sorted(glob(os.path.join(path, "masks", "*.png")))
    
    #for scenario 2
    X = sorted(glob(os.path.abspath(os.path.join(path, "images"+sep+"*"+sep+"*" , "*.png"))))
    Y = sorted(glob(os.path.abspath(os.path.join(path, "groundtruth"+sep+"*"+sep+"*", "*.png"))))

    #for scenario 1
    # X = sorted(glob(os.path.abspath(os.path.join(path, "original_A"+sep+"*"+sep+"*" , "*.png"))))
    # Y = sorted(glob(os.path.abspath(os.path.join(path, "original_AL"+sep+"*"+sep+"*", "*.png"))))

    # print(X)
    # Y = sorted(glob(os.path.abspath(os.path.join(path, "masks_denormalized"+sep+"*", "*.png"))))

    # print(os.path.join(path, "original_A" , "**", "*.png"))

    """ Spliting the data into training and testing """
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)
    
#     train_x, valid_x = train_test_split(X, test_size=split_size, random_state=42)
#     train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    
#     train_y, valid_y = train_test_split(Y, test_size=split_size, random_state=42)
#     train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)
    # return train_x, test_x


def augment_data(domain, images, masks, save_path, augment=True):

    # print(images)
    # print(masks)
    for x, y in tqdm(zip(images, masks), total=len(images)):
        # print(x[0].split(sep)[-1])
        """ Extract the name """
        if domain=="color":
            name = x.split(sep)[-1].split(".")[0]
            name_y = y.split(sep)[-1].split(".")[0]
            # name = x[0].split(sep)[-1]
#             print(x)
        # print(x)

        

        """ Reading the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        # print(A.shape)
        

        """ Augmentation """
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            try:
                # print(i[0].shape)
#                 """ Center Cropping """
#                 aug = CenterCrop(H, W, p=1.0)
#                 augmented = aug(image=i, mask=m)
#                 i = augmented["image"]
#                 m = augmented["mask"]
                aug = cv2.resize(i, (W, H))
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name_y}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "groundtruth", tmp_mask_name)
#             print(image_path)
#             print(mask_path)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the dataset """
#     data_path = "../people_segmentation"

    # data_path = "../../dataset/Fix_Bird_Dataset"
    # data_path = "dataset_cubbird"

    # data_path = "C:\\Baseline Dataset\\COVID-19_Radiography_Dataset\\"
    # domain_dataset = "radiography"

    data_path = "C:\\Baseline_Dataset\\color_separation\\scenario2\\"
    # data_path = "C:\\Baseline Dataset\\color_separation\\sharp_category\\"
    domain_dataset = "color"

    name_scenario = "scenario2_newDS"
    # name_scenario = "scenario2_newDS"

    # data_path = "C:\\Baseline Dataset\\BUSI\\"
    # domain_dataset = "busi"
    master_folder = "new_data_80_20_sharp_"
    # master_folder = "new_data_hybrid_(80_20)_10_"
    # data_path = os.path.join(os.getcwd(), data_path)
    
    (train_x, train_y), (test_x, test_y) = load_data(data_path, 0.2)
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_path)
    print(f"Train:\t {len(train_x)}")
#     print(f"Train:\t {len(valid_x)} - {len(valid_y)}")
    print(f"Test:\t {len(test_x)}")

    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_conference"+"/test/image")
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_conference"+"/test/groundtruth/")

    augment_data(domain_dataset, test_x, test_y, master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_conference"+"/test/", augment=False)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # kfold = KFold(n_splits=1, shuffle=True, random_state=42)
    # fold = 0
    # for train_idx, val_idx in kfold.split(train_x):
    fold = 1
    print(f"Training fold {fold}")
    # print((train_x))
    # print(train_x[0])

    train_x_fold = train_x
    train_y_fold = train_y

    print("Train Fold")
    print(len(train_x_fold))
    print(len(train_y_fold))

    # print("Valid Fold")
    # print(len(valid_x_fold))
    # print(len(valid_y_fold))

    # if fold==3:
    #     print(train_x_fold)


    """ Create directories to save the augmented data """
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_conference"+"/fold_"+str(fold)+"/train/image/")
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_conference"+"/fold_"+str(fold)+"/train/groundtruth/")
    # create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/train/mask/")
    # create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/")
    # create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/mask/")
    # create_dir("new_data_hybrid_radiography_no_aug/valid/image/")
    # create_dir("new_data_hybrid_radiography_no_aug/valid/mask/")
    

    """ Data augmentation """
    augment_data(domain_dataset, train_x_fold, train_y_fold, master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_conference"+"/fold_"+str(fold)+"/train/", augment=True)
    # augment_data(domain_dataset, valid_x_fold, master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/", augment=False)
    # augment_data(domain_dataset, valid_x, valid_y, "new_data_hybrid_radiography/valid/", augment=False)
