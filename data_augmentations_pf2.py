import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate, ElasticTransform
from sklearn.model_selection import KFold
from itertools import zip_longest

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
    # X = sorted(glob(os.path.abspath(os.path.join(path, "images"+sep+"*"+sep+"*" , "*.png"))))
    # Y = sorted(glob(os.path.abspath(os.path.join(path, "groundtruth"+sep+"*"+sep+"*", "*.png"))))

    #for scenario 1
    # X = sorted(glob(os.path.abspath(os.path.join(path, "original_A"+sep+"*"+sep+"*" , "*.png"))))
    # Y = sorted(glob(os.path.abspath(os.path.join(path, "original_AL"+sep+"*"+sep+"*", "*.png"))))

    #for scenario 1 and 2 after splitting
    X = sorted(glob(os.path.abspath(os.path.join(path, "train_A_scen_1", "*.png"))))
    Y = sorted(glob(os.path.abspath(os.path.join(path, "train_B_scen_1", "*.png"))))

    X1 = sorted(glob(os.path.abspath(os.path.join(path, "train_A_scen_2", "*.png"))))
    Y1 = sorted(glob(os.path.abspath(os.path.join(path, "train_B_scen_2", "*.png"))))

    # X = sorted(glob(os.path.abspath(os.path.join(path, "test_A", "*.png"))))
    # Y = sorted(glob(os.path.abspath(os.path.join(path, "test_B", "*.png"))))

    # print(X)
    # Y = sorted(glob(os.path.abspath(os.path.join(path, "masks_denormalized"+sep+"*", "*.png"))))

    # print(os.path.join(path, "original_A" , "**", "*.png"))

    """ Spliting the data into training and testing """
    split_size = int(len(X) * split)
    split_size1 = int(len(X1) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    train_x1, test_x1 = train_test_split(X1, test_size=split_size1, random_state=42)
    train_y1, test_y1 = train_test_split(Y1, test_size=split_size1, random_state=42)
    
#     train_x, valid_x = train_test_split(X, test_size=split_size, random_state=42)
#     train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    
#     train_y, valid_y = train_test_split(Y, test_size=split_size, random_state=42)
#     train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y), (train_x1, train_y1), (test_x1, test_y1)
    # return train_x, test_x


def augment_data(domain, images, masks, images1, masks1, save_path, augment=True):

    # print(images)
    # print(masks)
    # for x, y, x1, y1 in tqdm(zip(images, masks, images1, masks1), total=len(images)+len(images1)):
    for x, y, x1, y1 in tqdm(zip_longest(images, masks, images1, masks1), total=max(len(images), len(images1))):
        # print(x[0].split(sep)[-1])
        """ Extract the name """
        if domain=="color":
            if x is not None:
                name = x.split(sep)[-1].split(".")[0].split("_")
                name = '_'.join(name[0:3]) #scenario 1
                # name = '_'.join(name[0:4]) #scenario 2

                name_y = y.split(sep)[-1].split(".")[0].split("_")
                name_y = '_'.join(name_y[0:3]) #scenario 1
                # name_y = '_'.join(name_y[0:4]) #scenario 2
                # name = x[0].split(sep)[-1]
    #             print(x)
            
            if x1 is not None:
                name1 = x1.split(sep)[-1].split(".")[0].split("_")
                # name1 = '_'.join(name1[0:3]) #scenario 1
                name1 = '_'.join(name1[0:4]) #scenario 2

                name_y1 = y1.split(sep)[-1].split(".")[0].split("_")
                # name_y1 = '_'.join(name_y1[0:3]) #scenario 1
                name_y1 = '_'.join(name_y1[0:4]) #scenario 2
                # name = x[0].split(sep)[-1]
    #             print(x)
        # print(x)

        
        if x is not None:
            """ Reading the image and mask """
            x = cv2.imread(x, cv2.IMREAD_COLOR)
            y = cv2.imread(y, cv2.IMREAD_COLOR)

        if x1 is not None:
            x1 = cv2.imread(x1, cv2.IMREAD_COLOR)
            y1 = cv2.imread(y1, cv2.IMREAD_COLOR)
        # print(A.shape)
        

        """ Augmentation """
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            if x is not None:
                augmented = aug(image=x, mask=y)
                x2 = augmented["image"]
                y2 = augmented["mask"]

            if x1 is not None:
                augmented1 = aug(image=x1, mask=y1)
                x5 = augmented1["image"]
                y5 = augmented1["mask"]

            aug = VerticalFlip(p=1.0)
            if x is not None:
                augmented = aug(image=x, mask=y)
                x3 = augmented["image"]
                y3 = augmented["mask"]

            if x1 is not None:
                augmented1 = aug(image=x1, mask=y1)
                x6 = augmented1["image"]
                y6 = augmented1["mask"]
            
            # # Generate a random angle within your desired limits
            # np.random.seed()  # Ensure randomness
            # angle = np.random.uniform(low=-45, high=45)  # Adjust limits as needed

            # Define the augmentation with the generated angle
            aug = Rotate(limit=(45, 45), p=1.0)
            if x is not None:
                augmented = aug(image=x, mask=y)
                x4 = augmented["image"]
                y4 = augmented["mask"]

            if x1 is not None:
                augmented1 = aug(image=x1, mask=y1)
                x7 = augmented1["image"]
                y7 = augmented1["mask"]

            if x is not None:
                X = [x, x2, x3, x4]
                Y = [y, y2, y3, y4]

            if x1 is not None:
                X1 = [x1, x5, x6, x7]
                Y1 = [y1, y5, y6, y7]

        else:
            X = [x]
            Y = [y]

        index = 0

        if x is not None:
            for i, m in zip(X, Y):
                try:
                    aug = cv2.resize(i, (W, H))
                    augmented = aug(image=i, mask=m)
                    i = augmented["image"]
                    m = augmented["mask"]

                except Exception as e:
                    i = cv2.resize(i, (W, H))
                    m = cv2.resize(m, (W, H))


                tmp_image_name = f"{name}_{index}.png"
                tmp_mask_name = f"{name_y}_{index}.png"

                image_path = os.path.join(save_path, "train_A_aug_scen_1", tmp_image_name)
                mask_path = os.path.join(save_path, "train_B_aug_scen_1", tmp_mask_name)

                cv2.imwrite(image_path, i)
                cv2.imwrite(mask_path, m)

                index += 1
        
        index = 0
        if x1 is not None:
            for i1, m1 in zip(X1, Y1):
                try:
                    aug1 = cv2.resize(i1, (W, H))
                    augmented1 = aug1(image1=i1, mask1=m1)
                    i1 = augmented1["image1"]
                    m1 = augmented1["mask1"]

                except Exception as e:
                    i1 = cv2.resize(i1, (W, H))
                    m1 = cv2.resize(m1, (W, H))

                tmp_image_name1 = f"{name1}_{index}.png"
                tmp_mask_name1 = f"{name_y1}_{index}.png"

                image_path1 = os.path.join(save_path, "train_A_aug_scen_2", tmp_image_name1)
                mask_path1 = os.path.join(save_path, "train_B_aug_scen_2", tmp_mask_name1)

                cv2.imwrite(image_path1, i1)
                cv2.imwrite(mask_path1, m1)

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

    data_path = "C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_no_aug_scenario1_scenario2_newDS\\"
    # data_path = "C:\\Baseline Dataset\\color_separation\\sharp_category\\"
    domain_dataset = "color"

    name_scenario = "scenario1_scenario2_newDS"
    # name_scenario = "scenario2_newDS"

    # data_path = "C:\\Baseline Dataset\\BUSI\\"
    # domain_dataset = "busi"
    master_folder = "new_data_80_20_sharp_"
    # master_folder = "new_data_hybrid_(80_20)_10_"
    # data_path = os.path.join(os.getcwd(), data_path)
    
    (train_x, train_y), (test_x, test_y), (train_x1, train_y1), (test_x1, test_y1) = load_data(data_path, 0.2)
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_path)
    print(f"Train:\t {len(train_x)}")
#     print(f"Train:\t {len(valid_x)} - {len(valid_y)}")
    print(f"Test:\t {len(test_x)}")

    print(f"Train:\t {len(train_x1)}")
#     print(f"Train:\t {len(valid_x)} - {len(valid_y)}")
    print(f"Test:\t {len(test_x1)}")

    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/test/train_A_aug_scen_1/")
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/test/train_B_aug_scen_1/")

    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/test/train_A_aug_scen_2/")
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/test/train_B_aug_scen_2/")

    augment_data(domain_dataset, test_x, test_y, test_x1, test_y1, master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/test/", augment=True)
    # augment_data(domain_dataset, test_x1, test_y1, master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/test_scen_2/", augment=True)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    train_x1 = np.array(train_x1)
    train_y1 = np.array(train_y1)

    # kfold = KFold(n_splits=1, shuffle=True, random_state=42)
    # fold = 0
    # for train_idx, val_idx in kfold.split(train_x):
    fold = 1
    print(f"Training fold {fold}")
    # print((train_x))
    # print(train_x[0])

    train_x_fold = train_x
    train_y_fold = train_y

    train_x_fold1 = train_x1
    train_y_fold1 = train_y1

    print("Train Fold")
    print(len(train_x_fold))
    print(len(train_y_fold))

    print("Train Fold1")
    print(len(train_x_fold1))
    print(len(train_y_fold1))

    # print("Valid Fold")
    # print(len(valid_x_fold))
    # print(len(valid_y_fold))

    # if fold==3:
    #     print(train_x_fold)


    """ Create directories to save the augmented data """
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/fold_"+str(fold)+"/train/train_A_aug_scen_1/")
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/fold_"+str(fold)+"/train/train_B_aug_scen_1/")

    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/fold_"+str(fold)+"/train/train_A_aug_scen_2/")
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/fold_"+str(fold)+"/train/train_B_aug_scen_2/")
    # create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/train/mask/")
    # create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/")
    # create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/mask/")
    # create_dir("new_data_hybrid_radiography_no_aug/valid/image/")
    # create_dir("new_data_hybrid_radiography_no_aug/valid/mask/")
    

    """ Data augmentation """
    augment_data(domain_dataset, train_x_fold, train_y_fold, train_x_fold1, train_y_fold1, master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/fold_"+str(fold)+"/train/", augment=True)
    # augment_data(domain_dataset, train_x_fold1, train_y_fold1, master_path+master_folder+domain_dataset+"_fold_"+str(W)+f"_aug_{name_scenario}_journal"+"/fold_"+str(fold)+"/train_scen_2/", augment=True)
    # augment_data(domain_dataset, valid_x_fold, master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/", augment=False)
    # augment_data(domain_dataset, valid_x, valid_y, "new_data_hybrid_radiography/valid/", augment=False)