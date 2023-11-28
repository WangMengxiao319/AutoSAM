import numpy as np
import os
import shutil
from PIL import Image
from medpy.io import load
import SimpleITK as sitk
import nrrd

join = os.path.join


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def convert_volume_to_imgs(data_dir, output_dir):
    img_dir = join(output_dir, 'imgs')
    label_dir = join(output_dir, 'annotations')

    files_list = os.listdir(data_dir)
    for f in files_list:
        f = f.split(".")[0]

        target_img_dir = join(img_dir, f)
        target_label_dir = join(label_dir, f)
        if not os.path.exists(target_img_dir):
            os.makedirs(join(target_img_dir))
            os.makedirs(join(target_label_dir))
            print('Created' + target_img_dir + '...')

        data = np.load(os.path.join(data_dir, f + ".npy"))

        image = data[:, 0]
        label = data[:, 1]

        for i in range(image.shape[0]):
            img_array = (image[i] - image.min()) / (image.max() - image.min())
            img_array = np.clip(img_array * 127.5, 0, 255).astype('uint8')
            label_array = label[i].astype('uint8')
            if i < 10:
                save_files = f + "_00" + str(i) + ".png"
            elif 10 <= i < 100:
                save_files = f + "_0" + str(i) + ".png"
            else:
                save_files = f + "_" + str(i) + ".png"

            im = Image.fromarray(img_array)
            new_p = im.convert('RGB')
            new_p.save(join(target_img_dir, save_files))

            lb = Image.fromarray(label_array)
            lb.save(join(target_label_dir, save_files))

        print("finishing saving", f)


def convert_npz_to_imgs(data_dir, output_dir):
    img_dir = join(output_dir, 'imgs')
    label_dir = join(output_dir, 'annotations')

    files_list = os.listdir(data_dir)
    for f in files_list:
        f = f.split(".")[0]

        target_img_dir = join(img_dir, f)
        target_label_dir = join(label_dir, f)
        if not os.path.exists(target_img_dir):
            os.makedirs(join(target_img_dir))
            os.makedirs(join(target_label_dir))
            print('Created' + target_img_dir + '...')

        data = np.load(os.path.join(data_dir, f + ".npz"))["data"]
        print(data.shape)

        image = data[0]
        label = data[4]
        label[label < 0] = 0

        for i in range(image.shape[0]):
            img_array = (image[i] - image.min()) / (image.max() - image.min())
            img_array = np.clip(img_array * 127.5, 0, 255).astype('uint8')
            label_array = label[i].astype('uint8')

            if np.sum(label_array) == 0: continue
            if i < 10:
                save_files = f + "_00" + str(i) + ".png"
            elif 10 <= i < 100:
                save_files = f + "_0" + str(i) + ".png"
            else:
                save_files = f + "_" + str(i) + ".png"

            im = Image.fromarray(img_array)
            new_p = im.convert('RGB')
            new_p = new_p.resize((256, 256))
            new_p.save(join(target_img_dir, save_files))

            lb = Image.fromarray(label_array)
            lb = lb.resize((256, 256))
            lb.save(join(target_label_dir, save_files))

        print("finishing saving", f)


def convert_nii_to_imgs(data_dir, output_dir):
    img_dir = join(output_dir, 'imgs')
    label_dir = join(output_dir, 'annotations')

    files_list = sorted(os.listdir(data_dir))
    src_label_dir = data_dir.replace("imagesTr", "labelsTr")
    label_list = sorted(os.listdir(src_label_dir))
    for f, label_f in zip(files_list, label_list):
        f = f.split(".")[0]

        target_img_dir = join(img_dir, f)
        target_label_dir = join(label_dir, f)
        if not os.path.exists(target_img_dir):
            os.makedirs(join(target_img_dir))
            os.makedirs(join(target_label_dir))
            print('Created' + target_img_dir + '...')

        image = read_nii(join(data_dir, f + ".nii.gz"))
        label = read_nii(join(src_label_dir, label_f))
        label[label < 0] = 0

        for i in range(image.shape[0]):
            img_array = (image[i] - image.min()) / (image.max() - image.min())
            img_array = np.clip(img_array * 127.5, 0, 255).astype('uint8')
            label_array = label[i].astype('uint8')
            if np.sum(label_array) == 0: continue
            if i < 10:
                save_files = f + "_00" + str(i) + ".png"
            elif 10 <= i < 100:
                save_files = f + "_0" + str(i) + ".png"
            else:
                save_files = f + "_" + str(i) + ".png"

            im = Image.fromarray(img_array)
            new_p = im.convert('RGB')
            new_p.save(join(target_img_dir, save_files))

            lb = Image.fromarray(label_array)
            lb.save(join(target_label_dir, save_files))

        print("finishing saving", f)


def convert_acdc_to_imgs(data_dir, output_dir):
    '''Mancy 修改，可以直接操作acdc原始数据库'''
    img_dir = join(output_dir, 'imgs')
    label_dir = join(output_dir, 'annotations')

    patient_list = os.listdir(data_dir)
    patient_list = patient_list[:2]
    for ppl in patient_list:
        print(ppl)
        files_list = os.listdir(join(data_dir, ppl))
        for f in files_list:
            f = f.split(".")[0]
            if ("frame" in f) and ("gt" not in f):
                modified_f = f.replace('patient', 'patient_').replace('frame', 'frame_')
                frame_and_after = 'frame'+ modified_f.split('frame', 1)[-1].lstrip('0')
                target_img_dir = join(img_dir, modified_f)
                target_label_dir = join(label_dir, modified_f)
                if not os.path.exists(target_img_dir):
                    os.makedirs(join(target_img_dir))
                    os.makedirs(join(target_label_dir))
                    print('Created' + target_img_dir + '...')

                # data = np.load(os.path.join(data_dir, ppl, f + ".npy"))
                image = read_nii(join(data_dir, ppl, f + ".nii.gz"))
                label = read_nii(join(data_dir, ppl, f + "_gt.nii.gz"))
                print(image.shape)
                for i in range(image.shape[0]):
                    img_array = (image[i] - image.min()) / (image.max() - image.min())
                    img_array = np.clip(img_array * 255, 0, 255).astype('uint8')   # 映射到0到255
                    print(img_array.shape)
                    label_array = label[i].astype('uint8')
                    # print(np.unique(img_array))
                    # print(np.unique(label_array))
                    if i < 10:
                        save_files = frame_and_after + "_00" + str(i) + ".png"
                    elif 10 <= i < 100:
                        save_files = frame_and_after + "_0" + str(i) + ".png"
                    else:
                        save_files = frame_and_after + "_" + str(i) + ".png"

                    im = Image.fromarray(img_array)
                    new_p = im.convert('RGB')
                    new_p.save(join(target_img_dir, save_files))

                    lb = Image.fromarray(label_array)
                    lb.save(join(target_label_dir, save_files))

                print("finishing saving", f)

def convert_nrrd_to_imgs(data_dir, output_dir):
    '''load nrrd files and convert to png images'''
    img_dir = join(output_dir, 'imgs')
    label_dir = join(output_dir, 'annotations')
    patient_list = os.listdir(data_dir)
    patient_list.sort(key=lambda x: int(x)) # 按照序数排列
    # patient_list = patient_list[12:14]  # debug
    for ppl in patient_list:
        print(ppl)
        if int(ppl) < 10:
            patient_name = 'patient'+ "_00" + ppl
        elif 10 <= int(ppl) < 100:
            patient_name = 'patient' + "_0" + ppl 
        else:
            patient_name = 'patient' + "_" + ppl 
        subfolders_list = os.listdir(join(data_dir, ppl))   # ['原图nrrd文件', '标注文件'] 但是名字不太统一
        target_img_dir = join(img_dir, ppl)
        target_label_dir = join(label_dir, ppl)
        if not os.path.exists(target_img_dir):
            os.makedirs(join(target_img_dir))
            os.makedirs(join(target_label_dir))
            print('Created' + target_img_dir + '...')
            print('Created' + target_label_dir + '...')
        for folder in subfolders_list:
            files_list = os.listdir(join(data_dir, ppl, folder))
            for f in files_list:
                if f.split(".")[-1] == 'nrrd':  # 所有nrrd结尾的文件
                    f = f.split(".")[0]
                    if "Segmentation" in f:
                        '''处理annotations(label)文件'''
                        # data = np.load(os.path.join(data_dir, ppl, f + ".npy"))
                        label,header = nrrd.read(join(data_dir, ppl, folder, f + ".seg.nrrd"))    # Data Shape: (1120, 1120, 1)
                        label_array = label[:,:,0].transpose(1,0).astype('uint8')*255
                            # print(np.unique(img_array))
                        # print(np.unique(label_array))
                        frame = f.split("_")[-1]
                        if int(frame) < 10:
                            frame_name = 'frame' + "_00" + frame
                        elif 10 <= int(frame) < 100:
                            frame_name = 'frame' + "_0" + frame
                        else:
                            frame_name = 'frame' + "_" + frame
                        save_files = patient_name +'_'+ frame_name + ".png"
                        lb = Image.fromarray(label_array)
                        lb.save(join(target_label_dir, save_files))
                    else:
                        '''处理imgs文件'''
                        image,header = nrrd.read(join(data_dir, ppl, folder, f + ".nrrd"))  # Data Shape: (3, 1120, 1120, 1)
                        if len(image.shape) == 4:
                            image = image[:,:,:,0].transpose(2, 1, 0)  # Data Shape: (1120, 1120, 3)
                        else:
                            image = image[:,:,0]
                            image[image == 32767] = 2168
                            image = image.transpose(1,0).astype(np.float64)
                        img_array = (image - image.min()) / (image.max() - image.min())
                        img_array = np.clip(img_array * 255, 0, 255).astype('uint8')   # 映射到0到255
                        # print(np.unique(img_array))
                        frame = f.split("_")[-1]
                        if int(frame) < 10:
                            frame_name = 'frame' + "_00" + frame
                        elif 10 <= int(frame) < 100:
                            frame_name = 'frame' + "_0" + frame
                        else:
                            frame_name = 'frame' + "_" + frame
                        save_files = patient_name + '_'+frame_name + ".png"
                        im = Image.fromarray(img_array)
                        new_p = im.convert('RGB')
                        new_p.save(join(target_img_dir, save_files))
                    
                    print("finishing saving", f)


def convert_scribbles_to_imgs(data_dir, output_dir):
    file_list = os.listdir(data_dir)
    for f in file_list:
        scribble, _ = load(join(data_dir, f))
        fname = f.split('_scribble')[0]
        print(scribble.shape)
        if not os.path.exists(join(output_dir, fname)):
            os.mkdir(join(output_dir, fname))
        for i in range(scribble.shape[2]):

            srb = scribble[:,:,i].astype('uint8')

            if i < 10:
                save_files = fname + "_00" + str(i) + ".png"
            elif 10 <= i < 100:
                save_files = fname + "_0" + str(i) + ".png"
            else:
                save_files = fname + "_" + str(i) + ".png"

            lb = Image.fromarray(srb)
            lb.save(join(output_dir, fname, save_files))


def save_as_slices(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')
    else :
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')
    files_list = os.listdir(input_dir)

    for f in files_list:
        f = f.split(".")[0]

        target_dir = join(output_dir, f)
        if not os.path.exists(target_dir):
            os.makedirs(join(target_dir))
            print('Created' + target_dir + '...')

        data = np.load(os.path.join(input_dir, f+".npy"))

        image = data[:, 0]
        label = data[:, 1]

        for i in range(image.shape[0]):
            slice_data = np.concatenate((image[i][None], label[i][None]))
            if i < 10:
                save_path = join(target_dir, f + "_00" + str(i) + ".npy")
            elif 10 <= i < 100:
                save_path = join(target_dir, f + "_0" + str(i) + ".npy")
            else:
                save_path = join(target_dir, f + "_" + str(i) + ".npy")

            np.save(save_path, slice_data)

        print("finishing saving", f)


if __name__ == "__main__":
    # data_dir = "../../../DATA/brats_t1/preprocessed"
    # # data_dir = "../../../DATA/ACDC/acdc_scribbles"
    # # output_dir = "../../../DATA/synapse"
    # output_dir = "../../../DATA/brats"
    # convert_npz_to_imgs(data_dir, output_dir)
    # file_list = os.listdir(output_dir)
    # for f in file_list:
    #     slices = os.listdir(join(output_dir, f))
    #     for slice in slices:
    #         new_slice = slice[0:5] + "_" + slice[5:]
    #         print(new_slice)
    #         os.rename(join(output_dir, f, slice), join(output_dir, f, new_slice))

    # ### process ACDC dataset
    # data_dir = 'D:/Filez/dataset/ACDC/raw/training'
    # output_dir = 'D:/Filez/dataset/ACDC/processed'
    # convert_acdc_to_imgs(data_dir, output_dir)

    ### process LiPing_CTA dataset
    data_dir = 'D:/Filez/dataset/LiPing_multi_modal/CTA dataset/已标注（未植入支架）'
    output_dir = 'D:/Filez/dataset/LiPing_multi_modal/CTA dataset/processed_without_stent'
    convert_nrrd_to_imgs(data_dir, output_dir)