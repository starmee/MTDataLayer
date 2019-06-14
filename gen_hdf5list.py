import os
import random


def gen_hdf5list(in_dir, out_txt):

    f = open(out_txt, "w")
    hdf5_list = os.listdir(in_dir)
    random.shuffle(hdf5_list)

    for i in hdf5_list:
        f.writelines(os.path.join(in_dir, i)+"\n")

    f.close()

if __name__ == "__main__":
    txt = "landmark_hdf5.txt"
    hdf5_dir = "/data/proj/FaceLandmark/fast-facial-landmark-detection/data/preproc/data/112/hdf5-norm"

    gen_hdf5list( hdf5_dir, txt)

    txt = "eyelid_hdf5.txt"
    hdf5_dir = "/data/proj/FaceLandmark/eyelid-classify/data/preproc/data/112/hdf5"

    gen_hdf5list( hdf5_dir, txt)
