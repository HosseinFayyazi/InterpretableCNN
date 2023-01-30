import torch
import os
import glob
import argparse
import re
import numpy as np
import random
# from torchnlp.text_encoders import StaticTokenizerEncoder

# pip install pytorch-nlp==0.3.0


class DataUtils:
    def __init__(self):
        return

    @staticmethod
    def get_sid_class_labels(wav_file_list):
        """
        extracts TIMIT labels for list of wav file names given to it for SID task
        :param wav_file_list:
        :return:
        """
        class_lbls = set()
        for file_name in wav_file_list:
            indices = [_.start() for _ in re.finditer('/', file_name)]
            speaker_id = file_name[indices[-2]+1: indices[-1]]
            class_lbls.add(speaker_id)
        return list(class_lbls)

    @staticmethod
    def distinct_sid_train_test(wav_lst, lbls):
        """
        distincts train, val and test data for SID
        :param wav_lst:
        :param lbls:
        :return:
        """
        wav_lst_tr = []
        wav_lst_val = []
        wav_lst_te = []
        tr_counter = np.zeros(len(lbls))
        for lbl_ind, lbl in enumerate(lbls):
            for file_name in wav_lst:
                if lbl in file_name:
                    if tr_counter[lbl_ind] < 5:
                        wav_lst_tr.append(file_name)
                        tr_counter[lbl_ind] += 1
                    elif 5 <= tr_counter[lbl_ind] < 7:
                        wav_lst_val.append(file_name)
                        tr_counter[lbl_ind] += 1
                    else:
                        wav_lst_te.append(file_name)
        return wav_lst_tr, wav_lst_val, wav_lst_te

    @staticmethod
    def build_sid_lab_dict(wav_lst, class_lbls):
        """
        builds a dictionary which determines the speaker label of each file name for SID task
        :param wav_lst:
        :param class_lbls:
        :return:
        """
        lab_dict = {}
        for file_name in wav_lst:
            indices = [_.start() for _ in re.finditer('/', file_name)]
            speaker_id = file_name[indices[-2] + 1: indices[-1]]
            lbl = class_lbls.index(speaker_id)
            lab_dict[file_name] = lbl
        return lab_dict

    @staticmethod
    def get_core_wav_test_files(wav_file_names):
        """
        returns core test wave file names
        :param wav_file_names:
        :return:
        """
        test_speakers = [
            'MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', 'FPAS0', 'MJMP0', 'MLNT0', 'FPKT0', 'MLLL0', 'MTLS0', 'FJLM0',
            'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0', 'MJDH0', 'FMGD0', 'MGRT0', 'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0']
        core_wav_file_names = []
        for file_name in wav_file_names:
            for test_speaker in test_speakers:
                if test_speaker in file_name and 'SA1' not in file_name and 'SA2' not in file_name:
                    core_wav_file_names.append(file_name)
                    break
        return core_wav_file_names

    @staticmethod
    def remove_sa_wav_files(wav_file_names):
        """
        removes SA files from data
        :param wav_file_names:
        :return:
        """
        new_wav_file_names = []
        for file_name in wav_file_names:
            if 'SA1' not in file_name and 'SA2' not in file_name:
                new_wav_file_names.append(file_name.replace('\\', '/'))
        return new_wav_file_names

    @staticmethod
    def read_wav_file_names(data_path, train=1):
        """
        reads wav file names in the path specified
        train = 1 : train data will be read
        train != 1: test data will be read
        :param data_path:
        :param train:
        :return:
        """
        if train == 1:
            wav_file_names = glob.glob(os.path.join(data_path, "TRAIN/**/*.WAV"), recursive=True)
        else:
            wav_file_names = glob.glob(os.path.join(data_path, "TEST/**/*.WAV"), recursive=True)
        return wav_file_names



