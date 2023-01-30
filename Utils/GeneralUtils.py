import torch
from torch.autograd import Variable
import numpy as np
import math
import configparser as ConfigParser
from optparse import OptionParser
# import scipy.io.wavfile
import torch
import shutil
import os
import soundfile as sf
import sys
import re
import glob
import pickle


class GeneralUtils:
    @staticmethod
    def str_to_bool(s):
        """
        converts string to boolean
        :param s:
        :return:
        """
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            raise ValueError

    @staticmethod
    def check_folder_exist(folder_path):
        """
        checks a folder exist or not, if not exist it will be created
        :param folder_path:
        :return:
        """
        try:
            os.stat(folder_path)
        except:
            os.mkdir(folder_path)

    @staticmethod
    def read_conf(cfg_file):
        """
        reads cfg files, extracts parameter set in it and save them in options variable
        :param cfg_file:
        :return:
        """
        options = {}
        Config = ConfigParser.ConfigParser()
        Config.read(cfg_file)

        # [data]
        options['tr_lst'] = Config.get('data', 'tr_lst')
        options['te_lst'] = Config.get('data', 'te_lst')
        options['lab_dict'] = Config.get('data', 'lab_dict')
        options['data_folder'] = Config.get('data', 'data_folder')
        options['output_folder'] = Config.get('data', 'output_folder')
        options['pt_file'] = Config.get('data', 'pt_file')

        # [windowing]
        options['fs'] = Config.get('windowing', 'fs')
        options['cw_len'] = Config.get('windowing', 'cw_len')
        options['cw_shift'] = Config.get('windowing', 'cw_shift')

        # [cnn]
        options['cnn_N_filt'] = Config.get('cnn', 'cnn_N_filt')
        options['cnn_len_filt'] = Config.get('cnn', 'cnn_len_filt')
        options['cnn_max_pool_len'] = Config.get('cnn', 'cnn_max_pool_len')
        options['cnn_use_laynorm_inp'] = Config.get('cnn', 'cnn_use_laynorm_inp')
        options['cnn_use_batchnorm_inp'] = Config.get('cnn', 'cnn_use_batchnorm_inp')
        options['cnn_use_laynorm'] = Config.get('cnn', 'cnn_use_laynorm')
        options['cnn_use_batchnorm'] = Config.get('cnn', 'cnn_use_batchnorm')
        options['cnn_act'] = Config.get('cnn', 'cnn_act')
        options['cnn_drop'] = Config.get('cnn', 'cnn_drop')

        # [dnn]
        options['fc_lay'] = Config.get('dnn', 'fc_lay')
        options['fc_drop'] = Config.get('dnn', 'fc_drop')
        options['fc_use_laynorm_inp'] = Config.get('dnn', 'fc_use_laynorm_inp')
        options['fc_use_batchnorm_inp'] = Config.get('dnn', 'fc_use_batchnorm_inp')
        options['fc_use_batchnorm'] = Config.get('dnn', 'fc_use_batchnorm')
        options['fc_use_laynorm'] = Config.get('dnn', 'fc_use_laynorm')
        options['fc_act'] = Config.get('dnn', 'fc_act')

        # [class]
        options['class_lay'] = Config.get('class', 'class_lay')
        options['class_drop'] = Config.get('class', 'class_drop')
        options['class_use_laynorm_inp'] = Config.get('class', 'class_use_laynorm_inp')
        options['class_use_batchnorm_inp'] = Config.get('class', 'class_use_batchnorm_inp')
        options['class_use_batchnorm'] = Config.get('class', 'class_use_batchnorm')
        options['class_use_laynorm'] = Config.get('class', 'class_use_laynorm')
        options['class_act'] = Config.get('class', 'class_act')

        # [optimization]
        options['lr'] = Config.get('optimization', 'lr')
        options['batch_size'] = Config.get('optimization', 'batch_size')
        options['N_epochs'] = Config.get('optimization', 'N_epochs')
        options['N_batches'] = Config.get('optimization', 'N_batches')
        options['N_eval_epoch'] = Config.get('optimization', 'N_eval_epoch')
        options['seed'] = Config.get('optimization', 'seed')
        options['kernel_type'] = Config.get('optimization', 'kernel_type')

        return options

    @staticmethod
    def read_list(list_file):
        """
        read list of file names present in the file specified
        :param list_file:
        :return:
        """
        f = open(list_file, "r")
        lines = f.readlines()
        list_sig = []
        for x in lines:
            list_sig.append(x.rstrip())
        f.close()
        return list_sig

    @staticmethod
    def copy_folder(in_folder, out_folder):
        """
        copies the structure of one folder to another without files
        :param in_folder:
        :param out_folder:
        :return:
        """
        if not (os.path.isdir(out_folder)):
            shutil.copytree(in_folder, out_folder, ignore=GeneralUtils.ignore_files)

    @staticmethod
    def ignore_files(dir, files):
        """
        returns list of files in a folder
        :param dir:
        :param files:
        :return:
        """
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    @staticmethod
    def get_core_wav_test_files(wav_lst_te_comp):
        core_test_speakers = ['DAB0', 'WBT0', 'ELC0',
                              'TAS1', 'WEW0', 'PAS0',
                              'JMP0', 'LNT0', 'PKT0',
                              'LLL0', 'TLS0', 'JLM0',
                              'BPM0', 'KLT0', 'NLP0',
                              'CMJ0', 'JDH0', 'MGD0',
                              'GRT0', 'NJM0', 'DHC0',
                              'JLN0', 'PAM0', 'MLD0']
        wav_lst_te_core = []
        for i in range(len(wav_lst_te_comp)):
            for j in range(len(core_test_speakers)):
                if core_test_speakers[j] in wav_lst_te_comp[i]:
                    wav_lst_te_core.append(wav_lst_te_comp[i])
                    break
        return wav_lst_te_core

