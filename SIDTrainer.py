# import scipy.io.wavfile
import io

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from NetModels.Models import MLP
from NetModels.Models import NetModel as net_model
import random
from Utils.GeneralUtils import *
from Utils.SignalUtils import *
# from detecto.core import Model


class SIDTrainer:
    def __init__(self, options, wav_lst_tr, snt_tr, wav_lst_te, snt_te, lab_dict, save_path):
        self.gpu = torch.cuda.is_available()  # & torch.cuda.device_count() > 0
        self.options = options
        self.Batch_dev = 128
        self.kernel_type = options['kernel_type']
        # loss function
        self.cost = nn.NLLLoss()
        self.lr = float(self.options['lr'])
        self.batch_size = int(options['batch_size'])
        self.N_epochs = int(options['N_epochs'])
        self.N_batches = int(options['N_batches'])
        self.N_eval_epoch = int(options['N_eval_epoch'])
        self.data_folder = options['data_folder'] + '/'
        # windowing
        self.fs = int(options['fs'])
        cw_len = int(options['cw_len'])
        cw_shift = int(options['cw_shift'])

        self.wlen = int(self.fs * cw_len / 1000.00)
        self.wshift = int(self.fs * cw_shift / 1000.00)

        self.wav_lst_tr = wav_lst_tr
        self.snt_tr = snt_tr
        self.wav_lst_te = wav_lst_te
        self.snt_te = snt_te
        self.output_folder = options['output_folder']
        self.class_lay = list(map(int, options['class_lay'].split(',')))

        self.lab_dict = lab_dict  # np.load(options['lab_dict'], allow_pickle=True).item()
        self.save_path = save_path
        self.best_err_tot_dev_snt = 100

        return

    def train_one_batch(self, inp, lab):
        """
        trains the model for one batch
        :param inp:
        :param lab:
        :return:
        """
        pout = self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))
        pred = torch.max(pout, dim=1)[1]
        loss = self.cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())

        self.optimizer_CNN.zero_grad()
        self.optimizer_DNN1.zero_grad()
        self.optimizer_DNN2.zero_grad()

        loss.backward()
        self.optimizer_CNN.step()
        self.optimizer_DNN1.step()
        self.optimizer_DNN2.step()

        return loss, err

    def split_signals_into_chunks(self, lab_batch, signal):
        """
        splits a test signal into chunks and computes output of model for it
        :param lab_batch:
        :param signal:
        :return:
        """
        beg_samp = 0
        end_samp = self.wlen
        N_fr = int((signal.shape[0] - self.wlen) / (self.wshift))
        if self.gpu:
            sig_arr = torch.zeros([self.Batch_dev, self.wlen]).float().cuda().contiguous()
            lab = Variable((torch.zeros(N_fr + 1) + lab_batch).cuda().contiguous().long())
            pout = Variable(torch.zeros(N_fr + 1, self.class_lay[-1]).float().cuda().contiguous())
        else:
            sig_arr = torch.zeros([self.Batch_dev, self.wlen]).float().contiguous()
            lab = Variable((torch.zeros(N_fr + 1) + lab_batch).contiguous().long())
            pout = Variable(torch.zeros(N_fr + 1, self.class_lay[-1]).float().contiguous())
        count_fr = 0
        count_fr_tot = 0
        while end_samp < signal.shape[0]:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp + self.wshift
            end_samp = beg_samp + self.wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1
            if count_fr == self.Batch_dev:
                inp = Variable(sig_arr)
                pout[count_fr_tot - self.Batch_dev:count_fr_tot, :] = self.DNN2_net(
                    self.DNN1_net(self.CNN_net(inp)))
                count_fr = 0
                if self.gpu:
                    sig_arr = torch.zeros([self.Batch_dev, self.wlen]).float().cuda().contiguous()
                else:
                    sig_arr = torch.zeros([self.Batch_dev, self.wlen]).float().contiguous()

        if count_fr > 0:
            inp = Variable(sig_arr[0:count_fr])
            pout[count_fr_tot - count_fr:count_fr_tot, :] = self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))

        return pout, lab

    def test_one_signal(self, signal, index):
        """
        test one signal and determines its class label
        :param signal:
        :param index:
        :return:
        """
        if self.gpu:
            signal = torch.from_numpy(signal).float().cuda().contiguous()
        else:
            signal = torch.from_numpy(signal).float().contiguous()
        lab_batch = self.lab_dict[self.wav_lst_te[index]]
        pout, lab = self.split_signals_into_chunks(lab_batch, signal)

        pred = torch.max(pout, dim=1)[1]
        loss = self.cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())

        [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
        return lab, best_class, loss, err

    def test(self, epoch, loss_tot, err_tot):
        """
        test the model in one epoch
        :param epoch:
        :param loss_tot:
        :param err_tot:
        :return:
        """
        self.CNN_net.eval()
        self.DNN1_net.eval()
        self.DNN2_net.eval()
        test_flag = 1
        loss_sum = 0
        err_sum = 0
        err_sum_snt = 0

        with torch.no_grad():
            for i in range(self.snt_te):
                [signal, fs] = sf.read(self.data_folder + self.wav_lst_te[i])
                lab, best_class, loss, err = self.test_one_signal(signal, i)
                err_sum_snt = err_sum_snt + (best_class != lab[0]).float()
                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()

            err_tot_dev_snt = err_sum_snt / self.snt_te
            loss_tot_dev = loss_sum / self.snt_te
            err_tot_dev = err_sum / self.snt_te

        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
            epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        with open(self.output_folder + '/' + self.kernel_type + "_res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))
        if err_tot_dev_snt <= self.best_err_tot_dev_snt:
            with open(self.save_path + '_best.pth', 'wb') as filehandler:
                pickle.dump(self, filehandler)
            self.best_err_tot_dev_snt = err_tot_dev_snt

    def train(self, resume_epoch=0):
        """
        trains a model from specified epoch
        :param resume_epoch:
        :return:
        """
        self.init_cnn_arch()
        self.init_dnn1_arch()
        self.init_dnn2_arch()

        if self.options['pt_file'] != 'none':
            print(f"Loading pre-trained model from {self.options['pt_file']} file")
            self.load_pretrain_model()
        self.initialize_optimizers()
        for epoch in range(resume_epoch, self.N_epochs):
            test_flag = 0
            self.CNN_net.train()
            self.DNN1_net.train()
            self.DNN2_net.train()

            loss_sum = 0
            err_sum = 0
            for i in range(self.N_batches):
                [inp, lab] = self.create_batches_rnd(self.wav_lst_tr, self.snt_tr, 0.2)
                loss, err = self.train_one_batch(inp, lab)
                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()
            loss_tot = loss_sum / self.N_batches
            err_tot = err_sum / self.N_batches

            if epoch % self.N_eval_epoch == 0:
                self.test(epoch, loss_tot, err_tot)
            else:
                print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))

    def init_cnn_arch(self):
        """
        initialize convolution layers of the model based on options
        :return:
        """
        cnn_N_filt = list(map(int, self.options['cnn_N_filt'].split(',')))
        cnn_len_filt = list(map(int, self.options['cnn_len_filt'].split(',')))
        cnn_max_pool_len = list(map(int, self.options['cnn_max_pool_len'].split(',')))
        cnn_use_laynorm_inp = GeneralUtils.str_to_bool(self.options['cnn_use_laynorm_inp'])
        cnn_use_batchnorm_inp = GeneralUtils.str_to_bool(self.options['cnn_use_batchnorm_inp'])
        cnn_use_laynorm = list(map(GeneralUtils.str_to_bool, self.options['cnn_use_laynorm'].split(',')))
        cnn_use_batchnorm = list(map(GeneralUtils.str_to_bool, self.options['cnn_use_batchnorm'].split(',')))
        cnn_act = list(map(str, self.options['cnn_act'].split(',')))
        cnn_drop = list(map(float, self.options['cnn_drop'].split(',')))
        CNN_arch = {'input_dim': self.wlen,
                    'fs': self.fs,
                    'cnn_N_filt': cnn_N_filt,
                    'cnn_len_filt': cnn_len_filt,
                    'cnn_max_pool_len': cnn_max_pool_len,
                    'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                    'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                    'cnn_use_laynorm': cnn_use_laynorm,
                    'cnn_use_batchnorm': cnn_use_batchnorm,
                    'cnn_act': cnn_act,
                    'cnn_drop': cnn_drop,
                    'kernel_type': self.kernel_type
                    }
        self.CNN_net = net_model(CNN_arch)
        if self.gpu:
            self.CNN_net.cuda()

    def init_dnn1_arch(self):
        """
        initialize fully connected sections of the model based on options
        :return:
        """
        fc_lay = list(map(int, self.options['fc_lay'].split(',')))
        fc_drop = list(map(float, self.options['fc_drop'].split(',')))
        fc_use_laynorm_inp = GeneralUtils.str_to_bool(self.options['fc_use_laynorm_inp'])
        fc_use_batchnorm_inp = GeneralUtils.str_to_bool(self.options['fc_use_batchnorm_inp'])
        fc_use_batchnorm = list(map(GeneralUtils.str_to_bool, self.options['fc_use_batchnorm'].split(',')))
        fc_use_laynorm = list(map(GeneralUtils.str_to_bool, self.options['fc_use_laynorm'].split(',')))
        fc_act = list(map(str, self.options['fc_act'].split(',')))
        DNN1_arch = {'input_dim': self.CNN_net.out_dim,
                     'fc_lay': fc_lay,
                     'fc_drop': fc_drop,
                     'fc_use_batchnorm': fc_use_batchnorm,
                     'fc_use_laynorm': fc_use_laynorm,
                     'fc_use_laynorm_inp': fc_use_laynorm_inp,
                     'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
                     'fc_act': fc_act,
                     }

        self.DNN1_net = MLP(DNN1_arch)
        if self.gpu:
            self.DNN1_net.cuda()

    def init_dnn2_arch(self):
        """
        initialize classifier section of the model based on options
        :return:
        """
        fc_lay = list(map(int, self.options['fc_lay'].split(',')))
        class_drop = list(map(float, self.options['class_drop'].split(',')))
        class_use_laynorm_inp = GeneralUtils.str_to_bool(self.options['class_use_laynorm_inp'])
        class_use_batchnorm_inp = GeneralUtils.str_to_bool(self.options['class_use_batchnorm_inp'])
        class_use_batchnorm = list(map(GeneralUtils.str_to_bool, self.options['class_use_batchnorm'].split(',')))
        class_use_laynorm = list(map(GeneralUtils.str_to_bool, self.options['class_use_laynorm'].split(',')))
        class_act = list(map(str, self.options['class_act'].split(',')))
        DNN2_arch = {'input_dim': fc_lay[-1],
                     'fc_lay': self.class_lay,
                     'fc_drop': class_drop,
                     'fc_use_batchnorm': class_use_batchnorm,
                     'fc_use_laynorm': class_use_laynorm,
                     'fc_use_laynorm_inp': class_use_laynorm_inp,
                     'fc_use_batchnorm_inp': class_use_batchnorm_inp,
                     'fc_act': class_act,
                     }

        self.DNN2_net = MLP(DNN2_arch)
        if self.gpu:
            self.DNN2_net.cuda()

    def load_pretrain_model(self):
        """
        loads the pretrained model to resume training from it
        :return:
        """
        if self.gpu:
            checkpoint_load = torch.load(self.options['pt_file'])
        else:
            checkpoint_load = torch.load(self.options['pt_file'], map_location=torch.device('cpu'))
        self.CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
        self.DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
        self.DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

    def initialize_optimizers(self):
        """
        initialize optimizers of each section of the model
        :return:
        """
        self.optimizer_CNN = optim.RMSprop(self.CNN_net.parameters(), lr=self.lr, alpha=0.95, eps=1e-8)
        self.optimizer_DNN1 = optim.RMSprop(self.DNN1_net.parameters(), lr=self.lr, alpha=0.95, eps=1e-8)
        self.optimizer_DNN2 = optim.RMSprop(self.DNN2_net.parameters(), lr=self.lr, alpha=0.95, eps=1e-8)

    def create_batches_rnd(self, wav_lst, N_snt, fact_amp):
        """
        creates random batches from data to use in training process
        :param wav_lst:
        :param N_snt:
        :param fact_amp:
        :return:
        """
        # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
        sig_batch = np.zeros([self.batch_size, self.wlen])
        lab_batch = np.zeros(self.batch_size)

        snt_id_arr = np.random.randint(N_snt, size=self.batch_size)

        rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, self.batch_size)

        for i in range(self.batch_size):
            [signal, fs] = sf.read(self.data_folder + wav_lst[snt_id_arr[i]])

            # accesing to a random chunk
            snt_len = signal.shape[0]
            snt_beg = np.random.randint(snt_len - self.wlen - 1)  # randint(0, snt_len-2*wlen-1)
            snt_end = snt_beg + self.wlen

            channels = len(signal.shape)
            if channels == 2:
                print('WARNING: stereo to mono: ' + self.data_folder + wav_lst[snt_id_arr[i]])
                signal = signal[:, 0]
            selected_signal_part = signal[snt_beg:snt_end]
            sig_batch[i, :] = selected_signal_part * rand_amp_arr[i]
            lab_batch[i] = self.lab_dict[wav_lst[snt_id_arr[i]]]
        if self.gpu:
            inp = Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
            lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
        else:
            inp = Variable(torch.from_numpy(sig_batch).float().contiguous())
            lab = Variable(torch.from_numpy(lab_batch).float().contiguous())

        return inp, lab
