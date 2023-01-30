import librosa
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
import torch.nn.functional as F
from scipy.signal import butter, lfilter, freqz


class SignalUtils:
    @staticmethod
    def kernel_sinc(f1, f2, t):
        """
        builds kernel sinc function based on f1, f2
        :param f1:
        :param f2:
        :param t:
        :return:
        """
        b = f2 - f1
        fc = (f1 + f2) / 2
        y_right = 2 * b * (torch.sin(b * t) / (b * t)) * torch.cos(2 * torch.pi * fc * t)
        y_left = SignalUtils.flip(y_right, 0)
        if torch.cuda.is_available():
            y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])
        else:
            y = torch.cat([y_left, Variable(torch.ones(1)), y_right])
        return y

    @staticmethod
    def kernel_sinc2(A, f1, f2, t):
        """
        builds kernel sinc2 function based on f1, f2
        :param A:
        :param f1:
        :param f2:
        :param t:
        :return:
        """
        b = f2 - f1
        fc = (f1 + f2) / 2
        y_right = A * (torch.pow(torch.sin(b * t), 2) / (b * t)) * torch.cos(2 * torch.pi * fc * t)
        y_left = SignalUtils.flip(y_right, 0)
        if torch.cuda.is_available():
            y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])
        else:
            y = torch.cat([y_left, Variable(torch.ones(1)), y_right])
        return y

    @staticmethod
    def kernel_gamma(A, f1, f2, order, t):
        """
        builds gamma kernel function based on f1, f2 and Amplitude
        :param A:
        :param f1:
        :param f2:
        :param order:
        :param t:
        :return:
        """
        fc = (f1+f2) / 2
        b = f2 - f1
        y_right = 1e11 * A * torch.pow(t, order - 1) * torch.exp(-2 * torch.pi * b * t) * torch.cos(2 * torch.pi * fc * t)
        return y_right

    @staticmethod
    def kernel_gauss(A, f1, f2, t):
        """
        builds gauss kernel function based on f1, f2 and Amplitude
        :param A:
        :param f1:
        :param f2:
        :param t:
        :return:
        """
        b = f2 - f1
        fc = (f1+f2) / 2
        if b < 1:
            b = 1
        sigma = torch.sqrt(torch.log10(torch.tensor(2))) / (b)
        y_right = (A * torch.exp((-1 * torch.pow(t, 2)) / (2 * torch.pow(sigma, 2)))) * torch.cos(2 * torch.pi * fc * t)
        return y_right

    @staticmethod
    def get_mel_freqs(N_filt, fs):
        """
        gets mel scale frequencies
        :param N_filt:
        :param fs:
        :return:
        """
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        mel_freqs = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        return mel_freqs

    @staticmethod
    def get_f1_f2(mel_freqs, fs):
        """
        initialize f1 and f2 frequencies of the kernelized model with mel scale frequencies
        :param mel_freqs:
        :param fs:
        :return:
        """
        f1 = np.roll(mel_freqs, 1)
        f2 = mel_freqs  # np.roll(mel_freqs, -1)
        f1[0] = 30
        return f1, f2

    @staticmethod
    def get_f1_f4(mel_freqs, fs, N_filt):
        """
        initialize f1, f2, f3, f4 frequencies of the kernelized model with mel scale frequencies
        :param mel_freqs:
        :param fs:
        :return:
        """
        mel_freqs = np.insert(mel_freqs, 0, 30., axis=0)
        f1 = np.zeros(N_filt)
        f2 = np.zeros(N_filt)
        f3 = np.zeros(N_filt)
        f4 = np.zeros(N_filt)
        n_assigned_filts = 0
        for i in range(mel_freqs.__len__()-1):
            for j in range(i+1, mel_freqs.__len__()-1):
                if n_assigned_filts < N_filt:
                    f1[n_assigned_filts] = mel_freqs[i]
                    f2[n_assigned_filts] = mel_freqs[i+1]
                    f3[n_assigned_filts] = mel_freqs[j]
                    f4[n_assigned_filts] = mel_freqs[j+1]
                    n_assigned_filts += 1
        return f1, f2, f3, f4

    @staticmethod
    def flip(x, dim):
        """
        flips x list in specified dimension
        :param x:
        :param dim:
        :return:
        """
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.contiguous()
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                     -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)

    @staticmethod
    def get_freq_domain_filter(signal, N):
        """
        gets N-point fft of a signal
        :param signal:
        :param N:
        :return:
        """
        f = np.fft.fft(signal, n=N)
        abs_f = np.abs(f[:N // 2])
        abs_f_db = 20 * np.log10(abs_f + 1e-6)
        return abs_f_db, abs_f

    @staticmethod
    def get_phase_of_filter(signal, N):
        """
        gets phase response of the signal
        :param signal:
        :param N:
        :return:
        """
        f = np.fft.fft(signal, n=N)
        return np.unwrap(np.angle(f[:N // 2]))

    @staticmethod
    def get_cnn_learned_filters(model, fs):
        """
        gets CNN model learned filters
        :param model:
        :param fs:
        :return:
        """
        conv_weights = model['CNN_model_par']['conv.0.weight']
        N = conv_weights.shape[2]
        N_filt = conv_weights.shape[0]
        n = torch.linspace(0, N, steps=N)
        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / N)
        window = Variable(window.float())
        time_domain_filters = np.zeros([N_filt, N])
        freq_domain_filters = np.zeros([N_filt, fs // 2])
        phase_of_filters = np.zeros([N_filt, fs // 2])
        f1_list = []
        f2_list = []
        freq_centers = []
        amp_list = []
        for i in range(N_filt):
            impulse_response = conv_weights[i, 0, :]
            time_domain_filters[i, :] = impulse_response
            freq_domain_filters[i, :], _ = SignalUtils.get_freq_domain_filter(impulse_response, fs)
        return time_domain_filters, freq_domain_filters, phase_of_filters, freq_centers, f1_list, f2_list, amp_list

    @staticmethod
    def get_gauss_cascade_learned_filters(model, fs, N):
        """
        gets GausseCascade model learned filters
        :param model:
        :param fs:
        :param N:
        :return:
        """
        norm_f1_list = model['CNN_model_par']['conv.0.norm_f1_list']
        norm_f2_list = model['CNN_model_par']['conv.0.norm_f2_list']
        norm_f3_list = model['CNN_model_par']['conv.0.norm_f3_list']
        norm_f4_list = model['CNN_model_par']['conv.0.norm_f4_list']
        amplitude_list1 = model['CNN_model_par']['conv.0.amplitude_list1']
        amplitude_list2 = model['CNN_model_par']['conv.0.amplitude_list2']
        N_filt = len(norm_f1_list)
        t = Variable(torch.linspace(1, N, steps=N) / fs)
        n = torch.linspace(0, N, steps=N)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / N)
        window = Variable(window.float())

        min_freq = 50.0
        min_band = 50.0

        f1_freq = torch.abs(norm_f1_list) + min_freq / fs
        f1_freq = torch.clip(f1_freq, min=0, max=0.5)
        f2_freq = f1_freq + torch.abs(norm_f2_list - f1_freq) + min_freq / fs
        f2_freq = torch.clip(f2_freq, min=0, max=0.5)
        f3_freq = torch.abs(norm_f3_list) + min_freq / fs
        f3_freq = torch.clip(f3_freq, min=0, max=0.5)
        f4_freq = f3_freq + torch.abs(norm_f4_list - f3_freq) + min_freq / fs
        f4_freq = torch.clip(f4_freq, min=0, max=0.5)
        amplitude1 = torch.abs(amplitude_list1)
        amplitude2 = torch.abs(amplitude_list2)

        time_domain_filters = np.zeros([N_filt, N])
        freq_domain_filters = np.zeros([N_filt, fs // 2])
        phase_of_filters = np.zeros([N_filt, fs // 2])
        f1_list = []
        f2_list = []
        freq_centers = []
        amp_list = []
        for i in range(N_filt):
            f1 = f1_freq[i].float() * fs
            f2 = f2_freq[i].float() * fs
            f3 = f3_freq[i].float() * fs
            f4 = f4_freq[i].float() * fs
            amp1 = amplitude1[i].float()
            amp2 = amplitude2[i].float()
            f1_list.append([f1.float().numpy()*1, f2.float().numpy()*1])
            f2_list.append([f3.float().numpy()*1, f4.float().numpy()*1])
            freq_centers.append([(f1.numpy() + f2.numpy()) / (2 * fs), (f3.numpy() + f4.numpy()) / (2 * fs)])

            impulse_response1 = SignalUtils.kernel_gauss(amp1, f1, f2, t)
            impulse_response1 = ((2 * (impulse_response1 - torch.min(impulse_response1))) / (
                    torch.max(impulse_response1) - torch.min(impulse_response1) + 1e-6)) - 1
            impulse_response1 = (impulse_response1 - torch.mean(impulse_response1))

            impulse_response2 = SignalUtils.kernel_gauss(amp2, f3, f4, t)
            impulse_response2 = ((2 * (impulse_response2 - torch.min(impulse_response2))) / (
                    torch.max(impulse_response2) - torch.min(impulse_response2) + 1e-6)) - 1
            impulse_response2 = (impulse_response2 - torch.mean(impulse_response2))

            impulse_response = F.conv1d(impulse_response1.view(1, 1, N),
                                        impulse_response2.view(1, 1, N),
                                        padding=N // 2).view(N)
            impulse_response = ((2 * (impulse_response - torch.min(impulse_response))) / (
                    torch.max(impulse_response) - torch.min(impulse_response) + 1e-6)) - 1
            impulse_response = (impulse_response - torch.mean(impulse_response))
            impulse_response = impulse_response * window

            time_domain_filters[i, :] = impulse_response
            freq_domain_filters[i, :], _ = SignalUtils.get_freq_domain_filter(impulse_response, fs)
        return time_domain_filters, freq_domain_filters, phase_of_filters, freq_centers, f1_list, f2_list, amp_list

    @staticmethod
    def get_learned_filters(model_name, model, fs, N):
        """
        gets leaerned filters of kernelized models based on the learnt model
        :param model_name:
        :param model:
        :param fs:
        :param N:
        :return:
        """
        if model_name == 'CNN':
            return SignalUtils.get_cnn_learned_filters(model, fs)
        elif model_name == 'Gauss_Cascade':
            return SignalUtils.get_gauss_cascade_learned_filters(model, fs, N)
        norm_f1_list = model['CNN_model_par']['conv.0.norm_f1_list']
        norm_f2_list = model['CNN_model_par']['conv.0.norm_f2_list']
        N_filt = len(norm_f1_list)
        if model_name != 'Sinc':
            amplitude_list = model['CNN_model_par']['conv.0.amplitude_list']
            amplitude = torch.abs(amplitude_list)

        t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / fs)
        t = Variable(torch.linspace(1, N, steps=N) / fs)

        min_freq = 50.0
        min_band = 50.0

        f1_freq = torch.abs(norm_f1_list) + min_freq / fs
        f1_freq = np.clip(f1_freq, a_min=0, a_max=0.5)
        f2_freq = f1_freq + torch.abs(norm_f2_list - f1_freq) + min_freq / fs
        f2_freq = np.clip(f2_freq, a_min=0, a_max=0.5)

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / N)
        window = Variable(window.float())

        time_domain_filters = np.zeros([N_filt, N])
        freq_domain_filters_db = np.zeros([N_filt, fs // 2])
        freq_domain_filters_abs = np.zeros([N_filt, fs // 2])

        phase_of_filters = np.zeros([N_filt, fs // 2])
        f1_list = []
        f2_list = []
        freq_centers = []
        amp_list = []
        for i in range(N_filt):
            f1 = f1_freq[i].float() * fs
            f2 = f2_freq[i].float() * fs
            f1_list.append(f1.numpy())
            f2_list.append(f2.numpy())
            freq_centers.append((f1.numpy() + f2.numpy()) / (2 * fs))
            amp = 1
            if model_name != 'Sinc':
                amp = amplitude[i].float()
            amp_list.append(amp)

            if model_name == 'Sinc':
                impulse_response = SignalUtils.kernel_sinc(f1, f2, t_right)
                h_n = impulse_response
            if model_name == 'Sinc2':
                impulse_response = SignalUtils.kernel_sinc2(amp, f1, f2, t_right)
                h_n = impulse_response
            elif model_name == 'Gamma':
                impulse_response = SignalUtils.kernel_gamma(amp, f1, f2, 4, t)
                h_n = impulse_response
            elif model_name == 'Gauss':
                impulse_response = SignalUtils.kernel_gauss(amp, f1, f2, t)
                h_n = impulse_response

            impulse_response = impulse_response / torch.max(impulse_response)
            impulse_response = impulse_response * window
            h_n = h_n / torch.max(h_n)
            h_n = h_n * window

            time_domain_filters[i, :] = impulse_response

            freq_domain_filters_db[i, :], freq_domain_filters_abs[i, :] = SignalUtils.get_freq_domain_filter(impulse_response, fs)
            phase_of_filters[i, :] = SignalUtils.get_phase_of_filter(impulse_response, fs)

            freq_domain_filters_db[i, 0: 10] = freq_domain_filters_db[i, 10]
            phase_of_filters[i, 0: 10] = phase_of_filters[i, 10]
        return time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list
