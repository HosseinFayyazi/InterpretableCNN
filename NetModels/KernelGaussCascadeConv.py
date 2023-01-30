from Utils.SignalUtils import *
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class KernelGaussCascadeConv(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs):
        super(KernelGaussCascadeConv, self).__init__()
        self.gpu = torch.cuda.is_available()
        # Mel Initialization of the filterbanks
        n_mel_freqs = int(np.ceil((-1 + np.sqrt(1 + 8 * N_filt)) / 2)) + 1
        mel_freqs = SignalUtils.get_mel_freqs(n_mel_freqs, fs)
        f1, f2, f3, f4 = SignalUtils.get_f1_f4(mel_freqs, fs, N_filt)

        self.freq_scale = fs * 1.0
        self.norm_f1_list = nn.Parameter(torch.from_numpy(f1 / self.freq_scale))
        self.norm_f2_list = nn.Parameter(torch.from_numpy(f2 / self.freq_scale))
        self.norm_f3_list = nn.Parameter(torch.from_numpy(f3 / self.freq_scale))
        self.norm_f4_list = nn.Parameter(torch.from_numpy(f4 / self.freq_scale))
        self.amplitude_list1 = nn.Parameter(torch.ones(N_filt))
        self.amplitude_list2 = nn.Parameter(torch.ones(N_filt))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        N = self.Filt_dim
        if self.gpu:
            filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
            t = Variable(torch.linspace(1, N, steps=N) / self.fs).cuda()
        else:
            filters = Variable(torch.zeros((self.N_filt, self.Filt_dim)))
            t = Variable(torch.linspace(1, N, steps=N) / self.fs)

        min_freq = 50.0
        min_band = 50.0

        f1_freq = torch.abs(self.norm_f1_list) + min_freq / self.freq_scale
        f1_freq = torch.clip(f1_freq, min=0, max=0.5)
        f2_freq = f1_freq + torch.abs(self.norm_f2_list - f1_freq) + min_freq / self.freq_scale
        f2_freq = torch.clip(f2_freq, min=0, max=0.5)
        f3_freq = torch.abs(self.norm_f3_list) + min_freq / self.freq_scale
        f3_freq = torch.clip(f3_freq, min=0, max=0.5)
        f4_freq = f3_freq + torch.abs(self.norm_f4_list - f3_freq) + min_freq / self.freq_scale
        f4_freq = torch.clip(f4_freq, min=0, max=0.5)
        amplitude1 = torch.abs(self.amplitude_list1)
        amplitude2 = torch.abs(self.amplitude_list2)

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / N)
        if self.gpu:
            window = Variable(window.float().cuda())
        else:
            window = Variable(window.float())

        for i in range(self.N_filt):
            f1 = f1_freq[i].float() * self.freq_scale
            f2 = f2_freq[i].float() * self.freq_scale
            f3 = f3_freq[i].float() * self.freq_scale
            f4 = f4_freq[i].float() * self.freq_scale
            amp1 = amplitude1[i].float()
            amp2 = amplitude2[i].float()

            impulse_response1 = SignalUtils.kernel_gauss(amp1, f1, f2, t)
            # impulse_response1 = impulse_response1 / torch.max(impulse_response1)
            impulse_response1 = ((2 * (impulse_response1 - torch.min(impulse_response1))) / (
                    torch.max(impulse_response1) - torch.min(impulse_response1) + 1e-6)) - 1
            impulse_response1 = (impulse_response1 - torch.mean(impulse_response1))

            impulse_response2 = SignalUtils.kernel_gauss(amp2, f3, f4, t)
            # impulse_response2 = impulse_response2 / torch.max(impulse_response2)
            impulse_response2 = ((2 * (impulse_response2 - torch.min(impulse_response2))) / (
                    torch.max(impulse_response2) - torch.min(impulse_response2) + 1e-6)) - 1
            impulse_response2 = (impulse_response2 - torch.mean(impulse_response2))

            impulse_response = F.conv1d(impulse_response1.view(1, 1, self.Filt_dim), impulse_response2.view(1, 1, self.Filt_dim),
                     padding=N // 2).view(self.Filt_dim)
            # impulse_response = impulse_response / torch.max(impulse_response)
            impulse_response = ((2 * (impulse_response - torch.min(impulse_response))) / (
                    torch.max(impulse_response) - torch.min(impulse_response) + 1e-6)) - 1
            impulse_response = (impulse_response - torch.mean(impulse_response))

            if self.gpu:
                filters[i, :] = impulse_response.cuda() * window
            else:
                filters[i, :] = impulse_response * window
        out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim))
        return out
