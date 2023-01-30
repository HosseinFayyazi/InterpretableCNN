import matplotlib.pyplot as plt
import numpy as np

from Utils.SignalUtils import *
from Utils.GeneralUtils import *
from Utils.PlotUtils import *
import argparse
import matplotlib.pylab as pylab
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)

print('Initializing parameters ...')
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Sinc', choices=
                    ['CNN', 'Sinc', 'Sinc2', 'Gamma', 'Gauss', 'Gauss_Cascade'])
parser.add_argument('--model_path', type=str, default='IO/sinc/saved_model.pth')
parser.add_argument('--cfg_file', type=str, default='IO/sinc/KernelSinc_TIMIT.cfg')
parser.add_argument('--out_path', type=str, default='IO/imgs/sinc/')
args = parser.parse_args()

# -------------------------------------------------------------------------------------------------------------------- #
print(f'Loading {args.model_name} model ...')
with open(args.model_path, 'rb') as file:
    trainer = CPU_Unpickler(file).load()
trainer.gpu = False
model = {'CNN_model_par': trainer.CNN_net.state_dict(),
         'DNN1_model_par': trainer.DNN1_net.state_dict(),
         'DNN2_model_par': trainer.DNN2_net.state_dict()
         }
# model = torch.load(args.model_path, map_location=torch.device('cpu'))
fs = 16000
options = GeneralUtils.read_conf(args.cfg_file)
N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])

time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
    SignalUtils.get_learned_filters(args.model_name, model, fs, N)

# check folder exists
GeneralUtils.check_folder_exist(args.out_path)

# -------------------------------------------------------------------------------------------------------------------- #
print('Saving all filters diagram ...')
diagrams = []
if args.model_name != 'CNN':
    freq_domain_filters_db1 = np.nan_to_num(freq_domain_filters_db)
    if args.model_name == 'Gamma' or args.model_name == 'Gauss':
        for i in range(freq_domain_filters_db1.shape[0]):
            freq_domain_filters_db1[i, :] = ((2 * (freq_domain_filters_db1[i, :] - np.min(freq_domain_filters_db1[i, :]))) / (
                    np.max(freq_domain_filters_db1[i, :]) - np.min(freq_domain_filters_db1[i, :]) + 1e-6)) - 1
            freq_domain_filters_db1[i, :] = 20 * (freq_domain_filters_db1[i, :] - np.mean(freq_domain_filters_db1[i, :]))
    ymin = 0 # np.min(freq_domain_filters_db1)
    ymax = np.max(freq_domain_filters_db1)
    sorted_indices = sorted(range(len(freq_centers)), key=lambda k: freq_centers[k])
    for i in sorted_indices:
        tuples = []
        for j in range(freq_domain_filters_db1.shape[1]):
            val = freq_domain_filters_db1[i, j]
            if val < 0:
                val = 0
            tuples.append((j, val))
        diagrams.append(tuples)
    PlotUtils.draw_3d(diagrams, diagrams, ymin, ymax, colors=False)
    plt.savefig(args.out_path + '_AllInOne0.png')
    plt.close()

# -------------------------------------------------------------------------------------------------------------------- #
print('Saving all frequency responses of learned filters in one figure ...')
for i in range(N_filt):
    plt.plot(range(fs//2), freq_domain_filters_db[i, :], 'g', linewidth=0.2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title(f"Freq. domain Filters learnt with {args.model_name}")
plt.savefig(args.out_path + '_freq_responses.png')
plt.close()

print('Saving All in one ...')
n_cols = 8
n_rows = int(np.ceil((N_filt * 2) / n_cols))
counter = 1
ax = plt.figure()

for i in range(N_filt):
    plt.subplot(n_rows, n_cols, counter)
    plt.plot(range(N), time_domain_filters[i, :], linewidth=0.2)
    ax.axes[counter - 1].xaxis.set_visible(False)
    ax.axes[counter - 1].yaxis.set_visible(False)
    counter += 1
    plt.subplot(n_rows, n_cols, counter)
    nf = torch.linspace(0, 0.5, steps=int(fs // 2)).numpy()
    plt.plot(nf, freq_domain_filters_db[i, :], 'g', linewidth=0.2)
    ax.axes[counter - 1].xaxis.set_visible(False)
    ax.axes[counter - 1].yaxis.set_visible(False)
    counter += 1
plt.savefig(args.out_path + '_AllInOne.png', dpi=900)
plt.close()

# -------------------------------------------------------------------------------------------------------------------- #
freq_centers1 = fs * np.clip(freq_centers, a_min=0, a_max=0.5) / 1000
print('Saving the histogram of frequency centers of learned filters ...')
plt.hist(freq_centers1, bins=fs//500)
plt.xlabel("Frequency (kHz)")
plt.ylabel("#filters")
plt.title(f"Center frequency histogram of {args.model_name} learned filters")
plt.savefig(args.out_path + '_freqCentersHist.png')
plt.close()

# -------------------------------------------------------------------------------------------------------------------- #
print('Saving histogram of center frequencies learnt by all models in one figure ...')
model_names = ['Mel-Scale', 'Sinc', 'Sinc2', 'Gamma', 'Gauss']
model_labels = ['Mel-Scale', 'rectangular', 'triangular', 'gammatone', 'gaussian']
model_paths = ['IO/cnn/saved_model.pth',
               'IO/sinc/saved_model.pth',
               'IO/sinc2/saved_model.pth',
               'IO/gamma/saved_model.pth',
               'IO/gauss/saved_model.pth'
               ]
cfg_files = ['IO/cnn/KernelCNN_TIMIT.cfg',
             'IO/sinc/KernelSinc_TIMIT.cfg',
             'IO/sinc2/KernelSinc2_TIMIT.cfg',
             'IO/gamma/KernelGamma_TIMIT.cfg',
             'IO/gauss/KernelGauss_TIMIT.cfg'
             ]
marker = ['o-', 's-', 'x-', 'p-', '>-', '<-', '^-', 'v-', 'D-']
for i, model_name in enumerate(model_names):
    if model_name == 'Mel-Scale':
        freq_centers = SignalUtils.get_mel_freqs(N_filt, fs)
    else:
        model_path = model_paths[i]

        with open(model_path, 'rb') as file:
            trainer = CPU_Unpickler(file).load()
        trainer.gpu = False
        model = {'CNN_model_par': trainer.CNN_net.state_dict(),
                 'DNN1_model_par': trainer.DNN1_net.state_dict(),
                 'DNN2_model_par': trainer.DNN2_net.state_dict()
                 }

        # model = torch.load(model_path, map_location=torch.device('cpu'))
        cfg_file = cfg_files[i]
        options = GeneralUtils.read_conf(cfg_file)
        N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
        N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])
        time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
            SignalUtils.get_learned_filters(model_name, model, fs, N)
    hist, _ = np.histogram(freq_centers, bins=fs//2000)
    if model_name != 'Mel-Scale':
        model_name += 'Net'
    plt.plot(range(1, len(hist)+1), hist, marker[i], linewidth=0.7, label=model_labels[i])
plt.xlabel("Center Frequency (kHz)")
plt.ylabel("#filters")
plt.legend()
plt.savefig(args.out_path + '_overal_hist.png')
plt.close()

# -------------------------------------------------------------------------------------------------------------------- #
print("Saving QF for all models ...")
for i, model_name in enumerate(model_names):
    if model_name == 'Mel-Scale':
        freq_centers = SignalUtils.get_mel_freqs(N_filt, fs)
        f1_list, f2_list = SignalUtils.get_f1_f2(freq_centers, fs)
        f1_list[0] = 0
    else:
        model_path = model_paths[i]

        with open(model_path, 'rb') as file:
            trainer = CPU_Unpickler(file).load()
        trainer.gpu = False
        model = {'CNN_model_par': trainer.CNN_net.state_dict(),
                 'DNN1_model_par': trainer.DNN1_net.state_dict(),
                 'DNN2_model_par': trainer.DNN2_net.state_dict()
                 }

        # model = torch.load(model_path, map_location=torch.device('cpu'))
        cfg_file = cfg_files[i]
        options = GeneralUtils.read_conf(cfg_file)
        N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
        N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])
        time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
            SignalUtils.get_learned_filters(model_name, model, fs, N)

    qf = []
    cf = []
    for j in range(N_filt):
        w0 = (f1_list[j] + f2_list[j]) / 2
        sigma = f2_list[j] - f1_list[j]
        if sigma == 0:
            sigma = 50
        cf.append(w0)
        qf.append(w0 / (sigma))

    if model_name != 'Mel-Scale':
        model_name = model_labels[i]

    x = np.unique(cf)/1000
    y = np.poly1d(np.polyfit(cf, qf, 1))(np.unique(cf))
    if y[0] < 0:
        y = y - y[0]
    if model_name == 'Mel-Scale':
        plt.plot(x, qf, '.', label=model_labels[i], linewidth=0.5)
    else:
        plt.plot(x, y, '.', label=model_labels[i], linewidth=0.5)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Quality Factor")
plt.legend()
plt.savefig(args.out_path + '_qf1.png')
plt.close()

# -------------------------------------------------------------------------------------------------------------------- #
print(f'Saving the images in {args.out_path} directory ...')
with open(args.model_path, 'rb') as file:
    trainer = CPU_Unpickler(file).load()
trainer.gpu = False
model = {'CNN_model_par': trainer.CNN_net.state_dict(),
         'DNN1_model_par': trainer.DNN1_net.state_dict(),
         'DNN2_model_par': trainer.DNN2_net.state_dict()
         }

options = GeneralUtils.read_conf(args.cfg_file)
N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])
time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
    SignalUtils.get_learned_filters(args.model_name, model, fs, N)
for i in range(N_filt):
    plt.rcParams["figure.figsize"] = (12, 3.5)

    plt.subplot(1, 2, 1)
    plt.plot(range(N), time_domain_filters[i, :], linewidth=0.5)
    plt.xlabel("Samples")
    plt.ylabel("h[n]")

    ax1 = plt.subplot(1, 2, 2)
    ax1.plot(range(fs//2), freq_domain_filters_db[i, :], 'b', linewidth=0.5)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel('Mag. Resp. (dB)', color='k')

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(args.out_path + str(i+1) + '.png')
    plt.close()

print('Operation completed successfully!')
