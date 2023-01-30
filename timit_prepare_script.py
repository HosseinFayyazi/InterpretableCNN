import argparse
from Utils.GeneralUtils import *
import os


print('Initializing parameters ...')
parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str, default='../../Data/TIMIT/', help='path of saving the filter images')
parser.add_argument('--out_folder', type=str, default='../../Data/TIMIT_NORM', help='path of saving the filter images')
args = parser.parse_args()

print('Reading list of files ...')
list_sig = []
# r=root, d=directories, f = files
for r, d, f in os.walk(args.in_folder):
    for file in f:
        if '.WAV' in file:
            list_sig.append(os.path.join(r[len(args.in_folder):], file))

print('Replicating input folder structure to output folder ..')
GeneralUtils.copy_folder(args.in_folder, args.out_folder)

print('Removing silence sections from signals ...')
for i in range(len(list_sig)):
    wav_file = args.in_folder + '/' + list_sig[i]
    [signal, fs] = sf.read(wav_file)
    signal = signal.astype(np.float64)

    # Signal normalization
    signal = signal / np.max(np.abs(signal))

    # Read wrd file
    wrd_file = wav_file.replace(".WAV", ".WRD")
    wrd_sig = GeneralUtils.read_list(wrd_file)
    beg_sig = int(wrd_sig[0].split(' ')[0])
    end_sig = int(wrd_sig[-1].split(' ')[1])

    # Remove silences
    signal = signal[beg_sig:end_sig]

    # Save normalized speech
    file_out = args.out_folder + '/' + list_sig[i]
    os.makedirs(args.out_folder, exist_ok=True)

    sf.write(file_out, signal, fs)

    print(f'{file_out} processed.')
