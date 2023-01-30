# Adding Interpretability to CNNs Using Auditory Filter Models
This repository contains an extended version of SincNet [1] in which some general auditory filter models are added for the Speaker Identification (SID) task, which is presented in "Analyzing the Use of Auditory Filter Models for Making Interpretable Convolutional Neural Networks for Speaker Identification" [Slides] (https://github.com/HosseinFayyazi/SincNet/blob/master/IO/imgs/rsc/csicc_2023_pres.pdf). The goal here is understanding hearing models and adapting learning models closely to human hearing, examining the use of some auditory filter models as CNN front-ends and finally evaluating the resulted filter banks in the SID task. In the paper, rectangular, triangular, gammatone, Gaussian and cascaded filter types are selected to examine. 


## Model Architecture
One traditional view in the functional level description of the process occurring in the cochlea in the inner ear is that it acts as a frequency analyzer using some specific filter types. This view corresponds to the learning of a meaningful filter bank in the first layer of a CNN. High-level extracted features from these layers are then transferred to the central auditory nervous system in the brain for more complicated operations, which can be simulated using fully connected layers in the last layers of a CNN architecture.
The architecture used here is the one presented in SincNet. The following figure shows the details of this architecture and its correspondence with the human auditory system. 

<img src="https://github.com/HosseinFayyazi/InterpretableCNN/blob/master/IO/imgs/rsc/Architecture.png" width="400" img align="right">

## Results and Discussions
**Time and frequency domains shapes**
The impulse response and magnitude response of the three filters learned in each of the models are shown here. While audio filter models have a meaningful time domain shape and their magnitude response can be determined explicitly by a center frequency and bandwidth, the standard filters have unfamiliar, noisy shapes with no meaning. This property encourages the use of specific filter types as a strong replacement for standard ones to have a better understanding of the decision made by a CNN model. 

<img src="https://github.com/HosseinFayyazi/InterpretableCNN/blob/master/IO/imgs/rsc/Responses.png" width="400" img align="right">

**Learned filter bank**
The learned filter bank in the first layer of examined models is depicted here. The filters operate in very low frequencies are more than in high frequencies. While filters with sharper peaks are placed in lower frequencies, the peaks become shallower at high frequencies. These observations are consistent with the experiments that have been conducted on the filtering function of the human auditory system. The famous Mel-filterbank was inspired by this fact. 

<img src="https://github.com/HosseinFayyazi/InterpretableCNN/blob/master/IO/imgs/rsc/filterbanks.png" width="400" img align="right">


**Center frequencies**
To compare the distribution of the Mel-filterbanks with those presented here, the histogram of the center frequencies of the different filter banks is depicted here. It can be seen that the overall trend of learned filter banks is as the Mel-scale one, but the importance of frequencies close to 2 kHz is considered less in all models. In addition, the number of filters sensitive to high frequencies is not as low as the Mel-filterbanks. This feature reveals that in a specific application like SID, the fundamental frequency, below 1 kHz frequency, has more impact in distinguishing speakers than the two or three first formants of a speech signal. 
Moreover, some information related to speaker recognition is spread in higher frequencies. 

<img src="https://github.com/HosseinFayyazi/InterpretableCNN/blob/master/IO/imgs/rsc/overal_hist.png" width="400" img align="right">

**Quality Factor (QF)**
Quality Factor or QF, which is the fraction of the center frequency to the bandwidth of a filter, can be used to examine both parameters at the same time. The filters of a Mel-filterbank are designed based on auditory considerations, which have a lower bandwidth at low frequencies and a higher bandwidth at high frequencies. This fraction has a relatively gentle slope for these filters.  The best-fitted line to QFs of each model is considered, and their reflection on this line is depicted in the figure in the following picture. 
It is seen that the overall trend of QF for all filter types is incremental, and filters at high frequencies are further apart and have higher bandwidth. This property is the same as the Mel-scale with the difference in fitted line slope. The slope of the fitted lines of interpretable filters reveals the importance of higher frequencies in this specific task. 
The figure also reveals an interesting property of the filter bank appropriate for the SID task. The number of filters in 0 ~ 1.5 kHz and 2.5 ~ 4 kHz frequency bands is more than the others, which demonstrates that these frequency bands create more distinction between different speakers.

<img src="https://github.com/HosseinFayyazi/InterpretableCNN/blob/master/IO/imgs/rsc/qf.png" width="400" img align="right">

**Frequency analysis from speech production view**
The unique characteristics of a speaker are encoded during speech production. These features should be involved in the invariant factors in the physiology of the vocal tract. 
Some interesting findings presented in some researches are as follows: 
- The function of different articulatory speech organs that make speaker-dependent features lead to non-uniformly distribution of these features in high frequency bands. 
- glottis information is encoded between 100 Hz and 400 Hz, and 
- The information of the piriform fossa is encoded between 4 kHz and 5 kHz. 

These facts obviously show that the proper filter bank is completely task-dependent. For example, the first three formants which are encoded in 200 Hz to 3 kHz frequency region, are important in phone recognition while this region has lower importance in the SID task. 
The following figure shows the histogram of the learned filters of the gammatone model by a smaller bin size. It is seen that most filters operate in 0 ~ 250 Hz where glottis information is encoded. Other filters are operated in high frequencies with a non-uniform distribution and frequencies related to speech formants are not emphasized as much as Mel-filterbanks.

<img src="https://github.com/HosseinFayyazi/InterpretableCNN/blob/master/IO/imgs/rsc/gamma_hist.png" width="400" img align="right">


## How to run?
Even though the code can be easily adapted to any speech dataset, in the following part of the documentation we provide an example based on the popular TIMIT dataset.

**1. Run TIMIT data preparation**

This step is necessary to store a version of TIMIT in which start and end silences are removed and the amplitude of each speech utterance is normalized. To do it, run the following code:

``
python timit_prepare_script.py --in_folder $TIMIT_FOLDER --out_folder $OUTPUT_FOLDER
``

where:
- *$TIMIT_FOLDER* is the folder of the original TIMIT corpus
- *$OUTPUT_FOLDER* is the folder in which the normalized TIMIT will be stored

In TIMIT, each speaker reads ten phonetically rich sentences, two of which are calibration sentences designed to allow cross-speaker comparisons. Here, these sentences are removed. Five of the eight remaining sentences are used for training, two for validation, and one for testing, which leads to a 630-class classification problem.  

**2. Run the speaker id experiment**

- Modify the *[data]* section of *IO/$MODEL_NAME/$FILE.cfg* file according to your paths, where *$MODEL_NAME* is the model name that can be one of the following ones, 'cnn', 'sinc', 'sinc2', 'gamma', 'gauss' and 'gauss_cascade'. 
In particular, modify the *data_folder* with the *$OUTPUT_FOLDER* specified during the TIMIT preparation. The other parameters of the config file belong to the following sections:
 1. *[windowing]*, that defines how each sentence is split into smaller chunks.
 2. *[cnn]*,  that specifies the characteristics of the CNN architecture.
 3. *[dnn]*,  that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
 4. *[class]*, that specify the softmax classification part.
 5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

- Once setup the cfg file, you can run the speaker id experiments using the following command:

``
python speaker_id_script.py --cfg $CFG_FILE --resume_epoch $EPOCH_NUMBER --resume_model_path $RESUME_MODEL_PATH --save_path $SAVE_PATH
``

where:
- *$CFG_FILE* is the cfg file path
- *$EPOCH_NUMBER* resumes training from this epoch, which its default value is 0
- *$RESUME_MODEL_PATH* resumes training from the model with specified path
- *$SAVE_PATH* is save path of the model



**3. Results**

The results are saved into the *output_folder* specified in the cfg file. In this folder, you can find a file (*res.res*) summarizing training and test error rates. The model *saved_model.pth* is the model saved after the last iteration and the model *saved_model.pth_best.pth* is the model with best Classification Error Rate on validation data.

The fields of the res.res file have the following meaning:
- loss_tr: is the average training loss (i.e., cross-entropy function) computed at every frame.
- err_tr: is the classification error (measured at frame level) of the training data. Note that we split the speech signals into chunks of 200ms with 10ms overlap. The error is averaged for all the chunks of the training dataset.
- loss_te is the average test loss (i.e., cross-entropy function) computed at every frame.
- err_te: is the classification error (measured at frame level) of the test data.
- err_te_snt: is the classification error (measured at sentence level) of the test data. Note that we split the speech signal into chunks of 200ms with 10ms overlap. For each chunk, the model performs a prediction over the set of speakers. To compute this classification error rate the predictions are averaged and, for each sentence, the speaker with the highest average probability is voted.

**4. Evaluation**

The final model obtained from training process can be evaluated on test data using the following command:

``
python test_eval_script.py --cfg_file $CFG_FILE --save_path $SAVE_PATH
``

where:
- *$CFG_FILE* is the cfg file path
- *$SAVE_PATH* is save path of the model

**5. Visualizing the results**
For plotting the images shown in "Results and Discussions" section, use the following script:

``
python plot_filter_script.py --model_name $MODEL_NAME --model_path $MODEL_PATH --cfg_file $CFG_FILE --out_path $OUT_PATH
``

where:
- *$MODEL_NAME* is one of the models, 'CNN', 'Sinc', 'Sinc2', 'Gamma', 'Gauss' and 'Gauss_Cascade'
- *$MODEL_PATH* is save path of the model
- *$CFG_FILE* is the cfg file path
- *$OUT_PATH* is save path of the resulted figures



## Main References

[1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet”
[2] Erfan Loweimi, Peter Bell, Steve Renals, “On Learning Interpretable CNNs with Parametric Modulated Kernel-Based Filters”
[3] Lyon, Richard F., “Human and machine hearing: extracting meaning from sound”
[4] Hossein Fayyazi, Yasser Shekofteh, “Analyzing the Use of Auditory Filter Models for Making Interpretable Convolutional Neural Networks for Speaker Identification” 
