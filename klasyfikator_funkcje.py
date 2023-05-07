from numpy import ndarray
import librosa
import pickle
import os
import numpy as np


def open_file(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def extract_mfcc(file_path: str or ndarray, fs=16000, frame_length=100) -> ndarray:
    """
    Function extracts mfcc matrix from signal
    :param file_path: audio path or signal array
    :param fs: sampling rate in Hertz
    :param frame_length: length of the frames in milliseconds
    :return:
            MFCC: ndarray
    """
    if os.path.exists(file_path):
        y, sr = librosa.load(file_path, sr=fs)
    elif type(file_path) == ndarray and type(fs) == int:
        y = file_path
        # y = librosa.resample(file_path, orig_sr=fs, target_sr=4000)

    frame_length_in_samples = fs * frame_length / 1000
    hop_length = frame_length_in_samples / 4
    MFCC = librosa.feature.mfcc(
        y=y, n_fft=int(frame_length_in_samples), hop_length=int(hop_length),
    )
    return MFCC


def feature(file_path: str, fs=16000) -> ndarray:
    """
    Function extract mfcc, delta and delta delta form signal
    :param file_path: audio file or signal array
    :param fs: sampling rate in Herz
    :return:
            features: ndarray with mfcc, delta and delta delta values
    """
    mfcc = extract_mfcc(file_path, fs, frame_length=100)
    delta_1 = librosa.feature.delta(mfcc)
    delta_2 = librosa.feature.delta(mfcc, order=2)
    features = np.hstack((mfcc, delta_1, delta_2)).T
    return features


def classify(filename_path: str or ndarray, gmm_models: list, fs=16000):
    """
    Function classify gender of speaker based on max log-likelihood
    :param filename_path: audio path or signal array
    :param gmms_model: list of trained gmms
    :param fs: sampling rate in Hertz
    :return:
            audio_scores: list with 2 log-likelihood values
            class_prediction: gender classfication of the recording
    """

    feature_proba = feature(filename_path, fs)
    class_name = ["k", "m"]

    audio_scores_gender = []
    for i in range(0, len(gmm_models)):
        audio_scores_gender.append(gmm_models[i].score(feature_proba))
    # print(audio_scores_gender)
    i_max = np.argmax(audio_scores_gender)
    class_prediction = class_name[i_max]

    # print(f' predykcja = {class_name[i_max]}')
    return audio_scores_gender, class_prediction


def extract_f0(wav_path: str or ndarray, fs=4000, frame_length=100):
    """
    Function detects f0 (without nan) from signal
    :param wav_path: audio path or signal array
    :param fs: sampling rate in Hertz
    :param frame_length: length of the frames in milliseconds
    :return:
            f0: array with f0 values
    """
    if os.path.exists(wav_path):
        y, sr = librosa.load(wav_path, sr=fs)
    elif type(wav_path) == ndarray and type(fs) == int:
        y = librosa.resample(wav_path, orig_sr=fs, target_sr=4000)
    else:
        raise ValueError(f"Type of given input should be Path or signal in ndarray. Type of input argument: {type(wav_path)}")
    frame_length_in_samples = fs * frame_length / 1000
    hop_length = frame_length_in_samples / 4
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=fs,
        frame_length=int(frame_length_in_samples),
        win_length=None,
        hop_length=int(hop_length),
        n_thresholds=100,
        beta_parameters=(2, 18),
        boltzmann_parameter=2,
        resolution=0.25,
        max_transition_rate=35.92,
        switch_prob=0.01,
        no_trough_prob=0.01,
        fill_na=None,
        center=True,
        pad_mode="constant",
    )

    indx = [i for i, vf in enumerate(voiced_flag) if vf]
    return f0-np.mean(f0) if indx else f0


def classify_intonation(
    filename_path: str or ndarray, gmms_model: list, fs=4000
) -> tuple[list, int]:
    """
    Function classify 3 types of intonation based on max log-likelihood
    :param filename_path: audio path or signal array
    :param gmms_model: list of trained gmms
    :param fs: sampling rate in Hertz
    :return:
            audio_scores: list with 3 log-likelihood values
            class_prediction: intanation classification of recording
    """
    f0 = extract_f0(filename_path, fs, frame_length=100)
    new_f0_test = f0.reshape(-1, 1)
    if len(f0) == 0:
        class_prediction = -1
        audio_scores = []
    else:
        # class_name = ["płaska intonacja", "duża intonacja", "normalna intonacja"]
        class_name = [0, 1, 2]
        audio_scores = []
        for i in range(0, len(gmms_model)):
            audio_scores.append(gmms_model[i].score(new_f0_test))
        # print(audio_scores)
        i_max = np.argmax(audio_scores)
        class_prediction = class_name[i_max]
        # print(f'predykcja = {class_name[i_max]}')
    return audio_scores, class_prediction


def recognize_intonation(filename_path, gender_gmm, gmms_f, gmms_m, fs=4000, fs2=16000):
    """
    Function recognize the speaker's gender and intonation
    :param filename_path: audio file or signal array
    :param gender_gmm: gmm model for gender recognition
    :param gmms_f: female's gmm model for intonation recognition
    :param gmms_m: male's gmm model for intonation recognition
    :param fs: sampling rate in Hertz for f0 detection
    :param fs2: sampling rate in Hertz for mfcc detection
    :return:
            class_prediction_gender: name of the predicted gender class
            class_prediction_intonation: name of the predicted intonation class
    """
    audio_scores_gender, class_prediction_gender = classify(
        filename_path, gender_gmm, fs2
    )
    if class_prediction_gender == "k":
        audio_scores, class_prediction_intonation = classify_intonation(
            filename_path, gmms_f, fs
        )
    else:  # class_prediction_gender == "m"
        audio_scores, class_prediction_intonation = classify_intonation(
            filename_path, gmms_m, fs
        )
    return (
        audio_scores_gender,
        class_prediction_gender,
        audio_scores,
        class_prediction_intonation,
    )
