import os
import random
import heapq
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import keras


def get_top(arr, num):
    # Blank parts.
    if max(arr) == 0:
        return [0] * num
    else:
        return heapq.nlargest(num, range(len(arr)), key=arr.__getitem__)
# get_top


def slice_by_beat(signal, sampling_rate, hop_length):
    tempo, beat = lr.beat.beat_track(y=signal, sr=sampling_rate, hop_length=hop_length)
    slices = []
    for i in range(len(beat) - 1):
        slices.append(signal[beat[i]*hop_length: beat[i+1]*hop_length])
    slices = np.array(slices)
    return slices
# slice_by_beat


def is_consonant(height1, height2):
    diff = (height1 - height2) % 12
    if (diff == 0) or (diff == 3) or (diff == 4) or (diff == 5) or (diff == 7) or (diff == 8) or (diff == 9):
        return True
    else:
        return False
# is_consonant


def chroma_by_beat(signal, sampling_rate, hop_length):
    slices = slice_by_beat(signal, sampling_rate, hop_length=hop_length)
    chromas = []
    for cur_slice in slices:
        cur_chroma = lr.feature.chroma_stft(y=cur_slice, sr=sampling_rate)
        chromas.append(cur_chroma)
    return chromas
# chroma_by_beat


def consonance_rate(signal, sampling_rate):
    consonance_count = 0
    dissonance_count = 0
    chroma = lr.feature.chroma_stft(y=signal, sr=sampling_rate)
    for i in range(len(chroma[0])):
        cur_slice = chroma[:, i]
        top_pair = get_top(cur_slice, 2)
        if is_consonant(top_pair[0], top_pair[1]):
            consonance_count += 1
        else:
            dissonance_count += 1
    return consonance_count / (consonance_count + dissonance_count)
# consonance_rate


def diff_consonance_rate(signal, sampling_rate):
    consonance_count = 0
    dissonance_count = 0
    chroma = lr.feature.chroma_stft(y=signal, sr=sampling_rate)
    for i in range(len(chroma[0])):
        cur_slice = chroma[:, i]
        cur_slice_d = []
        for j in range(11):
            cur_slice_d.append(2 * cur_slice[j] - cur_slice[j-1] - cur_slice[j+1])
        cur_slice_d.append(2 * cur_slice[11] - cur_slice[0] - cur_slice[10])
        top_pair = get_top(cur_slice_d, 2)
        if is_consonant(top_pair[0], top_pair[1]):
            consonance_count += 1
        else:
            dissonance_count += 1
    return consonance_count / (consonance_count + dissonance_count)
# diff_consonance_rate


def conv_consonance_rate(signal, sampling_rate):
    consonance_count = 0
    dissonance_count = 0
    chroma = lr.feature.chroma_stft(y=signal, sr=sampling_rate)
    for i in range(len(chroma[0])):
        cur_slice = chroma[:, i]
        cur_slice_c = []
        for j in range(12):
            cur_sum = 0
            for k in range(12):
                cur_sum += cur_slice[j][(j+k) % 12]
            cur_slice_c.append(cur_sum)
    return
# conv_consonance_rate


def two_beat_detect(signal, sampling_rate, hop_length):
    slices = slice_by_beat(signal, sampling_rate, hop_length=hop_length)
    var = []
    for item in slices:
        var.append(np.var(item))
    counter = 0
    for i in range(len(var) - 4):
        if (var[i] < var[i+1]) and (var[i+1] > var[i+2]) and (var[i+2] < var[i+3]) and (var[i+3] > var[i+4]):
            counter += 1
    return 2 * counter / len(var)
# two_beat_detect


def three_beat_detect(signal, sampling_rate, hop_length):
    slices = slice_by_beat(signal, sampling_rate, hop_length=hop_length)
    var = []
    for item in slices:
        var.append(np.var(item))
    counter = 0
    for i in range(len(var) - 5):
        if (var[i] < var[i+1]) and (var[i+1] > var[i+2]) and (var[i+3] < var[i+4]) and (var[i+4] > var[i+5]):
            counter += 1
    return 3 * counter / len(var)
# three_beat_detect


def four_beat_detect(signal, sampling_rate, hop_length):
    slices = slice_by_beat(signal, sampling_rate, hop_length=hop_length)
    var = []
    for item in slices:
        var.append(np.var(item))
    counter = 0
    for i in range(len(var) - 4):
        if (var[i] > var[i+1]) and (var[i+1] < var[i+2]) and (var[i+2] > var[i+3]) and (var[i+3] < var[i+4]) \
                and (var[i] > var[i+2]) and (var[i+2] < var[i+4]):
            counter += 1
    return 4 * counter / len(var)
# four_beat_detect


def pattern_beat_detect(signal, sampling_rate, hop_length, beat):
    slices = slice_by_beat(signal, sampling_rate, hop_length)
    var = []
    for item in slices:
        var.append(np.var(item))
    counter = 0
    for i in range(len(var) - 2 * beat):
        cur = []    # current pattern
        nxt = []    # next pattern
        for j in range(beat):
            cur.append(var[i+j] < var[i+j+1])
            nxt.append(var[i+beat+j] < var[i+beat+j+1])
        if cur == nxt:
            counter += 1
    return counter / len(var)
# pattern_beat_detect


if __name__ == '__main__':
    
    data = []
    labels = []
    testdata = []

    for i in range(10):
        filename = str(i) + '.mp3'
        y, sr = lr.load(filename)

        for j in range(10):
            slr = np.random.randint(len(y) - 1323000)
            sl = y[slr:slr + 1323000]
            sl = np.reshape(sl, (1323, 1000))
            data.append(sl)
            labels.append(0)

        slr = np.random.randint(len(y) - 1323000)
        sl = y[slr:slr + 1323000]
        sl = np.reshape(sl, (1323, 1000))
        testdata.append(sl)

        print('File ' + filename + ' read successfully.')

    for i in range(10):
        filename = str(i + 10) + '.mp3'
        y, sr = lr.load(filename)

        for j in range(10):
            slr = np.random.randint(len(y) - 1323000)
            sl = y[slr:slr + 1323000]
            sl = np.reshape(sl, (1323, 1000))
            data.append(sl)
            labels.append(1)

        slr = np.random.randint(len(y) - 1323000)
        sl = y[slr:slr + 1323000]
        sl = np.reshape(sl, (1323, 1000))
        testdata.append(sl)

        print('File ' + filename + ' read successfully.')

    data = np.array(data)
    testdata = np.array(testdata)

    model0 = keras.Sequential()
    model0.add(keras.layers.LSTM(units=128, input_shape=(1323, 1000), return_sequences=True))
    model0.add(keras.layers.Dropout(rate=0.2))
    model0.add(keras.layers.LSTM(units=32))
    model0.add(keras.layers.Dropout(rate=0.2))
    model0.add(keras.layers.Dense(units=1))
    model0.add(keras.layers.Activation(activation='sigmoid'))
    model0.compile(optimizer='adam', loss='mean_squared_error')
    model0.summary()
    model0.fit(x=data, y=labels, epochs=100, batch_size=100)

    testlabels = model0.predict(x=testdata)

    data1 = []
    testdata1 = []

    for i in range(len(data)):
        new_item = []
        new_item.append(testlabels[i])
        new_item.append(consonance_rate(data[i], 22050))
        data1.append(new_item)

    model1 = keras.Sequential()
    model1.add(keras.layers.Dense(units=32, input_shape=(2, )))
    model1.add(keras.layers.Activation(activation='sigmoid'))
    model1.add(keras.layers.Dense(units=1))
    model1.add(keras.layers.Activation(activation='sigmoid'))
    model1.compile(optimizer='adam', loss='mean_squared_error')
    model1.summary()
    model1.fit(x=data1, y=labels, epochs=100, batch_size=100)

    selftest1 = model1.predict(x=data)
    testlabels1 = model1.predict(x=testdata)

    print(selftest1)
    print(testlabels1)
