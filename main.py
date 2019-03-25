import librosa as lr

if __name__ == '__main__':
    audio, sr = lr.load('1.mp3')
    print(audio)
