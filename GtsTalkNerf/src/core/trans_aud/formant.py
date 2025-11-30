import librosa
import numpy as np
from scipy.signal import lfilter, get_window


class FormantFeatureExporter:
    def __init__(self, input_file, sr=None, window_len=512, n_fft=None, win_step=5 / 8, window="hamming", preemph=0.83):
        """
        :param sr: sample_rate,
        :param window_len: window length,
        :param n_fft: Length of the windowed signal after padding with zeros, default is window length
        :param win_step: Percentage of window shift
        :param window: window type used in librosa.stft
        """
        self.input_file = input_file
        self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)
        self.window_len = window_len
        self.n_fft = n_fft or self.window_len
        self.hop_length = round(self.window_len * win_step)
        self.window = window
        self.preemph = preemph

    def preemphasis(self):
        return librosa.effects.preemphasis(self.wave_data, coef=self.preemph)

    def energy(self):
        # The sum of the squares of the amplitudes of all sample points in each frame is used as the energy value
        mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.n_fft, hop_length=self.hop_length,
                                       win_length=self.window_len, window=self.window))
        pow_spec = np.square(mag_spec)
        energy = np.sum(pow_spec, axis=0)
        energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # Avoid having an energy value of 0
        return energy

    def get_magnitude_spectrogram(self):
        mag_spec = np.abs(librosa.stft(self.preemphasis(), n_fft=self.n_fft, hop_length=self.hop_length,
                                       win_length=self.window_len, window=self.window))
        return mag_spec

    def formant(self, ts_e=0.01, ts_f_d=200, ts_b_u=2000):
        """
        The LPC method estimates the center frequency and bandwidth of the first three formants of each frame.
        :param ts_e: Energy threshold, Only when the energy exceeds ts_e, a formant peak is considered to be possible.
        :param ts_f_d: Threshold at the center frequency of the formant：When the center frequency exceeds TS f_d and
                       is less than half of the sampling frequency, a formant peak is considered to be possible.
        :param ts_b_u: Threshold on the formant bandwidth：Below ts_b_u is considered likely to be formant.
        :return: F1/F2/F3、B1/B2/B3
        """
        _data = lfilter([1., 0.83], [1], self.wave_data)  # High-pass filter
        inc_frame = self.hop_length  # window shift
        n_frame = int(np.ceil(len(_data) / inc_frame))
        n_pad = n_frame * self.window_len - len(_data)
        _data = np.append(_data, np.zeros(n_pad))
        win = get_window(self.window, self.window_len, fftbins=False)
        formant_frq = []
        formant_bw = []
        e = self.energy()
        e = e / np.max(e)  # Gets the energy value for each frame and normalizes it
        for i in range(n_frame):
            f_i = _data[i * inc_frame:i * inc_frame + self.window_len]
            if np.all(f_i == 0):
                f_i[0] = np.finfo(np.float64).eps
            f_i_win = f_i * win
            a = librosa.lpc(f_i_win, order=8)
            rts = np.roots(a)
            rts = np.array([r for r in rts if np.imag(r) >= 0])
            rts = np.where(rts == 0, np.finfo(np.float64).eps, rts)
            ang = np.arctan2(np.imag(rts), np.real(rts))
            frq = ang * (self.sr / (2 * np.pi))
            indices = np.argsort(frq)
            frequencies = frq[indices]
            bandwidths = -0.25 * (self.sr / np.pi) * np.log(np.abs(rts[indices]))
            formant_f = []  # F1/F2/F3
            formant_b = []  # B1/B2/B3
            if e[i] > ts_e:
                # 采用共振峰频率大于ts_f_d小于self.sr/2赫兹，带宽小于ts_b_u赫兹的标准来确定共振峰
                for j in range(len(frequencies)):
                    if (ts_f_d < frequencies[j] <= self.sr / 2) and (bandwidths[j] <= ts_b_u):
                        formant_f.append(frequencies[j] / (self.sr / 2))
                        formant_b.append(bandwidths[j] / ts_b_u)
            if len(formant_f) < 3:
                formant_f += ([0] * (3 - len(formant_f)))
                formant_b += ([0] * (3 - len(formant_b)))
            else:
                formant_f = formant_f[0:3]
                formant_b = formant_b[0:3]
            formant_frq.append(np.array(formant_f))
            formant_bw.append(np.array(formant_b))
        formant_frq = np.array(formant_frq).T
        formant_bw = np.array(formant_bw).T
        # print(formant_frq.shape, np.nanmean(formant_frq, axis=1))
        # print(formant_bw.shape, np.nanmean(formant_bw, axis=1))
        return formant_frq, formant_bw

    def plot(self, show=True):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        """
        绘制语音波形曲线和log功率谱、共振峰叠加图
        :param show: 默认最后调用plt.show()，显示图形
        :return: None
        """
        plt.figure(figsize=(8, 6))
        # 以下绘制波形图
        plt.subplot(2, 1, 1)
        plt.title("Wave Form")
        plt.ylabel("Normalized Amplitude")
        plt.xticks([])
        audio_total_time = int(len(self.wave_data) / self.sr * 1000)  # 音频总时间ms
        plt.xlim(0, audio_total_time)
        plt.ylim(-1, 1)
        x = np.linspace(0, audio_total_time, len(self.wave_data))
        plt.plot(x, self.wave_data, c="b", lw=1)  # 语音波形曲线
        plt.axhline(y=0, c="pink", ls=":", lw=1)  # Y轴0线
        # 以下绘制灰度对数功率谱图
        plt.subplot(2, 1, 2)
        log_power_spec = librosa.amplitude_to_db(self.get_magnitude_spectrogram(), ref=np.max)
        librosa.display.specshow(log_power_spec[:, 1:], sr=self.sr, hop_length=self.hop_length,
                                 x_axis="s", y_axis="linear", cmap="gray_r")
        plt.title("Formants on Log-Power Spectrogram")
        plt.xlabel("Time/ms")
        plt.ylabel("Frequency/Hz")

        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, y: "%d" % (1000 * x)))
        # 以下在灰度对数功率谱图上叠加绘制共振峰点图
        formant_frq, __ = self.formant()  # 获取每帧共振峰中心频率
        color_p = {0: ".r", 1: ".y", 2: ".g"}  # 用不同颜色绘制F1-3点，对应红/黄/绿
        # X轴为对应的时间轴ms 从第0帧中间对应的时间开始，到总时长结束，间距为一帧时长
        x = np.linspace(0.5 * self.hop_length / self.sr,
                        audio_total_time / 1000, formant_frq.shape[1])
        for i in range(formant_frq.shape[0]):  # 依次绘制F1/F2/F3
            plt.plot(x, formant_frq[i, :], color_p[i], label="F" + str(i + 1))
        plt.legend(prop={'family': 'Times New Roman', 'size': 10}, loc="upper right",
                   framealpha=0.5, ncol=3, handletextpad=0.2, columnspacing=0.7)

        plt.tight_layout()
        if show:
            plt.show()


if __name__ == "__main__":
    import os
    wavfile = os.path.join(os.path.dirname(__file__), '../data/voca/dataset/wav/FaceTalk_170725_00137_TA_sentence01.wav')
    ffe = FormantFeatureExporter(wavfile)
    fmt_frq, fmt_bw = ffe.formant()
    print(fmt_frq.shape, fmt_bw.shape)
    ff = np.concatenate([fmt_frq, fmt_bw], 0)
    print(fmt_bw.T.min())
