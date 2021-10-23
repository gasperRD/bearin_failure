
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, lfilter, sosfilt, sosfreqz
from scipy.fftpack import fft

#Direct input
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams["figure.figsize"] = (6.5,4.5)
plt.rcParams.update(params)


########sinteticni signal############
fs = 6e3
fres = 800
frot = 60
A=5
ff = 10
a = 0.02
T = 1/ff
K = 50
t = np.arange(0, 1, 1 / fs)
N = t/fs
dt = 1 / fs

x1 = np.array(N)
x2 = np.array(N)
x3 = np.array(N)

for k in range(K-1):
    for i in range(len(t)):
        if t[i] - k*T >=0:
            x1[i] = A * np.exp(-a*2*np.pi*fres*(t[i] - k * T))
            x2[i] = np.sin(2*np.pi*fres*(t[i] - k*T))
            x3[i] = x1[i]*x2[i]

sum = np.random.normal(N) #šum
x4 = 2*np.sin(2*np.pi*t*frot)
vib = x3+x4+sum
print(len(N))

plt.subplot(2, 1, 1)
plt.subplots_adjust(hspace=0.6)
plt.plot(t, vib,'b')
plt.ylabel('$x(t)$')
plt.xlabel('Čas t [s]')
plt.grid( linestyle='--', linewidth=0.5)
plt.ylim(-10,10)
plt.xlim(0,1)
plt.title('a)')


###FFT##
N = len(N)
fft= np.fft.rfft(vib)*2/N
fr = np.fft.rfftfreq(N,d=dt)


plt.subplot(2, 1, 2)

plt.plot(fr,np.abs(fft),'b')

plt.ylabel(r'$|x(t)|$')
plt.xlabel('Frekvenca $f$ [Hz]')
plt.xlim(0,1000)
plt.title('b)')
plt.grid( linestyle='--', linewidth=0.5)

plt.savefig('sint_x5.pdf',dpi=1000)
#filter definicja butterworth

lowcut= 700
highcut= 900
####################filter definition za etaloneee#######################


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal = sosfilt(sos, data)
    return filtered_signal

filtered_signal= butter_bandpass_filter(vib, lowcut, highcut, fs, order=4)

envel = hilbert(filtered_signal)
envelope = np.abs(envel)

fig1, ax = plt.subplots()
plt.subplot(2, 1, 1)
plt.subplots_adjust(hspace=0.6)
plt.title('a)')
plt.plot(t, filtered_signal,'b',linewidth=0.5,label='Filtriran signal')
plt.plot(t,envelope,'r',linewidth=1,label='Ovojnica signala')
plt.ylabel('$x(t)$')
plt.xlabel('Čas $t$ [s]')
plt.grid( linestyle='--', linewidth=0.5)
plt.ylim(-10,10)
plt.xlim(0,0.2)
plt.legend(loc=4,frameon=True)


fft_envelope = np.fft.rfft(envelope)*2/N

plt.subplot(2, 1, 2)
plt.plot(fr, np.abs(fft_envelope),'r')
plt.title('b)')
plt.ylabel(r'$|x(t)|$')
plt.xlabel('Frekvenca $f$ [Hz]')
plt.xlim(0,100)
plt.xticks(np.arange(0, 100, step=10))
plt.grid( linestyle='--', linewidth=0.5)
plt.savefig('sint_x5_envelope.pdf',dpi=1000)
plt.show()
