# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 03:42:21 2020

@author: APR
"""
import time
import sys
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from scipy import signal

format = pyaudio.paFloat32

        # Configuramos la ventana Qt que contiene los graficos
app = QtGui.QApplication([])
w = QtGui.QWidget()
layout = QtGui.QGridLayout()
w.setLayout(layout)
w.setWindowTitle("Procesado de señal de audio en tiempo real")
w.setFixedSize(1000,500)
p = w.palette()
p.setColor(w.backgroundRole(), pg.mkColor('b'))
w.setPalette(p)
pg.setConfigOptions(antialias=True)

class Audio():
    def __init__(self,CHANNELS, CHUNK, FRAME_RATE, layout, w):
        self.channels = CHANNELS
        self.chunk = CHUNK
        self.fs = FRAME_RATE
        self.lay = layout
        self.w = w
        self.plt1 = pg.PlotWidget()
        self.plt2 = pg.PlotWidget()
        self.lay.addWidget(self.plt1, 1, 0)
        self.lay.addWidget(self.plt2, 1, 1)
    
    def initPlot(self):
        self.plt1.setYRange(-0.5, 0.5)
        self.plt1.getPlotItem().setTitle(title="Representación temporal")
        self.plt1.getAxis('bottom').setLabel('Tiempo',units='ms')
        self.plt1.getAxis('bottom').enableAutoSIPrefix(enable=True)
        self.plt1.getAxis('left').setLabel('Amplitud')
        self.plt2.getPlotItem().setTitle(title="Representación frecuencial (FFT)")
        self.plt2.getAxis('bottom').setLabel('Frecuencia (kHz)')
        self.plt2.setYRange(0, 60)
        self.plt2.getAxis('bottom').enableAutoSIPrefix(enable=True)
        self.plt2.getAxis('left').setLabel('Nivel')
        self.w.show()
        
    def update(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,channels=self.channels,rate=self.fs, input=True, frames_per_buffer=self.chunk)
        data_bytes = stream.read(self.chunk)

        if not data_bytes:
            print("no data")
        else:
            #Señal Temporal (ms)
            data = self.a_ndarray(data_bytes,np.float32)
            t = (np.linspace(0,self.chunk/self.fs,self.chunk))*1000 
            self.plt1.plot(t[50:], data[50:], pen=(255,0,0), clear=True)
            
            #Frecuencia (kHz)
            A = np.fft.fftshift(np.fft.fft(data))
            freq = np.fft.fftshift(np.fft.fftfreq(len(data)))*self.fs/1000
            i = int(len(freq)/2) #Contador a partir de las frecuencias > 0
            #f = int(3*len(freq)/4)
            self.plt2.plot(freq[i:],np.abs(A)[i:], pen=(255,0,0), clear=True)
        stream.stop_stream()
        stream.close()
        p.terminate()


    # Definimos los tipos de señales a muestrear
    def sin(self, f, time = 5, volumen = 1): 
        return volumen*np.sin(2*np.pi*np.arange(self.fs*time)*f/self.fs)

    def sin_plus(self, f, time = 5, volumen = 1):
        a = self.sin(f,time,volumen)
        for i in range(len(a)):
            if a[i] < 0:
                a[i]=0
        return a

    def sin_abs(self, f, time = 5, volumen = 1):
        return np.abs(self.sin(f/2,time,volumen))

    def triangular(self, f, time = 5, volumen = 1):
        return volumen*signal.sawtooth(2*np.pi*f*(np.arange(self.fs*time)/self.fs),0.5)

    def triangular_plus(self, f, time = 5, volumen = 1):
        a = self.triangular(f,time,volumen)
        for i in range(len(a)):
            if a[i] < 0:
                a[i]=0
        return a  

    def triangular_abs(self, f, time = 5, volumen = 1):
        return np.abs(self.triangular(f/2,time,volumen))    

    def sierra(self, f, time = 5, volumen = 1):
         return volumen*signal.sawtooth(2*np.pi*f*(np.arange(self.fs*time)/self.fs))

    def square(self, f, time = 5, volumen = 1):
        return volumen*signal.square(2*np.pi*f*(np.arange(self.fs*time)/self.fs))

    def square_plus(self, f, time = 5, volumen = 1):
        a = self.square(f,time,volumen)
        for i in range(len(a)):
            if a[i] < 0:
                a[i]=0
        return a

    def play(self, data, format_pyaudio = pyaudio.paFloat32, format_numpy = np.float32): #Reproduce el sonido de la señal de audio
        if type(data)==np.ndarray:
            data = self.a_bytes(data,format_numpy)
        p = pyaudio.PyAudio()
        stream = p.open(format=format_pyaudio, channels=self.channels, rate=self.fs, output=True)
        stream.write(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def plot(self, data, format = np.float32, save = False, show = True, t_init = 0, t_end = 1): #Grafica la señal de audio
        if type(data)==bytes:
            data = self.a_ndarray(data,format)
        t = np.linspace(0, len(data)/self.fs, len(data))
        plt.plot(t,data)
        plt.xlim(t_init, t_end)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        if save:
            plt.savefig(str(sum(data))+'.jpg')
        if show:
            plt.show()

    def plotfft(self, data, format = np.float32, save = False, show = True): #Grafica el espectro de frecuencia de la señal de audio
        if type(data)==bytes:
            data = self.a_ndarray(data,format)
        A = np.fft.fftshift(np.fft.fft(data))
        freq = np.fft.fftshift(np.fft.fftfreq(len(data)))*self.fs #/1000
        plt.semilogx(freq,np.abs(A))
        plt.xlim(10,20000)
        plt.ylim(0,3e7)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Nivel')
        if save:
            plt.savefig(str(sum(data)+1)+'.jpg')
        if show:
            plt.show()

    def plot_together(self, data, format = np.float32, t_init=0, t_end=1, save = False):
        plt.subplot(2,1,1)
        self.plot(data, format, show = False, t_init = t_init, t_end = t_end)
        plt.subplot(2,1,2)
        self.plotfft(data, format, show = False)
        if save:
            plt.savefig(str(sum(data)+1)+'.jpg')
        plt.show()

    def grabar(self, time, name_file = None):
        audio=pyaudio.PyAudio()
        stream=audio.open(format=pyaudio.paInt16,channels=self.channels, rate=self.fs, input=True, frames_per_buffer=self.chunk)
        print("grabando...")
        frames=[]
        for i in range(0, int(self.fs/self.chunk*time)):
            data=stream.read(self.chunk)
            frames.append(data)
        print("grabación terminada")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        if name_file:
            wf.write(name_file, self.fs, self.a_ndarray(b''.join(frames),np.int16))
            return self.a_ndarray(b''.join(frames),np.int16)
        else:
            return self.a_ndarray(b''.join(frames),np.int16)

    def normal(self,data):
        return data/max(np.abs(data))

    def a_bytes(self,data,format):
        if format == np.float32:
            return data.astype(np.float32).tobytes()
        elif format == np.int16:
            return data.tobytes()

    def a_ndarray(self,data,format):
        return np.frombuffer(data,dtype=format)  

def init_osciloscopio(audio):
    audio.initPlot()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(audio.update)
    timer.start(0)

    if __name__ == '__main__':
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
     


fs = 44100   # Frecuencia de muestreo de la señal de audio
channels = 1 # Canales de audio
chunk = 1024 # Longitud de las mustras de audio a procesar 

a = Audio(channels,chunk,fs,layout,w)
        
        # Inicializamos el osciloscopio, cuya señal de entrada proviene del microfono
init_osciloscopio(a)

  
        # Graficamos las distintas señales y su espectro en frecuencia
f = 100
data = [a.sin(f,0.5), a.sin_plus(f,0.5), a.sin_abs(f,0.5), a.triangular(f,0.5), a.triangular_plus(f,0.5), a.triangular_abs(f,0.5), a.sierra(f,0.5), a.square(f,0.5), a.square_plus(f,0.5)]
for d in data:
    a.play(d)
    a.plot_together(d, t_end=5/f, save = True) # Para save=True guarda las imagenes




        # Creamos un efecto de batido con senos de diferente frecuencia, reproducimos y graficamos
data = a.sin(400,1) + a.sin(300,1) + a.sin(440,1,volumen=0.5) + a.sin(700,1,volumen=0.3)
data = a.normal(data)
a.play(data)
a.plot_together(data, t_end = 0.2, save = False)



        # Creamos un popurrí con funciones seno de diferentes frecuencias y amplitudes y lo guardamos en un archivo '.wav'
data = a.sin(f=100,time=0.5,volumen=np.random.rand())
for i in range(200,1000,100):
    data = np.append(data, a.sin(f=i,time=0.5,volumen=np.random.rand()))
for i in range(1000,100,-100):
    data = np.append(data, a.sin(f=i,time=0.5,volumen=np.random.rand()))
a.play(data)
a.plot_together(data,t_end=9, save = True)
wf.write('popurrit.wav',44100,data)



        # Grabamos 10 segundos del microfono lo guardamos en 'voz_humana.wav' reproducimos y graficamos
data = a.grabar(time = 10, name_file = 'song.wav')
a.play(data,pyaudio.paInt16, np.int16)
a.plot_together(data,t_end=10, save = False)




 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        