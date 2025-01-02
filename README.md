# 📡 **SSB Modulation and FDM Project**  

This project implements **Single Sideband (SSB) Modulation** within a **Frequency Division Multiplexing (FDM)** system to modulate and transmit three speech signals. The project includes signal recording, filtering, modulation, demodulation, and spectrum analysis, following the principles of communication systems.  

---

## 📜 **Overview**  

The project demonstrates SSB modulation using MATLAB for signal recording and Python for the signal processing pipeline. It explores signal filtering, modulation, and demodulation, ensuring compliance with the sampling theorem.  

### **Objectives**  

1. Record and process three audio signals for modulation.  
2. Apply low-pass filtering to limit the maximum frequency without compromising audio quality.  
3. Perform SSB modulation for a frequency-division multiplexed system.  
4. Analyze the spectrum of signals at various stages: original, filtered, modulated, and demodulated.  
5. Recover original signals through SSB demodulation and compare their quality.  

---

## 🛠️ **Features**  

1. **Audio Recording and Processing**  
   - Records audio signals with a configurable sampling rate and duration.  
   - Saves signals as WAV files.  

2. **Low-Pass Filtering**  
   - Designs and applies a Butterworth LPF.  
   - Visualizes and evaluates filtered signals for audio quality.  

3. **SSB Modulation and Demodulation**  
   - Modulates signals using custom SSB modulation without built-in functions.  
   - Demodulates FDM signals to recover the original speech signals.  

4. **Spectrum Analysis**  
   - Plots the magnitude spectrum at various stages: original, filtered, modulated, and demodulated.  

---

## 📁 **Project Structure**  

- **`SSB_Modulation_and_FDM_System.py`**: Contains the Python implementation for SSB modulation, demodulation, and FDM.  
- **`input1.wav, input2.wav, input3.wav`**: Recorded audio files for processing.  
- **`output1.wav, output2.wav, output3.wav`**: Demodulated signals after recovery.  
- **`content/`**: Directory for input/output files.  
- **`project_report.pdf`**: Comprehensive project report with explanations, results, and figures.  

---

## 🚀 **Getting Started**  

### **1. Clone the Repository**  
```bash  
git clone https://github.com/omarhashem80/SSB_Modulation_and_FDM_System.git
cd SSB_Modulation_and_FDM_System
```  

### **2. Install Dependencies**  
Use `pip` to install the required Python libraries:  
```bash  
pip install -r requirements.txt  
```  

### **3. Run the Project**  
Execute the `process()` method to record, filter, modulate, and demodulate signals.  
```python  
from SSB_FDM import SSB_FDM  

processor = SSB_FDM()  
processor.process(['input1.wav', 'input2.wav', 'input3.wav'], ['output1.wav', 'output2.wav', 'output3.wav'])  
```  

---

## 📊 **Outputs**  

1. **Audio Files**  
   - Original signals: `input1.wav`, `input2.wav`, `input3.wav`.  
   - Demodulated signals: `output1.wav`, `output2.wav`, `output3.wav`.  

2. **Plots**  
   - Magnitude spectrum of signals (original, filtered, modulated, demodulated).  
   - Combined FDM signal spectrum.  

3. **Report**  
   - Detailed analysis, justifications, and results in `project_report.pdf`.  

---

## 🌟 **Features in Action**  

1. **SSB Modulation**  
   - Each input signal is modulated using a unique carrier frequency.  
   - Ensures no spectral overlap in the FDM system.  

2. **SSB Demodulation**  
   - Recovers original signals with minimal distortion.  
   - Normalizes output signals for consistency.  

3. **Spectral Analysis**  
   - Visualizations of signal transformations across stages.  

---

## 🤝 **Contributors**  

[Omar Hashem](https://github.com/omarhashem80)
[Ahmed Mostafa](https://github.com/New-pro125)

---

## ⚙️ **Customization**  

- Change carrier frequencies or LPF cutoff values in the `SSB_FDM` initialization.  
- Adjust the sampling rate or recording duration to suit your requirements.  

---
