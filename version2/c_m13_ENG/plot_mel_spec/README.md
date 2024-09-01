## This folder is used to calculate the mel_spec and plot the mel_spec in order to imitate the python version.
1. The latest version of calculate the mel_spec: `cal_mel_spec_ver4.cpp` from Yang (My memember in graduate project)
2. The latest version of plot the mel_spec: `plot_mel_spec_from_txt_ver2.cpp` from Yang (My memember in graduate project)

### Note
The above two codes are used together due to the output format of mel_spectrogram.txt, whose size is (rows, columns)(128, 216).
The other is my verison. There exists some problems in my version, and my output format's size of mel_spectrogram.txt is (rows, columns)(216, 128).
* In my version, `cal_mel_spec_ver2.cpp` and plot_mel_spec_from_txt.cpp` are used together.


* `plot_mel_spec_from_txt_scale.cpp` can draw the target image with setting height and width.
