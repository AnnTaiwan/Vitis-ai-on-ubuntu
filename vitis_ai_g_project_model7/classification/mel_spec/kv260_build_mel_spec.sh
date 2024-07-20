CXX=${CXX:-g++}
GSTREAMER_FLAGS=$(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-fft-1.0)

$CXX -std=c++17 -O2 -o cal_mel_spec_ver1 cal_mel_spec_ver1.cpp  ${GSTREAMER_FLAGS}
