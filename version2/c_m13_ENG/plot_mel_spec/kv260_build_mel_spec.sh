CXX=${CXX:-g++}
GSTREAMER_FLAGS=$(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-fft-1.0)
CAIRO_FLAGS=$(pkg-config --cflags --libs cairo)


$CXX -std=c++17 -O2 -o cal_mel_spec_ver2 cal_mel_spec_ver2.cpp ${GSTREAMER_FLAGS}

$CXX -std=c++17 -O2 -o cal_mel_spec_ver4 cal_mel_spec_ver4.cpp ${GSTREAMER_FLAGS}

$CXX -std=c++17 -O2 -o plot_mel_spec_from_txt plot_mel_spec_from_txt.cpp ${CAIRO_FLAGS}

$CXX -std=c++17 -O2 -o plot_mel_spec_from_txt_ver2 plot_mel_spec_from_txt_ver2.cpp ${CAIRO_FLAGS}

$CXX -std=c++17 -O2 -o plot_mel_spec_from_txt_scale plot_mel_spec_from_txt_scale.cpp ${CAIRO_FLAGS}

$CXX -std=c++17 -O2 -o plot_mel_spec_from_txt_scale_ver2 plot_mel_spec_from_txt_scale_ver2.cpp ${CAIRO_FLAGS}
