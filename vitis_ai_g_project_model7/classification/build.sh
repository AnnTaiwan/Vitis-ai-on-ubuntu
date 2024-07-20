#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

GSTREAMER_FLAGS=`pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0`

CXX=${CXX:-g++}
$CXX -std=c++17 -O2 -I. -o demo_CNN_model7_normalize demo_CNN_model7_normalize.cpp -lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${OPENCV_FLAGS} -lopencv_core  -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

$CXX -std=c++17 -O2 -I. -o demo_final demo_final.cpp -lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${OPENCV_FLAGS} -lopencv_core  -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

$CXX -std=c++17 -O2 -I. -o demo_final_segment_audio demo_final_segment_audio.cpp -lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${OPENCV_FLAGS} -lopencv_core  -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

$CXX -std=c++17 -O2 -I. -o demo_CNN_model7_test demo_CNN_model7_test.cpp -lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${OPENCV_FLAGS} -lopencv_core  -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

$CXX -std=c++17 -O2 -I. -o demo_CNN_model7_gst demo_CNN_model7_gst.cpp -lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task ${OPENCV_FLAGS} $GSTREAMER_FLAGS -lopencv_core  -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
