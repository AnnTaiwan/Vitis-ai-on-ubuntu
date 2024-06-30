/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/classification.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vector>

#include <string>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/audio/audio.h>

using namespace std;
using namespace cv;

// Callback function to handle new spectrum data
static GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) {
        return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    // Print spectrum data
    float *spectrum_data = (float *)map.data;
    guint num_spectrum_values = map.size / sizeof(float);

    std::cout << "Spectrum data: ";
    for (guint i = 0; i < num_spectrum_values; ++i) {
        std::cout << spectrum_data[i] << " ";
    }
    std::cout << std::endl;

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // Create the GStreamer pipeline
    std::string pipeline_desc =
        "alsasrc device=hw:0 ! "
        "audioconvert ! "
        "audioresample ! "
        "audio/x-raw,channels=1,rate=16000 ! "
        "queue ! "
        "audioconvert ! "
        "spectrum interval=314000000 bands=128 threshold=-100 ! " // HOP_LEN 為 314 毫秒, bands 為 128
        "appsink name=sink sync=false";

    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
    if (error) {
        std::cerr << "Failed to create pipeline: " << error->message << std::endl;
        g_error_free(error);
        return -1;
    }

    // Get the appsink element from the pipeline
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    gst_app_sink_set_emit_signals((GstAppSink *)appsink, true);
    gst_app_sink_set_drop((GstAppSink *)appsink, true);
    gst_app_sink_set_max_buffers((GstAppSink *)appsink, 1);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);

    // Start the pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Wait for EOS or error
    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg;
    bool terminate = false;
    while (!terminate) {
        msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

        if (msg != nullptr) {
            switch (GST_MESSAGE_TYPE(msg)) {
                case GST_MESSAGE_ERROR: {
                    GError *err;
                    gchar *debug_info;
                    gst_message_parse_error(msg, &err, &debug_info);
                    std::cerr << "Error received from element " << GST_OBJECT_NAME(msg->src) << ": " << err->message << std::endl;
                    std::cerr << "Debugging information: " << (debug_info ? debug_info : "none") << std::endl;
                    g_clear_error(&err);
                    g_free(debug_info);
                    terminate = true;
                    break;
                }
                case GST_MESSAGE_EOS:
                    std::cout << "End-Of-Stream reached." << std::endl;
                    terminate = true;
                    break;
                default:
                    // Should not reach here
                    std::cerr << "Unexpected message received." << std::endl;
                    break;
            }
            gst_message_unref(msg);
        }
    }

    // Clean up
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    gst_deinit();

    return 0;
}

