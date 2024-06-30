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
#include <deque>
#include <string>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/audio/audio.h>

#include <gst/fft/gstfft.h>
#include <gst/fft/gstfftf32.h>

#define SAMPLE_RATE 16000
#define DURATION 5.0
#define AUDIO_LEN (SAMPLE_RATE * DURATION)
#define N_MELS 128
#define N_FFT 2048
#define SPEC_WIDTH 256
#define HOP_LEN (AUDIO_LEN / (SPEC_WIDTH - 1))
#define FMAX (SAMPLE_RATE / 2)
#define SPEC_SHAPE {SPEC_WIDTH, N_MELS}

using namespace std;
using namespace cv;

static GstElement *pipeline;
static GstElement *appsink;
static GstBus *bus;
static GstMessage *msg;

std::deque<float> audio_buffer; // Buffer to store incoming audio samples
/*
// Function to calculate mel-spectrogram
std::vector<float> get_mel_spectrogram(const std::vector<float>& audio, int sr) {
    guint fft_size = N_FFT;
    guint num_fft_bins = N_MELS;
    guint signal_size = audio.size();

    GstFFTF32Complex *fft_result = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);
    GstFFTF32Complex *fft_input = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);

    // Initialize FFT
    GstFFTF32 *fft = gst_fft_f32_new(fft_size, FALSE);

    // Prepare fft_input (complex to complex)
    for (guint i = 0; i < fft_size; ++i) {
        fft_input[i].r = (i < signal_size) ? (gfloat)audio[i] : 0.0f;
        fft_input[i].i = 0.0f; // Assuming no imaginary part for simplicity
    }

    // Perform FFT
    gst_fft_f32_fft(fft, (const gfloat *)fft_input, fft_result);

    // Compute mel-spectrogram (example calculation)
    std::vector<float> mel_spec(num_fft_bins * SPEC_WIDTH, 0.0f);
    for (guint i = 0; i < SPEC_WIDTH; ++i) {
        for (guint j = 0; j < num_fft_bins; ++j) {
            guint index = i * num_fft_bins + j;
            if (index < fft_size) {
                // Assign FFT magnitude to mel-spectrogram
                mel_spec[index] = std::abs(fft_result[index].r) + std::abs(fft_result[index].i);
            }
        }
    }

    // Clean up FFT resources
    gst_fft_f32_free(fft);
    g_free(fft_result);
    g_free(fft_input);

    return mel_spec;
}
*/
// Function to calculate Mel Spectrogram
vector<float> get_mel_spectrogram(const vector<float>& audio, int sr) {
    guint fft_size = N_FFT;
    guint num_fft_bins = N_MELS;
    guint signal_size = audio.size();

    GstFFTF32Complex *fft_result = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);
    GstFFTF32Complex *fft_input = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);

    // Initialize FFT
    GstFFTF32 *fft = gst_fft_f32_new(fft_size, FALSE);

    // Prepare fft_input (complex to complex)
    for (guint i = 0; i < fft_size; ++i) {
        fft_input[i].r = (i < signal_size) ? (gfloat)audio[i] : 0.0f;
        fft_input[i].i = 0.0f; // Assuming no imaginary part for simplicity
    }

    // Perform FFT
    gst_fft_f32_fft(fft, (const gfloat *)fft_input, fft_result);

    // Compute power spectrum
    vector<float> power_spectrum(fft_size);
    for (guint i = 0; i < fft_size; ++i) {
        power_spectrum[i] = pow(fft_result[i].r, 2) + pow(fft_result[i].i, 2);
    }

    // Clean up FFT resources
    gst_fft_f32_free(fft);
    g_free(fft_result);
    g_free(fft_input);

    // Calculate Mel Spectrogram
    vector<float> mel_spec(num_fft_bins * SPEC_WIDTH, 0.0f);
    float mel_step = FMAX / (num_fft_bins + 1);  // Mel filterbank step

    for (guint i = 0; i < SPEC_WIDTH; ++i) {
        float freq = FMAX * i / SPEC_WIDTH;  // Convert bin index to frequency

        for (guint j = 0; j < num_fft_bins; ++j) {
            float mel_center = mel_step * (j + 1);  // Mel center frequency of filterbank
            float lower_edge = (j == 0) ? 0 : mel_step * j;
            float upper_edge = (j == num_fft_bins - 1) ? FMAX : mel_step * (j + 2);

            if (freq >= lower_edge && freq <= upper_edge) {
                float weight = (freq - lower_edge) / (mel_center - lower_edge);
                if (weight > 1) weight = 1;
                mel_spec[i * num_fft_bins + j] = weight * power_spectrum[i];
            }
        }
    }

    return mel_spec;
}


// Function to process each audio segment
void process_audio_segment(const std::vector<float>& segment, int segment_index) {
    // Calculate mel-spectrogram
    auto mel_spec = get_mel_spectrogram(segment, SAMPLE_RATE);

    // Print Mel Spectrogram values
    std::cout << "Mel Spectrogram for segment " << segment_index << ":" << std::endl;
    for (int i = 0; i < N_MELS; ++i) {
        for (int j = 0; j < SPEC_WIDTH; ++j) {
            std::cout << mel_spec[j * N_MELS + i] << " ";
        }
        std::cout << std::endl;
    }

    // Reshape mel-spectrogram into image format (transpose and normalize)
    cv::Mat mel_image(N_MELS, SPEC_WIDTH, CV_32F);
    for (int i = 0; i < N_MELS; ++i) {
        for (int j = 0; j < SPEC_WIDTH; ++j) {
            mel_image.at<float>(i, j) = mel_spec[j * N_MELS + i]; // Transpose
        }
    }

    // Normalize to 0-255 range
    cv::normalize(mel_image, mel_image, 0, 255, cv::NORM_MINMAX);

    // Convert to 8-bit unsigned integer (CV_8UC1) for color mapping
    mel_image.convertTo(mel_image, CV_8UC1);

    // Apply color map (Viridis)
    cv::Mat mel_color;
    cv::applyColorMap(mel_image, mel_color, cv::COLORMAP_VIRIDIS);

    // Construct filename to save the image
    std::string filename = "mel_spectrogram_" + std::to_string(segment_index) + ".png";

    // Save the spectrogram image
    cv::imwrite(filename, mel_color);

    // Save the audio segment (optional, uncomment if needed)
    std::string audio_filename = "audio_segment_" + std::to_string(segment_index) + ".raw";
    std::ofstream audio_file(audio_filename, std::ios::binary);
    audio_file.write(reinterpret_cast<const char*>(&segment[0]), segment.size() * sizeof(float));
    audio_file.close();
}
// Callback for new audio sample
static GstFlowReturn new_sample(GstAppSink *appsink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    // Convert audio data to float vector and add to buffer
    float *audio_samples = (float *)map.data;
    int num_samples = map.size / sizeof(float);
    std::vector<float> segment(audio_samples, audio_samples + num_samples);

    // Store audio segment in buffer (optional, uncomment if needed)
    audio_buffer.insert(audio_buffer.end(), segment.begin(), segment.end());

    // Release resources
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    // Check if user entered 'p' to stop recording
    if (!segment.empty() && getchar() == 'p') {
        static int segment_index = 0;
        process_audio_segment(segment, segment_index);
        segment_index++;
    }

    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // Create GStreamer pipeline
    pipeline = gst_parse_launch(
        "alsasrc device=hw:0 ! "
        "audioconvert ! "
        "audioresample ! "
        "audio/x-raw,channels=1,rate=16000 ! "
        "queue ! "
        "audioconvert ! "
        "appsink name=sink sync=false",
        NULL);

    // Initialize appsink and connect new-sample signal
    appsink = GST_ELEMENT(gst_bin_get_by_name(GST_BIN(pipeline), "sink"));
    g_object_set(appsink, "emit-signals", TRUE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample), NULL);

    // Start the GStreamer pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Main loop to handle GStreamer messages and stop conditions
    bus = gst_element_get_bus(pipeline);
    msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                     static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    // Handle GStreamer messages
    if (msg != NULL) {
        GError *err = NULL;
        gchar *debug_info = NULL;

        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error(msg, &err, &debug_info);
                g_printerr("Error received from element %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
                g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
                g_clear_error(&err);
                g_free(debug_info);
                break;
            case GST_MESSAGE_EOS:
                g_print("End-Of-Stream reached.\n");
                break;
            default:
                g_printerr("Unexpected message received.\n");
                break;
        }
        gst_message_unref(msg);
    }

    // Cleanup GStreamer resources
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}

