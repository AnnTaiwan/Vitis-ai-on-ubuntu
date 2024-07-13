#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <fstream> // For file operations

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/audio/audio.h>
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

static GstElement *pipeline;
static GstElement *appsink;
static std::deque<float> audio_buffer; // Buffer to store incoming audio samples

// Function to apply Hamming window
std::vector<float> apply_hamming_window(const std::vector<float>& signal) {
    std::vector<float> windowed_signal(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        windowed_signal[i] = signal[i] * (0.54 - 0.46 * cos(2 * M_PI * i / (signal.size() - 1)));
    }
    return windowed_signal;
}

// Function to generate a Mel filter bank
std::vector<std::vector<float>> generate_mel_filter_bank(int sample_rate, int n_fft, int n_mels, int fmax) {
    std::vector<std::vector<float>> mel_filter_bank(n_mels, std::vector<float>(n_fft / 2 + 1, 0.0f));
    
    // Step 1: Define Mel scale transformation functions
    auto hz_to_mel = [](float hz) { return 2595.0 * log10(1.0 + hz / 700.0); };
    auto mel_to_hz = [](float mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); };
    
    // Step 2: Calculate Mel frequency points
    float mel_min = hz_to_mel(0);
    float mel_max = hz_to_mel(fmax);
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }
    
    // Step 3: Convert Mel points back to Hz
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Step 4: Map Hz points to FFT bin numbers
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bin_points[i] = static_cast<int>(floor((n_fft + 1) * hz_points[i] / sample_rate));
    }
    
    // Step 5: Create Mel filter bank
    for (int i = 1; i <= n_mels; ++i) {
        int start = bin_points[i - 1];
        int center = bin_points[i];
        int end = bin_points[i + 1];
        
        for (int j = start; j < center; ++j) {
            mel_filter_bank[i - 1][j] = (j - start) / static_cast<float>(center - start);
        }
        for (int j = center; j < end; ++j) {
            mel_filter_bank[i - 1][j] = (end - j) / static_cast<float>(end - center);
        }
    }
    
    return mel_filter_bank;
}
// Function to convert power spectrogram to dB scale
std::vector<float> power_to_db(const std::vector<float>& power_spec, float ref_value=1.0, float amin=1e-10, float top_db=80.0) {
    std::vector<float> db_spec(power_spec.size());
    float log_spec_ref = 10.0 * log10(ref_value);

    for (size_t i = 0; i < power_spec.size(); ++i) {
        float val = std::max(power_spec[i], amin);
        db_spec[i] = 10.0 * log10(val) - log_spec_ref;
        if (top_db > 0.0) {
            db_spec[i] = std::max(db_spec[i], -top_db);
        }
    }

    return db_spec;
}
// Function to calculate Mel spectrogram
std::vector<float> get_mel_spectrogram(const std::vector<float>& audio, int sr) {
    guint fft_size = N_FFT;
    guint num_fft_bins = N_MELS;

    // Initialize FFT
    GstFFTF32 *fft = gst_fft_f32_new(fft_size, FALSE);

    // Prepare Mel spectrogram container
    std::vector<float> mel_spec(num_fft_bins * SPEC_WIDTH, 0.0f);
    std::vector<std::vector<float>> mel_filter_bank = generate_mel_filter_bank(sr, fft_size, num_fft_bins, FMAX);

    for (size_t i = 0; i < SPEC_WIDTH; ++i) {
        size_t start = i * HOP_LEN;
        size_t end = std::min(start + fft_size, audio.size());
        std::vector<float> segment(audio.begin() + start, audio.begin() + end);

        // Apply Hamming window
        segment = apply_hamming_window(segment);

        GstFFTF32Complex *fft_result = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);
        GstFFTF32Complex *fft_input = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);

        // Prepare fft_input (complex to complex)
        for (guint j = 0; j < fft_size; ++j) {
            fft_input[j].r = (j < segment.size()) ? segment[j] : 0.0f;
            fft_input[j].i = 0.0f; // Assuming no imaginary part for simplicity
        }

        // Perform FFT
        gst_fft_f32_fft(fft, (const gfloat *)fft_input, fft_result);

        // Compute power spectrum
        std::vector<float> power_spectrum(fft_size / 2 + 1, 0.0f);
        for (guint j = 0; j < fft_size / 2 + 1; ++j) {
            power_spectrum[j] = pow(fft_result[j].r, 2) + pow(fft_result[j].i, 2);
        }

        // Apply Mel filter bank
        for (guint m = 0; m < num_fft_bins; ++m) {
            float mel_value = 0.0f;
            for (guint k = 0; k < fft_size / 2 + 1; ++k) {
                mel_value += mel_filter_bank[m][k] * power_spectrum[k];
            }
            mel_spec[i * num_fft_bins + m] = mel_value + 1e-6; // Logarithmic scaling
        }

        // Clean up FFT resources
        g_free(fft_result);
        g_free(fft_input);
    }

    gst_fft_f32_free(fft);

    // Convert power spectrogram to dB scale
    mel_spec = power_to_db(mel_spec);

    return mel_spec;
}



// Function to write Mel spectrogram to a .txt file
void write_mel_spectrogram_to_txt(const std::vector<float>& mel_spec, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < mel_spec.size(); ++i) {
            file << mel_spec[i] << " ";
            if ((i + 1) % SPEC_WIDTH == 0) {
                file << std::endl;
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
    }
}

// Callback for new audio sample
static GstFlowReturn new_sample(GstAppSink *appsink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    // Convert audio data to float vector
    float *audio_samples = reinterpret_cast<float *>(map.data);
    int num_samples = map.size / sizeof(float);
    std::vector<float> audio_vec(audio_samples, audio_samples + num_samples);

    // Store audio samples in buffer
    audio_buffer.insert(audio_buffer.end(), audio_vec.begin(), audio_vec.end());

    // If enough samples collected, process Mel spectrogram
    if (audio_buffer.size() >= AUDIO_LEN) {
        std::vector<float> audio_segment(audio_buffer.begin(), audio_buffer.begin() + AUDIO_LEN);
        std::vector<float> mel_spec = get_mel_spectrogram(audio_segment, SAMPLE_RATE);

        // Write Mel spectrogram to .txt file (only for the first time)
        static bool first_time = true;
        if (first_time) {
            write_mel_spectrogram_to_txt(mel_spec, "mel_spectrogram.txt");
            first_time = false;
        }

        // Example: Output the first few values of mel_spec
        std::cout << "Processed Mel spectrogram: " << mel_spec.size() << std::endl;
        for (int i = 0; i < 20 && i < mel_spec.size(); ++i) {
            std::cout << mel_spec[i] << " ";
        }
        std::cout << std::endl;

        // Clean up resources
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        return GST_FLOW_OK; // Return to stop further processing
    }

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}


int main(int argc, char *argv[]) {
    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Build pipeline
    std::string pipeline_str = "filesrc location=" + std::string(argv[1]) + " ! "
                               "decodebin ! "
                               "audioconvert ! "
                               "audioresample ! "
                               "audio/x-raw,format=F32LE,channels=1,rate=" + std::to_string(SAMPLE_RATE) + " ! "
                               "appsink name=sink sync=false";

    pipeline = gst_parse_launch(pipeline_str.c_str(), NULL);
    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");

    // Configure appsink
    GstCaps *caps = gst_caps_new_simple("audio/x-raw",
                                        "format", G_TYPE_STRING, "F32LE",
                                        "rate", G_TYPE_INT, SAMPLE_RATE,
                                        "channels", G_TYPE_INT, 1,
                                        NULL);
    gst_app_sink_set_caps(GST_APP_SINK(appsink), caps);
    gst_caps_unref(caps);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Set up appsink's new-sample signal handler
    GstAppSinkCallbacks callbacks = { NULL, NULL, new_sample };
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks, NULL, NULL);

    // Run main loop
    GstBus *bus = gst_element_get_bus(pipeline);
    gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    gst_object_unref(bus);

    // Clean up
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    gst_deinit();

    return 0;
}

