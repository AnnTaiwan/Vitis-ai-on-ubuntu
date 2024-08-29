#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <fstream> // For file operations

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/audio/audio.h>
#include <gst/fft/gstfftf32.h>

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <limits>

#define SAMPLE_RATE 22050
#define DURATION 5.0
#define AUDIO_LEN (SAMPLE_RATE * DURATION)
#define N_MELS 128
#define N_FFT 2048
#define HOP_LEN 512
#define SPEC_WIDTH 216 // (AUDIO_LEN / HOP_LEN + 1)
#define FMAX (SAMPLE_RATE / 2)
#define SPEC_SHAPE {SPEC_WIDTH, N_MELS}

static GstElement *pipeline;
static GstElement *appsink;
static std::deque<float> audio_buffer; // Buffer to store incoming audio samples
// Global segment counter, starting from 1
static int segment_count = 1;

// Function to apply Hamming window
std::vector<float> apply_hamming_window(const std::vector<float>& signal) {
    std::vector<float> windowed_signal(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        windowed_signal[i] = signal[i] * (0.54 - 0.46 * cos(2 * M_PI * i / (signal.size() - 1)));
    }
    return windowed_signal;
}

// used for creating mel_bank
// Function to compute FFT frequencies
std::vector<float> fft_frequencies(float sr, int n_fft) {
    std::vector<float> freqs(n_fft / 2 + 1);
    // Calculate the frequency for each FFT bin
    for (int i = 0; i <= n_fft / 2; ++i) {
        freqs[i] = i * sr / n_fft;
    }
    return freqs;
}

// Helper function to convert frequency to Mel scale
float hz_to_mel(float freq, bool htk = false) {
    if (htk) {
        // HTK Mel scale: 2595 * log10(1 + freq / 700)
        return 2595.0f * std::log10(1.0f + freq / 700.0f);
    } else {
        // Slaney Mel scale
        const float f_min = 0.0f;
        const float f_sp = 200.0f / 3;
        float mels = (freq - f_min) / f_sp;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - f_min) / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;

        if (freq >= min_log_hz) {
            mels = min_log_mel + std::log(freq / min_log_hz) / logstep;
        }

        return mels;
    }
}

// Helper function to convert Mel scale to frequency
float mel_to_hz(float mel, bool htk = false) {
    if (htk) {
        // HTK Mel scale: 700 * (10^(mel / 2595) - 1)
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    } else {
        // Slaney Mel scale
        const float f_min = 0.0f;
        const float f_sp = 200.0f / 3;
        float freqs = f_min + f_sp * mel;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - f_min) / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;

        if (mel >= min_log_mel) {
            freqs = min_log_hz * std::exp(logstep * (mel - min_log_mel));
        }

        return freqs;
    }
}

// Helper function to generate Mel frequencies
std::vector<float> mel_frequencies(int n_mels, float fmin, float fmax, bool htk) {
    std::vector<float> mels(n_mels);
    // Calculate the minimum and maximum Mel frequencies
    float min_mel = hz_to_mel(fmin, htk);
    float max_mel = hz_to_mel(fmax, htk);
    // Calculate the step size between Mel frequencies
    float mel_step = (max_mel - min_mel) / (n_mels - 1);

    // Generate Mel frequency points and convert back to Hz frequencies
    for (int i = 0; i < n_mels; ++i) {
        mels[i] = mel_to_hz(min_mel + i * mel_step, htk);
    }

    return mels;
}

// Function to normalize filter weights
void normalize(std::vector<std::vector<float>>& weights, const std::string& norm) {
    if (norm == "slaney") {
        for (size_t i = 0; i < weights.size(); ++i) {
            float sum = 0.0f;
            // Calculate the sum of weights for each filter
            for (size_t j = 0; j < weights[i].size(); ++j) {
                sum += weights[i][j];
            }
            // Normalize the weights if the sum is not zero
            if (sum != 0.0f) {
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    weights[i][j] /= sum;
                }
            }
        }
    } else {
        throw std::invalid_argument("Unsupported norm=" + norm);
    }
}

// Function to create Mel filter-bank
std::vector<std::vector<float>> generate_mel_filter_bank(float sample_rate, int n_fft, int n_mels, int fmax) {
    std::vector<std::vector<float>> mel_filter_bank(n_mels, std::vector<float>(n_fft / 2 + 1, 0.0f));

    // Calculate Mel frequency points
    float mel_min = hz_to_mel(0);
    float mel_max = hz_to_mel(fmax);
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }

    // Convert Mel points back to Hz
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Map Hz points to FFT bin numbers
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bin_points[i] = static_cast<int>(floor((n_fft + 1) * hz_points[i] / sample_rate));
    }

    // Create Mel filter bank
    for (int i = 1; i <= n_mels; ++i) {
        int start = bin_points[i - 1];
        int center = bin_points[i];
        int end = bin_points[i + 1];

        // Compute the weights for each filter
        for (int j = start; j < center; ++j) {
            mel_filter_bank[i - 1][j] = (j - start) / static_cast<float>(center - start);
        }
        for (int j = center; j < end; ++j) {
            mel_filter_bank[i - 1][j] = (end - j) / static_cast<float>(end - center);
        }
    }

    // Normalize the filter bank
    normalize(mel_filter_bank, "slaney");

    return mel_filter_bank;
}


///////////////////////////////
// Function to convert power spectrogram to dB scale
// use global value as max to cal with top_db, this way can create a more correct picture
std::vector<float> power_to_db(const std::vector<float>& power_spec, float ref_value = 1.0, float amin = 1e-10, float top_db = 80.0) {
    std::vector<float> db_spec(power_spec.size());
    float log_spec_ref = 10.0 * log10(ref_value);

    // First pass: Convert power to dB scale and find the max value
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < power_spec.size(); ++i) {
        float val = std::max(power_spec[i], amin);
        float magnitude_db = 10.0 * log10(val) - log_spec_ref;
        db_spec[i] = magnitude_db;
        if (magnitude_db > max_val) {
            max_val = magnitude_db;
        }
    }

    // Apply top_db threshold based on the global max value
    float threshold = max_val - top_db;
    for (size_t i = 0; i < db_spec.size(); ++i) {
        db_spec[i] = std::max(db_spec[i], threshold);
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
    //std::cout << "mel_spec.max_size(): " <<  mel_spec.max_size() << std::endl; // 2305843009213693951
    //std::cout << "AAA" << std::endl;
    // get the mel_filter_bank to turn the scale into mel_scale
    // mel_filter bank's shape = (n_mels, N_FFT / 2 + 1)
    std::vector<std::vector<float>> mel_filter_bank = generate_mel_filter_bank(sr, fft_size, num_fft_bins, FMAX);
	//std::cout << "BBB" << audio.size() << std::endl;
	// apply stft on audio
    for (size_t i = 0; i < SPEC_WIDTH; ++i) { // has SPEC_WIDTH frames totally
        size_t start = i * HOP_LEN;
        size_t end = std::min(start + fft_size, audio.size());
        std::vector<float> segment(audio.begin() + start, audio.begin() + end);
		//std::cout << "CCC" << segment.size() << std::endl;
        // Apply Hamming window
        segment = apply_hamming_window(segment);

        GstFFTF32Complex *fft_result = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);
        GstFFTF32Complex *fft_input = (GstFFTF32Complex *) g_malloc(sizeof(GstFFTF32Complex) * fft_size);

        // Prepare fft_input (complex to complex)
        for (guint j = 0; j < fft_size; ++j) {
            fft_input[j].r = (j < segment.size()) ? segment[j] : 0.0f; // padding 0 if frame isn't big enough
            fft_input[j].i = 0.0f; // Assuming no imaginary part for simplicity
        }

        // Perform FFT
        gst_fft_f32_fft(fft, (const gfloat *)fft_input, fft_result);
		//std::cout << "DDD" << std::endl;
        // Compute power spectrum, like np.abs(spec)**2, turn complex into float
        std::vector<float> power_spectrum(fft_size / 2 + 1, 0.0f); // just need first half value
        //std::cout << "EEE" << std::endl;
        for (guint j = 0; j < fft_size / 2 + 1; ++j) {
            power_spectrum[j] = pow(fft_result[j].r, 2) + pow(fft_result[j].i, 2);
        }
		//std::cout << "HHH" << std::endl;
        // Apply Mel filter bank on each frame
        // like the rows in mel_filter_bank(128, 1025) dot(*) the column of power_spectrum(1025, 216) = final_spec(128, 216)
        for (guint m = 0; m < num_fft_bins; ++m) {
            float mel_value = 0.0f;
            for (guint k = 0; k < fft_size / 2 + 1; ++k) {
                mel_value += mel_filter_bank[m][k] * power_spectrum[k];
            }
            mel_spec[i * num_fft_bins + m] = mel_value; // Logarithmic scaling
        }

        // Clean up FFT resources
        g_free(fft_result);
        g_free(fft_input);
    }
	///std::cout << "FFF" << std::endl;
    gst_fft_f32_free(fft);

    // Convert power spectrogram to dB scale
    mel_spec = power_to_db(mel_spec);
	///std::cout << "GGG" << std::endl;
    return mel_spec;
}



// Function to write Mel spectrogram to a .txt file
void write_mel_spectrogram_to_txt(const std::vector<float>& mel_spec, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < mel_spec.size(); ++i) {
            file << mel_spec[i] << " ";
            if ((i + 1) % N_MELS == 0) {
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
    
	// std::cout << "audio_buffer.size() " << audio_buffer.size() << std::endl; 
    // While there are enough samples to process a 5-second segment
    while (audio_buffer.size() >= AUDIO_LEN) {
        // Extract 5-second segment from the buffer
        std::vector<float> audio_segment(audio_buffer.begin(), audio_buffer.begin() + AUDIO_LEN);

        // Get Mel spectrogram for the current segment
        std::vector<float> mel_spec = get_mel_spectrogram(audio_segment, SAMPLE_RATE);

        // Save Mel spectrogram to file with global segment count
        std::string filename = "mel_spectrogram_segment_" + std::to_string(segment_count++) + ".txt";
        write_mel_spectrogram_to_txt(mel_spec, filename);

        // Remove the processed samples from the buffer
        audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + AUDIO_LEN);
    }


    // Clean up resources
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}



int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout << "Error: Please add the audio files after the execution command, like ./cal_mel_spec_ver2 <files.wav>" << std::endl;
        return 0;
    }
    
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
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks, pipeline, NULL);

    // Run main loop
    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg;
    bool terminate = false;

    while (!terminate) {
        msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
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
                case GST_MESSAGE_EOS: // Here, deal with the last segment, which is padding the last audio into `AUDIO_LEN`
                    std::cout << "End-Of-Stream reached." << std::endl;
                
		            // Handle EOS: pad final segment if necessary
		            if (audio_buffer.size() > 0 && audio_buffer.size() < AUDIO_LEN) {
		                //std::cout << "Padding audio_buffer.size() " << audio_buffer.size() << std::endl;

		                // Convert deque to vector
		                std::vector<float> audio_vector(audio_buffer.begin(), audio_buffer.end());

		                // Calculate how many samples are needed to reach AUDIO_LEN
		                size_t remaining_samples = AUDIO_LEN - audio_vector.size();

		                // Create a new vector to hold the padded audio
		                std::vector<float> padded_buffer(audio_vector);

		                // Repeat the original audio vector's content to fill the remaining space
		                while (remaining_samples > 0) {
		                    size_t to_copy = std::min(remaining_samples, audio_vector.size());
		                    padded_buffer.insert(padded_buffer.end(), audio_vector.begin(), audio_vector.begin() + to_copy);
		                    remaining_samples -= to_copy;
		                }

		                // Process the final padded segment
		                std::vector<float> audio_segment(padded_buffer.begin(), padded_buffer.begin() + AUDIO_LEN);
		                std::vector<float> mel_spec = get_mel_spectrogram(audio_segment, SAMPLE_RATE);

		                // Save the final Mel spectrogram
		                std::string filename = "mel_spectrogram_segment_" + std::to_string(segment_count++) + ".txt";
		                write_mel_spectrogram_to_txt(mel_spec, filename);

		                // Clear the buffer
		                audio_buffer.clear();
		            }

		            terminate = true;
		            break;
                default:
                    // We should not reach here because we only asked for ERRORs and EOS
                    std::cerr << "Unexpected message received." << std::endl;
                    break;
            }
            gst_message_unref(msg);
        }
    }

    // Clean up
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    gst_object_unref(bus);
    gst_deinit();

    return 0;
}

