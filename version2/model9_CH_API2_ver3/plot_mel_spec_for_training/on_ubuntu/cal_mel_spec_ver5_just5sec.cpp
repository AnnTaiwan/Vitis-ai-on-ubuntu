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
#define CENTER true // used in STFT padded mode

static GstElement *pipeline;
static GstElement *appsink;
static std::deque<double> audio_buffer; // Buffer to store incoming audio samples
// Global segment counter, starting from 1
static int segment_count = 1;
static std::string dest_txt_folder_name = "";
static std::string txt_name = "";
// Function to generate Hanning window
std::vector<float> hanning_window(int window_Size) {
    std::vector<float> window(window_Size);
    for (int i = 0; i < window_Size; ++i) {
        window[i] = 0.5 - 0.5 * cos(2.0f * M_PI * i / (window_Size - 1));
    }
    return window;
}

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
std::vector<double> fft_frequencies(float sr, int n_fft) {
    std::vector<double> freqs(n_fft / 2 + 1);
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
std::vector<double> mel_frequencies(int n_mels, float fmin, float fmax, bool htk) {
    std::vector<double> mels(n_mels);
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
// Function to generate Mel filter bank
std::vector<std::vector<float>> generate_mel_filter_bank(int sr, int n_fft, int n_mels, float fmin, float fmax) {
    std::vector<std::vector<float>> mel_filter_bank(n_mels, std::vector<float>(n_fft / 2 + 1, 0.0f));

    // Compute mel points
    std::vector<float> mel_points(n_mels + 2);
    float mel_fmin = 2595.0f * std::log10(1.0f + fmin / 700.0f);
    float mel_fmax = 2595.0f * std::log10(1.0f + fmax / 700.0f);

    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = 700.0f * (std::pow(10.0f, (mel_fmin + (mel_fmax - mel_fmin) * i / (n_mels + 1)) / 2595.0f) - 1.0f);
    }

    // Convert Hz to FFT bin numbers
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bin_points[i] = static_cast<int>(std::floor((n_fft + 1) * mel_points[i] / sr));
    }

    // Create filters
    for (int i = 1; i <= n_mels; ++i) {
        for (int j = bin_points[i - 1]; j < bin_points[i]; ++j) {
            mel_filter_bank[i - 1][j] = (j - bin_points[i - 1]) / static_cast<float>(bin_points[i] - bin_points[i - 1]);
        }
        for (int j = bin_points[i]; j < bin_points[i + 1]; ++j) {
            mel_filter_bank[i - 1][j] = (bin_points[i + 1] - j) / static_cast<float>(bin_points[i + 1] - bin_points[i]);
        }
    }

    return mel_filter_bank;
}

///////////////////////////////
// Function to convert power spectrogram to dB scale
// use global value as max to cal with top_db, this way can create a more correct picture
// Function to convert power spectrogram to dB scale
std::vector<std::vector<float>> power_to_db(
    const std::vector<std::vector<float>>& mel_spec, 
    float ref = 1.0f, 
    float amin = 1e-10f, 
    float top_db = 80.0f
) {
    int num_frames = mel_spec.size();
    int n_mels = mel_spec[0].size();

    std::vector<std::vector<float>> log_mel_spec(num_frames, std::vector<float>(n_mels));

    float max_val = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < n_mels; ++j) {
            float value = std::max(mel_spec[i][j], amin);
            float db = 10.0f * std::log10(value);
            log_mel_spec[i][j] = db;
            if (db > max_val) {
                max_val = db;
            }
        }
    }

    float lower_bound = max_val - top_db;
    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < n_mels; ++j) {
            log_mel_spec[i][j] = std::max(log_mel_spec[i][j], lower_bound);
        }
    }

    return log_mel_spec;
}

// Function to compute STFT
std::vector<std::vector<GstFFTF32Complex>> compute_stft(
    const std::vector<float>& audio, 
    const std::vector<float>& window, 
    int n_fft, 
    int hop_length) {
    std::vector<std::vector<GstFFTF32Complex>> stft_result;

    // initial FFT from Gstreamer
    GstFFTF32* fft = gst_fft_f32_new(n_fft, false);

    // apply FFT
    for (int frame = 0; frame + n_fft < audio.size(); frame += hop_length) {
        std::vector<float> segment(audio.begin() + frame, audio.begin() + frame + n_fft);
        std::vector<float> windowed_segment(n_fft);
        for (int i = 0; i < n_fft; ++i) {
            windowed_segment[i] = segment[i] * window[i];
        }

        GstFFTF32Complex out_fft[n_fft];
        gst_fft_f32_fft(fft, windowed_segment.data(), out_fft);

        std::vector<GstFFTF32Complex> frame_fft(out_fft, out_fft + n_fft / 2 + 1);
        stft_result.push_back(frame_fft);
    }

    gst_fft_f32_free(fft);

    return stft_result;
}

// Function to compute power spectrogram
std::vector<std::vector<float>> compute_power_spectrogram(const std::vector<std::vector<GstFFTF32Complex>>& stft_matrix) {
    int spec_height = stft_matrix[0].size();
    int spec_width = stft_matrix.size();

    std::vector<std::vector<float>> power_spec(spec_height, std::vector<float>(spec_width, 0.0f));

    for (int i = 0; i < spec_width; ++i) {
        for (int j = 0; j < N_FFT /2 + 1; ++j) {
            power_spec[j][i] = pow(stft_matrix[i][j].r, 2) + pow(stft_matrix[i][j].i, 2);
        }
    }
    return power_spec;
}


// Function to apply Mel filter bank to power spectrogram
std::vector<std::vector<float>> apply_mel_filter_bank(
    const std::vector<std::vector<float>>& power_spec, 
    const std::vector<std::vector<float>>& mel_filter_bank
) {
    int num_frames = power_spec[0].size();
    int n_mels = mel_filter_bank.size();
    // (1025 , 216) and (128 , 1025)
    // std:: cout << power_spec.size() << " " << power_spec[0].size()
    // << " " << mel_filter_bank.size() << " " << mel_filter_bank[0].size() << "\n";

    std::vector<std::vector<float>> mel_spec(n_mels, std::vector<float>(num_frames, 0.0f));

    for (int i = 0; i < n_mels; ++i) {
        for (int j = 0; j < num_frames; ++j) {
            for (size_t k = 0; k < mel_filter_bank[i].size(); ++k) {
                mel_spec[i][j] += mel_filter_bank[i][k] * power_spec[k][j];
            }
        }
    }

    return mel_spec;
}

std::vector<std::vector<float>> get_mel_spectrogram(const std::vector<float>& audio, int sr) {
    guint fft_size = N_FFT;
    guint num_fft_bins = N_MELS;

    // Calculate padding length if CENTER mode is enabled
    int paddingLength = 0;
    std::vector<float> paddedSignal;
    if (CENTER) {
    int pad_Length = fft_size / 2;
    int audioLength = audio.size();

    // Create a new padded signal vector with the required padding
    paddedSignal.resize(pad_Length + audioLength + pad_Length, 0.0f);

    // Copy the original audio data into the new padded signal vector
    std::copy(audio.begin(), audio.end(), paddedSignal.begin() + pad_Length);
    } else {
        // If no padding is required, just copy the original signal
        paddedSignal = audio;
    }

    // Create a Hanning window of appropriate length
    std::vector<float> window(fft_size);
    window = hanning_window(fft_size);

    // Compute STFT
    auto stft_matrix = compute_stft(paddedSignal, window, N_FFT, HOP_LEN);
    // std::cout << "stft_matrix: " << stft_matrix.size() 
    // << " " << stft_matrix[0].size() << "\n"; // (216, 1025)

    // Compute power spectrogram
    auto power_spec = compute_power_spectrogram(stft_matrix);
    // std::cout << "power_spec: " << power_spec.size() 
    // << " " << power_spec[0].size() << "\n"; // (1025 , 216)

    // Generate Mel filter bank
    auto mel_filter_bank = generate_mel_filter_bank(sr, N_FFT, N_MELS, 0.0, FMAX);
    // std::cout << "mel_filter_bank: " << mel_filter_bank.size() 
    // << " " << mel_filter_bank[0].size() << "\n"; // (128, 1025)

    // Prepare Mel spectrogram container
    std::vector<std::vector<float>> mel_spec(N_MELS, std::vector<float>(SPEC_WIDTH, 0.0f));
    // Apply Mel filter bank
    mel_spec = apply_mel_filter_bank(power_spec, mel_filter_bank);
    // std::cout << "mel_spec: " << mel_spec.size() 
    // << " " << mel_spec[0].size() << "\n";

    // Convert to dB scale
    auto log_mel_spec = power_to_db(mel_spec);
    // (128, 216) in result
    // std::cout << "Gel_mel_spectrogram's shape : (" << log_mel_spec.size() << " " 
    //           << log_mel_spec[0].size() << ")\n";

    return log_mel_spec;
}

// Function to write Mel spectrogram to a .txt file
void write_mel_spectrogram_to_txt(const std::vector<std::vector<float>>& mel_spec, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < mel_spec.size(); ++i) {
            for (size_t j =0; j < SPEC_WIDTH; j++){
                file << mel_spec[i][j] << " ";
            }
            file << "\n";
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
        std::vector<std::vector<float>> mel_spec = get_mel_spectrogram(audio_segment, SAMPLE_RATE);

        // Save Mel spectrogram to file with global segment count
        std::string filename = dest_txt_folder_name + "/" + txt_name;
		if(segment_count == 1)
		{
			segment_count++;
			write_mel_spectrogram_to_txt(mel_spec, filename);
		}
        // Remove the processed samples from the buffer
        audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + AUDIO_LEN);
    }


    // Clean up resources
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}



int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cout << "Error: Please add the audio files after the execution command, like ./cal_mel_spec_ver2 <files.wav> <txt_folder_name>, and output will be <txt_folder_name/files_mel_spec_text.txt>" << std::endl;
        return 0;
    }
    dest_txt_folder_name = std::string(argv[2]);
    
    // Get the base name of the audio file (without directory and extension)
	std::string audio_name = std::string(argv[1]);

	// Find the last '/' or '\' to remove directory path
	size_t lastSlash = audio_name.find_last_of("/\\");
	if (lastSlash != std::string::npos) {
	    audio_name = audio_name.substr(lastSlash + 1);  // Keep only the base name
	}

	// Find the last '.' to remove the extension
	size_t lastDot = audio_name.find_last_of(".");
	if (lastDot != std::string::npos) {
	    audio_name = audio_name.substr(0, lastDot);  // Remove the extension
	}

	// Set new txt name
	txt_name = audio_name + "_mel_spec_text.txt";

    
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
		                std::vector<std::vector<float>> mel_spec = get_mel_spectrogram(audio_segment, SAMPLE_RATE);

		                // Save the final Mel spectrogram
		                std::string filename = dest_txt_folder_name + "/" + txt_name;
		                if(segment_count == 1)
						{
							segment_count++;
							write_mel_spectrogram_to_txt(mel_spec, filename);
						}

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
