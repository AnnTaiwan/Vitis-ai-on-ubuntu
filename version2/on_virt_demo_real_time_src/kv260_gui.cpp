#include <gtk/gtk.h>
#include <pthread.h>
#include <atomic>
#include <cstdlib>
#include <string>

// Atomic flags for thread control
std::atomic<bool> stopRecording(false);
std::atomic<bool> stopComputing(false);

// GTK Widgets for status updates
GtkWidget *status_label;

// Function for recording audio in a loop
void* recordAudio(void* arg) {
    stopRecording = false;
    while (!stopRecording) {
        int ret = system("./recording_gstreamer.sh");  // Capture the return value to avoid warnings
        if (ret != 0) {
            g_print("Error executing recording script\n");
        }
    }
    return nullptr;
}

// Function for computing audio in a loop
void* processAudio(void* arg) {
    stopComputing = false;
    while (!stopComputing) {
        int ret = system("./CH_predict_audio_real_time.sh");  // Capture the return value to avoid warnings
        if (ret != 0) {
            g_print("Error executing computing script\n");
        }
    }
    return nullptr;
}

// Thread references
pthread_t recordThread;
pthread_t computeThread;

// Start recording thread
void start_recording(GtkWidget *widget, gpointer data) {
    if (!stopRecording) {
        gtk_label_set_text(GTK_LABEL(status_label), "Status: Recording");
        pthread_create(&recordThread, nullptr, recordAudio, nullptr);
    }
}

// Start computing thread
void start_processing(GtkWidget *widget, gpointer data) {
    if (!stopComputing) {
        gtk_label_set_text(GTK_LABEL(status_label), "Status: Processing");
        pthread_create(&computeThread, nullptr, processAudio, nullptr);
    }
}

// Stop both threads
void stop_processes(GtkWidget *widget, gpointer data) {
    stopRecording = true;
    stopComputing = true;

    // Wait for threads to finish
    if (recordThread) pthread_join(recordThread, nullptr);
    if (computeThread) pthread_join(computeThread, nullptr);

    gtk_label_set_text(GTK_LABEL(status_label), "Status: Stopped");
}
// GUI setup and main loop
int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    // Main window
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "KV260 Audio Processing GUI");
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);
    gtk_widget_set_size_request(window, 300, 200);

    // Vertical layout container using non-deprecated gtk_box_new
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    // Start recording button
    GtkWidget *record_button = gtk_button_new_with_label("Start Recording");
    g_signal_connect(record_button, "clicked", G_CALLBACK(start_recording), NULL);
    gtk_box_pack_start(GTK_BOX(vbox), record_button, FALSE, FALSE, 0);

    // Start processing button
    GtkWidget *process_button = gtk_button_new_with_label("Start Processing");
    g_signal_connect(process_button, "clicked", G_CALLBACK(start_processing), NULL);
    gtk_box_pack_start(GTK_BOX(vbox), process_button, FALSE, FALSE, 0);

    // Status label
    status_label = gtk_label_new("Status: Idle");
    gtk_box_pack_start(GTK_BOX(vbox), status_label, FALSE, FALSE, 0);

    // Stop button
    GtkWidget *stop_button = gtk_button_new_with_label("Stop");
    g_signal_connect(stop_button, "clicked", G_CALLBACK(stop_processes), NULL);
    gtk_box_pack_start(GTK_BOX(vbox), stop_button, FALSE, FALSE, 0);

    // Signal to close the window
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    // Show all widgets
    gtk_widget_show_all(window);

    // Start GTK main loop
    gtk_main();

    return 0;
}

