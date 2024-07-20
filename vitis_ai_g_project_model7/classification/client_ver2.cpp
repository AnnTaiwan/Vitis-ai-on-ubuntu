#include <iostream>
#include <string>
#include <fstream>
#include <curl/curl.h>
#include <json-c/json.h> // json-c header

using namespace std;

string read_file(string filename)
{
    ifstream file(filename);
    string content((istreambuf_iterator<char>(file)), (istreambuf_iterator<char>()));
    return content;
}

int upload(string url, string strData)
{
    CURL *curl;
    CURLcode res;
    struct curl_slist *headers = NULL;

    curl = curl_easy_init();
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "charset: utf-8");

    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strData.c_str());

    res = curl_easy_perform(curl);
    if (res != CURLE_OK)
        cerr << "Error: " << curl_easy_strerror(res) << endl;

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    return res;
}

int main(void)
{
    std::string filepath = "Result.txt";
    std::string readBuffer = read_file(filepath);

    // Construct JSON object using json-c
    json_object *root = json_object_new_object();
    json_object *text = json_object_new_string(readBuffer.c_str());
    json_object_object_add(root, "text", text);

    // Print JSON data
    cout << "json data: " << json_object_to_json_string(root) << endl;

    // Convert json_object to string
    string strData = json_object_to_json_string(root);

    // Clean up json_object
    json_object_put(root);

    string url = "https://ml-project-1r0x.onrender.com/receive_text";
    int res = upload(url, strData);
    cout << "Upload result: " << res << endl;

    return 0;
}

