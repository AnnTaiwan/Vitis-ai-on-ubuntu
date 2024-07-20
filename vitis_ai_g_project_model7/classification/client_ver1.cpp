#include <iostream>
#include <string>
#include <fstream>
#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/reader.h>
using namespace std;

string read_file(string filename)
{
    ifstream file(filename);
    // 讀取整個文件中的值到string中
    string content((istreambuf_iterator<char>(file)), (istreambuf_iterator<char>()));
    return content;
}

int upload(string url, string strData)
{
    CURL *curl;
    CURLcode res;
    struct curl_slist *headers = NULL;  // an linked list structure for storing strings

    curl = curl_easy_init();    // initializer easy handler
    // add header data
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "charset: utf-8");

    // set up some attributes
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strData.c_str());

    res = curl_easy_perform(curl);
    if(res)
        printf("error: %s\n", curl_easy_strerror(res));

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers); // free the list

    return int(res);
}

int main(void)
{

    // 指定file path
    std::string filepath = "test.txt";
    // 讀取file
    std::string readBuffer = read_file(filepath);

    Json::Value root;   // creates a JSON object
    root["text"] = readBuffer;  // add a key-value pair to the object

    // JSON object to string
    Json::FastWriter fw;
    cout << "json data: " << fw.write(root) << endl;

    string strData = root.toStyledString();
    string url = "https://ml-project-1r0x.onrender.com/receive_text";

    int res = upload(url, strData);
    cout << res << endl;

    return 0;
}
