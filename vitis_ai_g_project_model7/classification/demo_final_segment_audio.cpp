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
 /*
 	Used for several segments of an audio, output a txt to show the fake and real number.
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
#include<vector>
using namespace std;
using namespace cv;
// Function to compute the softmax of a vector
std::vector<float> softmax(const std::vector<float>& input) {
    // Step 1: Compute the maximum value in the input vector
    float max_val = *max_element(input.begin(), input.end());

    // Step 2: Compute the sum of exp(input[i] - max_val)
    float sum = 0.0;
    std::vector<float> exp_values(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        exp_values[i] = std::exp(input[i] - max_val);
        sum += exp_values[i];
    }

    // Step 3: Divide each exp_value by the sum to get the softmax output
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp_values[i] / sum;
    }

    return output;
}


cv::Mat crop_with_points(const cv::Mat& img) {
    std::vector<cv::Point> points = { cv::Point(79, 57), cv::Point(575, 428), cv::Point(575, 57), cv::Point(79, 426) };
    
    // Define the four points
    int x1 = points[0].x;
    int y1 = points[0].y;
    int x2 = points[1].x;
    int y2 = points[1].y;
    int x3 = points[2].x;
    int y3 = points[2].y;
    int x4 = points[3].x;
    int y4 = points[3].y;

    // Find the bounding box for cropping
    int left = std::min({x1, x4});
    int upper = std::min({y1, y2});
    int right = std::max({x2, x3});
    int lower = std::max({y3, y4});
	//printf("%d, %d, %d, %d\n", left, upper, right, lower);
	//std::cout << "Original Image size: " << img.rows << " rows x " << img.cols << " cols" << std::endl;
	//std::cout << "Original Image shape: " << img.size() << std::endl;
	
    // Crop the image
    cv::Rect crop_region(left, upper, right - left, lower - upper);
    cv::Mat cropped_img = img(crop_region);

    return cropped_img.clone();
}

int main(int argc, char* argv[]) {

  	if (argc != 4) {
		std::cerr << "Please input a model name as the first param!" << std::endl;
        std::cerr << "Please input your image path as the second param!" << std::endl;
        std::cerr << "The third param is a txt to store results!" << std::endl;
        return -1; // Exit with error code
    }
    // Initialize Google Logging
    google::InitGoogleLogging(argv[0]);
    
  	//string model = argv[1] + string("_acc");
  	cv::String path = argv[2];
  	std::ofstream out_fs(argv[3], std::ofstream::out);
  	int length = path.size();
  
  	cout << "############# Initial information ################" << endl;
  	auto kernel_name = argv[1]; // User should input /usr/share/vitis_ai_library/models/CNN_model7_netname/CNN_model7_netname.xmodel
	cout << "Model name: " << kernel_name << endl;
	cout << "IMAGE FOLDER PATH: " << path << endl;
  	cout << "OUTPUT file name: " << argv[3] << endl;
  	
	// join the path of all the images
	vector<cv::String> files;
	cv::glob(path, files);  // files[0] will be 'folder_name/aa.png'
	int count = files.size();
	
	cout << "The image count = " << count << endl;
	
	// Read image from a path.
  	vector<Mat> imgs;
  	vector<string> imgs_names;
  	for (int i = 0; i < count; i++) 
  	{
  	  // image file names.
  	  auto image = cv::imread(files[i]);
  	  
  	  cout << "image.channels: " << image.channels() << endl;
  	  if (image.empty()) {
 	     std::cerr << "Cannot load " << files[i] << std::endl;
 	     continue;
  	  }
  	  imgs.push_back(image);
  	  imgs_names.push_back(string(cv::String(files[i]).substr(length)));
  	}
  	
 	if (imgs.empty()) 
 	{
 	   std::cerr << "No image load success!" << std::endl;
 	   return -1; // Exit with error code
	}
	
	cout << "############# Start predicting ##################" << endl;
	cout << "Create a dpu task object." << endl;
	// Create a dpu task object.
 	auto task = vitis::ai::DpuTask::create(kernel_name);
 	if (!task) {
        std::cerr << "Failed to create DpuTask with model: " << kernel_name << std::endl;
        return -1; // Exit with error code
    }
 	auto batch = task->get_input_batch(0, 0);
 	cout << "batch " << batch << endl;
 	
 	// Set the mean values and scale values.
 	task->setMeanScaleBGR({0.0f, 0.0f, 0.0f}, {0.00392157f, 0.00392157f, 0.00392157f}); // subtract the mean, multiply the scale
 	//task->setMeanScaleBGR({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
 	// 0.00392157f is equivalent to 1/256, which is commonly used to scale pixel values from the range [0, 255] to [0, 1].
    //task->setMeanScaleBGR({0.23829615f, 0.33467807f, 0.46707764f}, {8.2451f, 3.7922f, 8.2363f});
    //task->setMeanScaleBGR({0.23829615f, 0.33467807f, 0.46707764f}, {0.12129097f, 0.26373705f, 0.12141656f});
	
	// due to transforms.Normalize(mean=[0.23829615, 0.33467807, 0.46707764], std=[0.12129097, 0.26373705, 0.12141656]) in training model
 	//task->setMeanScaleBGR({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
 	// void setMeanScaleBGR(const std::vector< float > &mean, const std::vector< float > &scale)=0;
  	auto input_tensor = task->getInputTensor(0u);
  	cout << "############# Input tensor information ################" << endl;
  	cout << "input_tensor" << input_tensor[0] << endl; 
  	cout << "\n**input_tensor.dtype" << input_tensor[0].dtype << endl;
  	CHECK_EQ((int)input_tensor.size(), 1) 
  		<< " the dpu model must have only one input";
  	// get the needed img size
  	auto width = input_tensor[0].width; 
  	auto height = input_tensor[0].height;
  	auto size = cv::Size(width, height); 
  	cout << "Input image info:" << endl;
  	cout << "\twidth " << width << endl;
  	cout << "\theight " << height << endl;
  	cout << "\tsize " << size << endl;
  	cout << "############ Input tensor information END ##############" << endl;
  	// Create a config and set the correlating data to control post-process.
  	//vitis::ai::proto::DpuModelParam config;
  	// Fill all the parameters.
 	//auto ok = google::protobuf::TextFormat::ParseFromString(cnn_model6_int_config, &config);
  	//if (!ok) {
   	// 	cerr << "Set parameters failed!" << endl;
   	// 	abort();
  	//}
	
	vector<Mat> inputs;
 	vector<int> input_cols, input_rows;
 	int fake = 0, real = 0; // record the final result of how many real and fake in total
 	double fake_value = 0, real_value = 0;
  	for (long unsigned int i = 0, j = -1; i < imgs.size(); i++) {
		/* Pre-process Part */
		// (1) delete margin
    	//cv::imwrite("original.jpg", imgs[i]);
		
		cv::Mat cropped_image = crop_with_points(imgs[i]);
		
		//cv::imwrite("cropped.jpg", cropped_image);
		// (2) Resize it if its size is not match.
		cv::Mat image; // accept the image after resized
		
		input_cols.push_back(cropped_image.cols);
		input_rows.push_back(cropped_image.rows);
		cout << "Before resize " << cropped_image.size() << endl;
		if (size != cropped_image.size()) {
			//cropped_image.convertTo(cropped_image, CV_32F, 1.0 / 255.0);
		  	cv::resize(cropped_image, image, size, 0, 0, cv::INTER_LINEAR); // void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR );
		  	//image.convertTo(image, CV_32F, 1.0 / 255.0);
    		//cv::imwrite("Resized.jpg", image);
		  	//cout << "After resizing, image.channels: " << image.channels() << endl;
		} else {
		  	image = cropped_image;
		}
		
		// convert the pizel data from 0~255 to 0~1
		//image.convertTo(image, CV_32F, 1.0 / 255.0);
		
		inputs.push_back(image);
		j++;
		if (j < batch - 1 && i < imgs.size() - 1) {
		  	continue;
		}
		
		for (const auto& img : inputs) {
			cout << "After resizing, Image size: " << img.size() << endl;
		}
		
		
		
		// Set the input images into dpu.
		task->setImageRGB(inputs);
		/* DPU Runtime */
		// Run the dpu.
		task->run(0u);

	
		/* Post-process part */
		// Get output
		
		// Retrieve the first output tensor (e.g., class probabilities)
		auto output_tensors = task->getOutputTensor(0u); // 'u' is unsigned
		
		inputs.clear();
		
		cout << i << "th: Finish image=> " << imgs_names[i] << endl;
		cout << "############## Output tensor information ##################" << endl;
		//cout << "Output tensor's size: " << output_tensors.size() << endl; // only 1 [batch_size, 
		for(int x = 0; x < output_tensors.size(); x++){
			cout << output_tensors[x] << " ";
			// format will be like :
			// output [ CNN_model6__CNN_model6_Sequential_class_layers__ModuleList_0__Linear_1__248_fix ] 
			// {, size=2, h=1, w=1, c=2, fixpos=4, virt= (0xffff9b55b000 )}
		}
		cout << endl;
		cout << "############ Output tensor information END ###############" << endl;
		
		
		cout << "\n############ RESULT ###############" << endl;
		for (const auto& output_tensor : output_tensors) {
			const int8_t* raw_data = nullptr; // Initialize raw_data to nullptr

			if (output_tensor.dtype == vitis::ai::library::DT_INT8) {
				// reinterpret_cast: convert one pointer type to another pointer type.
				raw_data = reinterpret_cast<const int8_t*>(output_tensor.get_data(0));
			} else {
				std::cout << "***Output_tensor.dtype is not DT_INT8. It is index " 
				          << output_tensor.dtype 
				          << ", so use this index to find real data type in tensor.hpp.***" 
				          << std::endl << std::endl;
				//continue; // Skip to the next output_tensor if the dtype is not DT_INT8
			}

			// retrieves a pointer to the start of the data buffer for the first element (index 0) of the tensor.
			size_t output_size = (output_tensor.height * output_tensor.width * output_tensor.channel);
			std::vector<float> output_data(output_size);

			int fixpos = output_tensor.fixpos; // Number of fractional bits
			/*std::cout << "output_tensor's fixpos: " << fixpos << std::endl;
			
			for (int x = 0; x < output_size; x++) {
				
				cout << x << "RAW1: " << static_cast<int>(raw_data[x]) << " "; // use this pointer to access the data
			}
			cout << endl << endl;
			
			
			for (int x = 0; x < output_size; x++) {
				
				cout << x << "RAW2: " << static_cast<float>(raw_data[x]) << " "; // use this pointer to access the data
			}
			cout << endl << endl;
			*/
			
			for (size_t x = 0; x < output_size; ++x) {
				// This line converts the 8-bit fixed-point value (raw_data[i]) to a 32-bit floating-point
				output_data[x] = static_cast<float>(raw_data[x]) / static_cast<float>(1 << fixpos);
			}
			
			
			// do the softmax
			std::vector<float> probabilities = softmax(output_data);
			//cout << "probabilities_size: " << probabilities.size() << endl;
			//cout << "output_size: " << output_size << endl;
			
			// get the reault
			if(probabilities[0] > probabilities[1]) // real
			{
				real++;
				real_value += probabilities[0];
				cout << "Pic_" << i << "Image name:" << imgs_names[i] << " is Bonafide.\n";
			}
			else // fake
			{
				fake++;
				fake_value += probabilities[1];
				cout << "Pic_" << i << "Image name:" << imgs_names[i] << " is Spoof.\n";
			}
			// write result into file output.txt
			cout << "Below, Label 0 is bonafide. Label 1 is spoof.\n";
			out_fs << "Pic_" << i << "Image name:" << imgs_names[i] << " Raw data(float) = ";
			for (int x = 0; x < output_size; x++) {
				
				out_fs << x << ": " << static_cast<float>(raw_data[x]) << " "; // use this pointer to access the data
			}
			out_fs << endl << endl;
			
			out_fs << "Pic_" << i << "Image name:" << imgs_names[i] << " Raw data(before softmax) = ";
			for (int x = 0; x < output_size; x++) {
				
				out_fs << x << ": " << output_data[x] << " "; // use this pointer to access the data
			}
			out_fs << endl << endl;
			
			out_fs << "Pic_" << i << "Image name:" << imgs_names[i] << " Result data = ";
			cout << "Pic_" << i << "Image name:" << imgs_names[i] << " Result data = ";
			
			for (int x = 0; x < output_size; x++) {
				out_fs << x << ": " << probabilities[x] << " "; // use this pointer to access the data
				cout << x << ": " << probabilities[x] << " ";
			}
			out_fs << "\n--------------------------------------------------------------------------------------------------------------\n";
		}
		cout << "\n############ END ##################" << endl;
		inputs.clear();
		input_cols.clear();
		input_rows.clear();
		j = -1;
	}
    std::ofstream out_re("Result.txt", std::ofstream::out);
    // use value to compare when equalness happens
    if(fake == real)
    {
    	if(fake_value >= real_value)
    		fake++;
		else
			real++;
	}
	out_re << "fake: " << fake << " ;real: " << real;
	cout << "############ Prediction complete ##################" << endl;
	out_fs.close();
	out_re.close();
  	return 0;
}
