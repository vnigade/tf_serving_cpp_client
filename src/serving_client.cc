/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================

Modifications copyright (C) 2018 Vitaly Bezgachev, vitaly.bezgachev@gmail.com

==============================================================================
*/

#include <fstream>
#include <iostream>

#include "generated/prediction_service.grpc.pb.h"
#include "generated/tensor.grpc.pb.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include <opencv2/opencv.hpp>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictionService;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

typedef google::protobuf::Map<std::string, tensorflow::TensorProto> OutMap;

#define TENSOR_DIM 5
#define FRAME_WIDTH 224
#define FRAME_HEIGHT 224
#define FRAME_CHANNEL 3
#define FRAME_NUM 64
#define BATCH_SIZE 1
#define MIN_SIZE 256.

const unsigned int tensor_shape[TENSOR_DIM] = {
    BATCH_SIZE, FRAME_NUM, FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNEL};

const cv::Size resize(int width, int height) {
  float factor = std::max<float>(MIN_SIZE / width, MIN_SIZE / height);
  width = std::floor(factor * width);
  height = std::floor(height * factor);
  width -= width % 2;
  height -= height % 2;
  return cv::Size(width, height);
}
std::vector<cv::Mat> readFrames(std::string video_path, int frame_count) {
  std::vector<cv::Mat> frames(frame_count);

  for (int i = 0; i < frame_count; ++i) {
    std::stringstream filepath;
    filepath << video_path << "/frame" << i << ".jpg";
    cv::Mat sample = cv::imread(filepath.str(), cv::IMREAD_COLOR);
    // cv::cvtColor(sample, sample, CV_BGR2RGB);
    cv::Size re_size = resize(sample.cols, sample.rows);
    cv::resize(sample, sample, re_size, 0.0, 0.0, cv::INTER_LINEAR);
    sample.convertTo(frames[i], CV_32F);
    cv::Scalar mean = cv::mean(frames[i]);
    frames[i] = frames[i] - mean;
    frames[i] = frames[i] / 255.0;
    cv::Size size(FRAME_WIDTH, FRAME_HEIGHT);
    cv::Rect crop(cv::Point(0.5 * (frames[i].cols - FRAME_WIDTH),
                            0.5 * (frames[i].rows - FRAME_HEIGHT)),
                  size);
    frames[i] = frames[i](crop);
    cv::imshow("", frames[i]);
    cv::waitKey(30);
  }
  return frames;
}

void convertToTensorProto(std::vector<cv::Mat> &frames,
                          tensorflow::TensorProto &proto) {
  // tensorflow::TensorProto proto;
  const size_t frame_data_size =
      BATCH_SIZE * FRAME_NUM * FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNEL;
  uchar *const frame_data = new uchar[frame_data_size * sizeof(float)];
  uchar *dst = frame_data;
  for (auto frame : frames) {
    if (frame.isContinuous()) {
      std::cerr << "Frame is not continous " << std::endl;
    }
    uchar *src = frame.data;
    int size = frame.size().width * frame.size().height * frame.channels();
    // proto.add_string_val(data, size * sizeof(float));
    memcpy(dst, src, size * sizeof(float));
    dst += size;
  }

  proto.set_tensor_content(std::string(reinterpret_cast<char *>(frame_data),
                                       frame_data_size * sizeof(float)));
  proto.set_dtype(tensorflow::DT_FLOAT);
  for (int i = 0; i < TENSOR_DIM; ++i) {
    proto.mutable_tensor_shape()->add_dim()->set_size(tensor_shape[i]);
  }
  delete[] frame_data;
  return;
}

/*
Serving client for the prediction service
*/
class ServingClient {

private:
  std::unique_ptr<PredictionService::Stub> stub_;

public:
  // Constructor: create a stub for the prediction service
  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

  // Call the prediction service
  std::string callPredict(const std::string &model_name,
                          const std::string &model_signature_name,
                          const std::string &video_path) {
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    // set model specification: name and signature name
    predictRequest.mutable_model_spec()->set_name(model_name);
    // predictRequest.mutable_model_spec()->set_signature_name(
    //     model_signature_name);

    std::vector<cv::Mat> frames = readFrames(video_path, FRAME_NUM);
    tensorflow::TensorProto proto;
    convertToTensorProto(frames, proto);

    // initialize prediction service inputs
    google::protobuf::Map<std::string, tensorflow::TensorProto> &inputs =
        *predictRequest.mutable_inputs();
    inputs["rgb_input"] = proto;

    // issue gRPC call to the service
    Status status = stub_->Predict(&context, predictRequest, &response);

    // check the response
    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is " << response.outputs_size() << std::endl;
      OutMap &map_outputs = *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;

      // read the response
      for (iter = map_outputs.begin(); iter != map_outputs.end(); ++iter) {
        tensorflow::TensorProto &result_tensor_proto = iter->second;
        std::cout << "number of probabilies "
                  << result_tensor_proto.float_val_size() << std::endl;

        int maxIdx = -1;
        float maxVal = -1;
        for (int i = 0; i < result_tensor_proto.float_val_size(); ++i) {
          float val = result_tensor_proto.float_val(i);
          std::cout << "probability of " << i << " is " << val << std::endl;

          if (maxVal < val) {
            maxVal = val;
            maxIdx = i;
          }
        }

        std::cout << std::endl
                  << "most probably the digit on the image is " << maxIdx
                  << std::endl
                  << std::endl;

        ++output_index;
      }

      return "Done.";
    } else {
      std::cout << "gRPC call return code: " << status.error_code() << ": "
                << status.error_message() << std::endl;
      return "gRPC failed.";
    }
  }
};

/*
Application entry point
*/
int main(int argc, char **argv) {
  const std::string model_name = "inception_i3d";
  const std::string model_signature_name = "predict_images";

  std::string server = "localhost:8500";
  std::string video_path = "/home/vinod/action_stream/server/test/smoking/";

  std::cout << "calling prediction service on " << server << std::endl;

  // create and call serving client
  ServingClient guide(
      grpc::CreateChannel(server, grpc::InsecureChannelCredentials()));
  std::cout << "calling predict for video path " << video_path << "  ..."
            << std::endl;
  std::cout << guide.callPredict(model_name, model_signature_name, video_path)
            << std::endl;
  return 0;
}
