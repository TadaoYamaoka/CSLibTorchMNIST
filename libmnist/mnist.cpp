#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C"
{
  DLL_EXPORT void init();
  DLL_EXPORT void train(float *dataset, int64_t *targetset, int dataset_size);
  DLL_EXPORT void test(float* dataset, int64_t* targetset, int dataset_size);
}

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

struct Net : torch::nn::Module
{
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10)
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

std::unique_ptr<Net> model;
std::unique_ptr<torch::Device> device;
std::unique_ptr<torch::optim::Optimizer> optimizer;

DLL_EXPORT
void init()
{
  torch::DeviceType device_type;
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  }
  else
  {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  device.reset(new torch::Device(device_type));

  model.reset(new Net());
  model->to(*device);

  optimizer.reset(new torch::optim::SGD(
      model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5)));
}

DLL_EXPORT
void train(
    float *dataset,
    int64_t *targetset,
	int dataset_size) {
  model->train();
  size_t batch_idx = 0;
  for (size_t i = 0; i <= dataset_size - kTrainBatchSize; i += kTrainBatchSize)
  {
    /*std::cout << "batch " << i << std::endl;
    for (int y = 0; y < 28; ++y) {
      for (int x = 0; x < 28; ++x) {
        std::cout << (dataset + i * 28 * 28)[y * 28 + x] << ",";
      }
      std::cout << std::endl;
    }
    std::cout << "label " << targetset[i] << std::endl;*/

    auto data = torch::from_blob(dataset + i * 28 * 28, {kTrainBatchSize, 1, 28, 28}, torch::dtype(torch::kFloat32)).to(*device);
    auto targets = torch::from_blob(targetset + i, {kTrainBatchSize}, torch::dtype(torch::kInt64)).to(*device);
    optimizer->zero_grad();
    auto output = model->forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer->step();

    if (batch_idx++ % kLogInterval == 0)
    {
      std::printf(
          "\rTrain [%5ld/%5ld] Loss: %.4f",
          batch_idx * kTrainBatchSize,
          dataset_size,
          loss.template item<float>());
    }
  }
}

DLL_EXPORT
void test(
    float *dataset,
    int64_t *targetset,
	int dataset_size) {
  torch::NoGradGuard no_grad;
  model->eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (size_t i = 0; i <= dataset_size - kTestBatchSize; i += kTestBatchSize)
  {
    auto data = torch::from_blob(dataset + i * 28 * 28, {kTestBatchSize, 1, 28, 28}, torch::dtype(torch::kFloat32)).to(*device);
    auto targets = torch::from_blob(targetset + i, {kTestBatchSize}, torch::dtype(torch::kInt64)).to(*device);
    auto output = model->forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

// for test
void main()
{
  int dataset_size = 2;
  float* dataset = new float[kTestBatchSize * 28 * 28 * dataset_size];
  int64_t* targetset = new int64_t[kTestBatchSize * dataset_size];

  std::fill_n(dataset, kTestBatchSize * 28 * 28 * dataset_size, 0);
  std::fill_n(targetset, kTestBatchSize * dataset_size, 0);

  init();
  test(dataset, targetset, dataset_size);
}