# Architectural Technical Report: Seamless Java 25 & PyTorch Integration via Project Panama

## Project Objective
The primary goal of this research and development project is to demonstrate high-performance, low-level interoperability between modern Java 25 and native PyTorch (LibTorch C++) without using slow legacy JNI layouts or heavy automated third-party frameworks. By utilizing Project Panama (Foreign Function & Memory API), we achieve safe, ultra-fast off-heap memory allocation to pass unstructured text from the JVM stack into a native C++ module, execute neural-like structural vector transformation via PyTorch, and calculate structural semantic similarity natively.

## Environment Specification
- **Operating System:** Lubuntu Linux (x86_64)
- **Java Platform:** JDK 25.0.3 (GraalVM Distribution via SDKMAN)
- **Native Platform:** LibTorch 2.5.1 (Official Standalone C++ Distribution from PyTorch, CPU-only architecture)
- **Interoperability Layer:** Project Panama (java.lang.foreign ecosystem)

---

## Step-by-Step Implementation Guide

### Step 1: Toolchain and Environment Setup
First, update your system repositories and install the essential C++ compilers and build tools:

```bash
sudo apt update && sudo apt install -y build-essential cmake unzip wget
```

Once the installation is complete, create an isolated directory for the PyTorch core layout and navigate into it:

```bash
mkdir -p ~/torch_demo && cd ~/torch_demo
```


### Step 2: Downloading the PyTorch Native Core (LibTorch)
Download the verified standalone CPU core binary layout directly using clean quotes to prevent any shell layout variations:

```bash
cd ~/torch_demo
rm -f libtorch.zip

# Download the real 250MB LibTorch archive directly
wget "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1%2Bcpu.zip" -O libtorch.zip

# Extract the core files and clean up the archive
unzip libtorch.zip && rm libtorch.zip

# Verify that the folder exists
ls -lh ~/torch_demo/libtorch
```

### Step 3: Native C++ Bridge Code
Create a file named `search_engine.cpp` inside your `~/torch_demo` folder and paste the following source code into it. This file is simulation, real one in folder "with model":

```cpp
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

bool is_model_loaded = false;

const std::vector<std::string> vocabulary = {
    "water", "drink", "cola", "coca", "mineral", "sparkling",
    "smartphone", "apple", "iphone", "pro", "max",
    "winter", "jacket", "hooded", "down", "men",
    "fresh", "croissant", "chocolate", "filling"
};

std::string to_lower(std::string data) {
    std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c){ return std::tolower(c); });
    return data;
}

extern "C" {
    bool load_model(const char* model_path) {
        std::ifstream file(model_path);
        if (file.good()) {
            is_model_loaded = true;
            return true;
        }
        return false;
    }

    void get_embedding(const char* text, float* output_vector) {
        if (!is_model_loaded) return;
        
        std::string s = to_lower(std::string(text));
        std::vector<float> features(384, 0.0f);
        
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            if (s.find(vocabulary[i]) != std::string::npos) {
                features.at(i) = 1.0f;
            }
        }
        
        if (s.find("want") != std::string::npos && s.find("drink") != std::string::npos) {
            std::vector<float> context_weights(384, 0.0f);
            context_weights.at(0) = 2.5f; // water
            context_weights.at(1) = 2.5f; // drink
            context_weights.at(2) = 2.0f; // cola
            context_weights.at(3) = 2.0f; // coca
            context_weights.at(4) = 2.0f; // mineral
            
            torch::Tensor context_tensor = torch::tensor(context_weights);
            torch::Tensor embedding = torch::tensor(features) + context_tensor;
            
            torch::manual_seed(42);
            embedding = embedding + torch::randn({384}) * 0.01;
            embedding = embedding / embedding.norm();
            std::memcpy(output_vector, embedding.data_ptr<float>(), 384 * sizeof(float));
            return;
        }

        torch::Tensor embedding = torch::tensor(features);
        torch::manual_seed(42);
        embedding = embedding + torch::randn({384}) * 0.01;
        embedding = embedding / embedding.norm();
        std::memcpy(output_vector, embedding.data_ptr<float>(), 384 * sizeof(float));
    }
}
```

### Step 4: Compiling the Native Shared Object (.so)
Create a file named `CMakeLists.txt` inside your `~/torch_demo` folder and paste these compilation rules:

```cmake
cmake_minimum_required(VERSION 3.10)
project(search_engine)

set(CMAKE_PREFIX_PATH "\${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "\({CMAKE_CXX_FLAGS}\){TORCH_CXX_FLAGS}")

add_library(search_engine SHARED search_engine.cpp)
target_link_libraries(search_engine "\${TORCH_LIBRARIES}")
set_target_properties(search_engine PROPERTIES CXX_STANDARD 17)
```

Now execute the complete compilation sequence in your terminal to build the native library package:

```bash
cd ~/torch_demo
mkdir -p build
cd build
cmake ..
make
```

### Step 5: Run Configurations (IntelliJ IDEA VM Options)
To allow foreign memory preview access allocations and dynamically point JVM runtime components to the target LibTorch link paths, append this statement string into your IntelliJ IDEA **VM Options** field:

```bash
--enable-preview --enable-native-access=ALL-UNNAMED -Djava.library.path=/home/admin/torch_demo/libtorch/lib
```
