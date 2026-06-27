# PyTorch / LibTorch Setup Guide (Ubuntu / Lubuntu)

This guide installs all required dependencies for running the Java + C++ embedding engine using LibTorch and a TorchScript model.

---

# 1. Install system packages

Update repositories and install the required development tools:

```bash
sudo apt update

sudo apt install -y \
build-essential \
cmake \
git \
wget \
unzip \
curl \
python3-pip \
python3-venv
```

---

# 2. Create the project directory

```bash
mkdir -p ~/torch_demo
cd ~/torch_demo
```

---

# 3. Download LibTorch (CPU)

```bash
wget "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1%2Bcpu.zip" -O libtorch.zip

unzip libtorch.zip

rm libtorch.zip
```

Verify installation:

```bash
ls ~/torch_demo/libtorch
```

Expected output:

```
bin
include
lib
share
```

---

# 4. Create a Python virtual environment

Ubuntu 24.04+ blocks installing Python packages into the system environment.

Create a virtual environment:

```bash
cd ~/torch_demo

python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Your prompt should become:

```
(venv) user@computer:~/torch_demo$
```

---

# 5. Upgrade pip

```bash
pip install --upgrade pip
```

---

# 6. Install Python dependencies

```bash
pip install torch transformers sentence-transformers
```

These packages are only required for exporting or regenerating the TorchScript model (`model.pt`).

---

# 7. Export the TorchScript model

Run:

```bash
python export_model.py
```

This creates:

```
model.pt
```

---

# 8. Build the native C++ library

From the project root:

```bash
mkdir -p build

cd build

cmake ..

cmake --build . --config Release
```

This produces:

```
build/libsearch_engine.so
```

---

# 9. Configure LibTorch runtime libraries

Before running the Java application:

```bash
export LD_LIBRARY_PATH=$HOME/torch_demo/libtorch/lib:$LD_LIBRARY_PATH
```

To make this permanent:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/torch_demo/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
```

---

# 10. Java tokenizer

The Java application uses:

```java
HuggingFaceTokenizer.newInstance(
    "sentence-transformers/all-MiniLM-L6-v2"
);
```

No manual installation is required.

The tokenizer files are downloaded automatically on the first application run and cached locally.

---

# Project Structure

```
torch_demo/
│
├── build/
│   └── libsearch_engine.so
│
├── libtorch/
│
├── model.pt
│
├── search_engine.cpp
│
├── CMakeLists.txt
│
├── export_model.py
│
├── Main.java
│
└── venv/
```

---

# Notes

* `libtorch` is the native C++ runtime used for inference.
* `torch`, `transformers`, and `sentence-transformers` are only needed to generate or regenerate `model.pt`.
* Once `model.pt` has been created, the Java + C++ application runs using LibTorch and does not require the Python environment.
