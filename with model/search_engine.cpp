#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

static torch::jit::Module module;
static bool model_loaded = false;

extern "C" {

/**
 * Load TorchScript model
 */
bool load_model(const char* model_path) {
    try {
        module = torch::jit::load(model_path);
        module.eval();
        model_loaded = true;

        std::cout << "Model loaded: " << model_path << std::endl;
        return true;

    } catch (const c10::Error& e) {
        std::cerr << "Load failed: " << e.what() << std::endl;
        model_loaded = false;
        return false;
    }
}

/**
 * get_embedding
 */
void get_embedding(
    const int64_t* input_ids,
    const int64_t* attention_mask,
    int64_t length,
    float* out_vector
) {
    if (!model_loaded) {
        std::cerr << "Model not loaded" << std::endl;
        return;
    }

    try {
        // -----------------------------
        // Convert Java memory → Tensor
        // -----------------------------
        torch::Tensor ids = torch::from_blob(
            (void*)input_ids,
            {1, length},
            torch::kInt64
        ).clone();

        torch::Tensor mask = torch::from_blob(
            (void*)attention_mask,
            {1, length},
            torch::kInt64
        ).clone();

        // -----------------------------
        // Run model
        // -----------------------------
        std::vector<torch::jit::IValue> inputs = {ids, mask};

        torch::jit::IValue model_output = module.forward(inputs);

        // -----------------------------
        // Extract tensor safely
        // -----------------------------
        torch::Tensor hidden;

        if (model_output.isTensor()) {
            hidden = model_output.toTensor();
        }
        else if (model_output.isGenericDict()) {
            auto dict = model_output.toGenericDict();
            hidden = dict.at("last_hidden_state").toTensor();
        }
        else if (model_output.isTuple()) {
            auto tuple = model_output.toTuple();
            hidden = tuple->elements()[0].toTensor();
        }
        else {
            std::cerr << "Unknown model output type" << std::endl;
            return;
        }

        // -----------------------------
        // Mean pooling (ignore padding)
        // -----------------------------
        torch::Tensor mask_f = mask.unsqueeze(-1).to(torch::kFloat32);

        torch::Tensor summed = (hidden * mask_f).sum(1);
        torch::Tensor counts = mask_f.sum(1).clamp_min(1e-9);

        torch::Tensor emb = summed / counts;

        // -----------------------------
        // Normalize
        // -----------------------------
        emb = torch::nn::functional::normalize(
            emb,
            torch::nn::functional::NormalizeFuncOptions().p(2)
        );

        emb = emb.squeeze(0);

        // -----------------------------
        // Copy to Java
        // -----------------------------
        std::memcpy(
            out_vector,
            emb.data_ptr<float>(),
            384 * sizeof(float)
        );

    } catch (const c10::Error& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
    }
}

} // extern "C"
