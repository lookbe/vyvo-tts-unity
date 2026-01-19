# VyvoTTS for Unity

A Unity 6 integration for **VyvoTTS**. VyvoTTS is a Text-to-Speech (TTS) training and inference framework built on top of the LLM model. This package is **experimental**, designed specifically to test local TTS capabilities on mobile devices.

---

## ⚠️ Platform Support
* **Platforms:** * **Windows** and **Android**

---

## Installation

Follow these steps exactly to ensure all native dependencies are resolved.

### Configure manifest.json
Open your project's `Packages/manifest.json` and update it to include the scoped registry and the Git dependencies.

```json
{
  "scopedRegistries": [
    {
      "name": "npm",
      "url": "[https://registry.npmjs.com](https://registry.npmjs.com)",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ],
  "dependencies": {
    "com.github.asus4.onnxruntime": "0.4.2",
    "com.github.asus4.onnxruntime.unity": "0.4.2",
    "ai.lookbe.llamacpp": "[https://github.com/lookbe/llama-cpp-unity.git](https://github.com/lookbe/llama-cpp-unity.git)",
    "ai.lookbe.vyvotts": "[https://github.com/lookbe/vyvo-tts-unity.git](https://github.com/lookbe/vyvo-tts-unity.git)",

    ... other dependencies
  }
}
```

---

## Requirements: Models

You must download the following two models separately:

1. **VyvoTTS Model (GGUF format):** Recommended version: [VyvoTTS-LFM2-350M-Jenny-i1-GGUF](https://huggingface.co/mradermacher/VyvoTTS-LFM2-350M-Jenny-i1-GGUF).
2. **SNAC Decoder (ONNX format):** Use the exact `decoder_model.onnx` file from [snac_24khz-ONNX](https://huggingface.co/onnx-community/snac_24khz-ONNX/tree/main/onnx).

---

## Testing

1. **Import Samples:** Go to the Package Manager, select **VyvoTTS Unity**, and import the **Basic Vyvo** sample.
2. **Configure Paths:**
    * Select the `VyvoTTS` object in the Hierarchy.
    * In the Inspector, locate the **Model Path** and **SNAC Model Path** fields.
    * **Important:** Paste the **absolute path** for both files. For Android testing, ensure the models are accessible via the persistent data path or scoped storage.
3. **Run:** Press Play. The system will initialize the Vulkan backend on your device.

> **Note:** This package is currently in an experimental state. It is provided specifically to test the feasibility and performance of local neural TTS on mobile hardware.

---


# Credits & License

* **[VyvoTTS](https://github.com/Vyvo-Labs/VyvoTTS)** Developed by **Vyvo Labs** – A TTS training and inference framework built on top of LLM architectures.
* **License:** The VyvoTTS models are distributed under **CC BY-NC 4.0**. This means they are for **Non-Commercial** use only.
* **[onnxruntime-unity](https://github.com/asus4/onnxruntime-unity)** Developed by **asus4** – ONNX Runtime integration for Unity.

---

## ☕ Support the Developer

If this helps you, consider supporting me:

[<img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" width="200">](https://www.buymeacoffee.com/lookbe)
