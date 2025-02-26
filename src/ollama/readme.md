# Model conversion

1. Merge model with lora weights if you haven't done it yet.  (https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.merge_and_unload)

2. Clone llama

```git clone https://github.com/ggerganov/llama.cpp```

3. Convert to gguf format

```python llama.cpp/convert-hf-to-gguf.py model_merged/ --outfile model_name.gguf```

4. Create a Makefile that is based on the basic model.
```ollama show --modelfile phi3:3.8b >>  Modelfile```

To use your local model, replace FROM with your model's path.

5. Create model 
```ollama create phi3-5-local -f Modelfile```

6. Check if model appeared
```ollama list```