* to gguf
    python convert-hf-to-gguf.py Qwen/Qwen2.5-7B-Instruct --outfile Qwen2.5-7b-instruct-f16.gguf
* llama.cpp 推理
    ./llama-cli -m /mnt/c/jhq/huggingface_model/Qwen/Qwen_1_5B_nongye-f32.gguf -co -cnv -p "你是农业小助手"  -fa -ngl 80 -n 512