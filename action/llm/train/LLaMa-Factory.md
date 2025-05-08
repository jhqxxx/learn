<!--
 * @Author: jhq
 * @Date: 2025-04-26 15:24:31
 * @LastEditTime: 2025-04-28 20:33:27
 * @Description: 
-->
LLaMa-Factory微调qwen,推理时一直循环生成，有问题
* lora-sft-qwen:
  torchrun $DISTRIBUTED_ARGS src/train.py     --stage sft     --do_train     --use_fast_tokenizer     --flash_attn fa2    --model_name_or_path "/mnt/c/jhq/huggingface_model/Qwen/Qwen2___5-1___5B"     --dataset nongye_sft_data    --template qwen     --finetuning_type lora     --lora_target q_proj,v_proj    --output_dir "/mnt/c/jhq/huggingface_model/Qwen/Qwen_lora_nongye"     --overwrite_cache     --overwrite_output_dir     --warmup_steps 100     --weight_decay 0.1     --per_device_train_batch_size 4     --gradient_accumulation_steps 4     --ddp_timeout 9000     --learning_rate 5e-6     --lr_scheduler_type cosine     --logging_steps 1     --cutoff_len 1024     --save_steps 1000     --plot_loss     --num_train_epochs 5     --bf16

* lora-合并：
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli export     --model_name_or_path "/mnt/c/jhq/huggingface_model/Qwen/Qwen2___5-1___5B"     --adapter_name_or_path "/mnt/c/jhq/huggingface_model/Qwen/Qwen_lora_nongye"     --template qwen     --finetuning_type lora     --export_dir "/mnt/c/jhq/huggingface_model/Qwen/Qwen_1_5B_nongye"     --export_size 2     --export_legacy_format False

