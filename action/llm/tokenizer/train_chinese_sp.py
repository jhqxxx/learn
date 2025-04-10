'''
Author: jhq
Date: 2025-04-09 14:06:14
LastEditTime: 2025-04-10 09:28:45
Description: 训练分词器
数据集使用huggingface的answer722qzd/baidubaike2
数据集质量不高，没有清洗过
下载：huggingface-cli download --repo-type dataset answer722qzd/baidubaike2 --local-dir ./answer722qzd/baidubaike2
'''

from datasets import load_dataset
import sentencepiece as spm
import os
import tqdm
import re

def parquet2txt(parquet_path, txt_path):
    parquet_list = os.listdir(parquet_path)
    for parquet in parquet_list:
        parquet_file = os.path.join(parquet_path, parquet)
        dataset = load_dataset("parquet", data_files={'train': parquet_file})
        txt_file = os.path.join(txt_path, parquet.split(".")[0]+".txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            for item in tqdm.tqdm(dataset['train']):
                txt = item["text"]
                endings_pattern = r'(?<![?!。？！])[?!。？！]'
                sentence_end_positions = [m.end() for m in re.finditer(endings_pattern, txt)]
                if txt and (txt[-1] not in "?!。？！"):
                    sentence_end_positions.append(len(txt))
                sentences = [txt[start:end] for start, end in zip([0]+sentence_end_positions[:-1], sentence_end_positions)]
                for sentence in sentences:
                    if sentence[-1] != "\n":
                        sentence += "\n"
                    f.write(sentence)

def train_sentencepiece(txt_path, vocab_size, output_dir="."):
    prefix = os.path.join(output_dir, "chinese_spm_{vocab_size}")
    text_files = [os.path.join(txt_path, file) for file in os.listdir(txt_path)]
    print("text_files:", text_files)
    
    print("will now train vocab...")
    spm.SentencePieceTrainer.train(
        input=text_files,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=0.9995,
        num_threads=(os.cpu_count()-2),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207",
        max_sentence_length=24000
    )
    print(f"trained tokenizer is in {prefix}.model")
    print("Done.")


# 由于sentencepiece输入为txt或tsv,需要处理数据集
parquet_path = r"D:\huggingface_dataset\answer722qzd\baidubaike2\data"
txt_path = r"D:\huggingface_dataset\answer722qzd\baidubaike2\txt"

if __name__ == "__main__":
    # parquet2txt(parquet_path, txt_path)
    output_dir = "./sp_output"
    os.makedirs(output_dir, exist_ok=True)
    vocab_size = 20000
    train_sentencepiece(txt_path, vocab_size, output_dir)