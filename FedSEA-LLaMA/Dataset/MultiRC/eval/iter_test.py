def take_first_16_jsonl(input_file, output_file):
    """
    从 input_file 中读取前 18 行（JSONL 格式），
    并将它们写入 output_file（仍保持 JSONL 格式）。
    """
    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if count < 18:
                fout.write(line)
                count += 1
            else:
                break

if __name__ == "__main__":
    input_file = "SplitFederated-LLaMA/Dataset/MultiRC/val.jsonl"
    output_file = "SplitFederated-LLaMA/Dataset/MultiRC/sample_val.jsonl"
    take_first_16_jsonl(input_file, output_file)
    print(f"已将前 16 行写入 {output_file}")
