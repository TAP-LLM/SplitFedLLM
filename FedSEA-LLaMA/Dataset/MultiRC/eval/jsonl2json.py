import json

def jsonl_to_json(input_file, output_file):
    data = []
    # 打开 JSONL 文件，按行读取
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # strip 去除空格或换行，忽略空行
            line = line.strip()
            if not line:
                continue
            # 将每行内容解析为 JSON 对象并添加到列表
            data.append(json.loads(line))

    # 将列表写入标准 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 示例用法
    input_file_path = "SplitFederated-LLaMA/Dataset/MultiRC/sample_val.jsonl"
    output_file_path = "SplitFederated-LLaMA/Dataset/MultiRC/sample_val.json"
    jsonl_to_json(input_file_path, output_file_path)
    print(f"转换完成，结果已写入：{output_file_path}")


# 变成json文件主要是为了计算指标的时候需要指定json文件