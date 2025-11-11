import json

def convert_jsonl_to_official_format(input_jsonl_file, output_json_file):
    """
    将验证集从 JSON Lines (NDJSON) 格式转为官方评测脚本需要的格式。
    每行数据(例: item) 结构大致如下:
        {
          "idx": 0,
          "passage": {
              "text": "Some text",
              "questions": [
                  {
                      "idx": 0,
                      "question": "...",
                      "answers": [
                          {"text": "...", "idx": 0, "label": 1},
                          ...
                      ]
                  },
                  ...
              ]
          }
        }

    转换输出: 一个JSON数组, 其中每个元素为:
        {
          "pid": str(item["idx"]),
          "qid": str(q["idx"]),
          "scores": [ans["label"], ...]
        }
    """
    results = []
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 逐行解析 JSON
            item = json.loads(line)

            # 顶层 "idx" 作为 "pid"
            passage_id = str(item["idx"])
            questions = item["passage"]["questions"]

            for q in questions:
                question_id = str(q["idx"])
                # 取出所有答案的 label, 组成 scores 数组
                scores = [ans["label"] for ans in q["answers"]]

                results.append({
                    "pid": passage_id,
                    "qid": question_id,
                    "scores": scores
                })

    # 将结果写入到一个 JSON 数组文件中
    with open(output_json_file, 'w', encoding='utf-8') as fo:
        json.dump(results, fo, ensure_ascii=False, indent=2)

    print(f"转换完成！共生成 {len(results)} 条记录，输出文件: {output_json_file}")


if __name__ == "__main__":
    # 假设你的验证集 JSON 文件名为 dev_set.json
    input_file = "SplitFederated-LLaMA/Dataset/MultiRC/val.jsonl"
    # 想输出成 official_format.json
    output_file = "SplitFederated-LLaMA/Dataset/MultiRC/fit_val.jsonl"
    convert_jsonl_to_official_format(input_file, output_file)

