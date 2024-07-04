import os
from transformers import AutoModel
import torch
import torch.nn as nn
import argparse

def split_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glm_path",type=str, default = None)
    parser.add_argument("--save_path",type=str, default = None)
    args = parser.parse_args()
    return args
    

def main(args):
    model = AutoModel.from_pretrained(args.glm_path, trust_remote_code=True).half().cuda()

    model1 = model.transformer.word_embeddings
    model2 = model.transformer.layers[0]
    model3 = nn.ModuleList([model1,model2])

    model4 = model.transformer.layers[1:27]

    model5 = model.transformer.layers[27]
    model6 = model.transformer.final_layernorm
    model7 = model.lm_head
    model8 = nn.MoudlueList([model5,model6,model7])

    torch.save(model3.state_dict(), os.path.join(args.save_path, 'client_model_partA_param.bin'))
    torch.save(model4.state_dict(), os.path.join(args.save_path, 'server_model.bin'))
    torch.save(model8.state_dict(), os.path.join(args.save_path, 'client_model_partC_param.bin'))


if __name__ == "__main__":
    args = split_args()
    main(args)
