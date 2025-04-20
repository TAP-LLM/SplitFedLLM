from multirc_eval_v1 import official_eval
import logging
import json

logging.basicConfig(filename="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/Official_eval/test.log",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO,
                    filemode='w')

# create a logger object
logger = logging.getLogger(__name__)
from multirc_measures import Measures

# this is the location of your data; has to be downloaded from http://cogcomp.org/multirc/
inputFile = '/home/zhangzishuai/SplitFederated-LLaMA/Dataset/MultiRC/val.json'

measures = Measures()
def official_eval(outFile,logger):
    input = json.load(open(inputFile))
    output = json.load(open(outFile))
    output_map = dict([[a["pid"] + "==" + a["qid"], a["scores"]] for a in output])

    assert len(output_map) == len(output), "You probably have redundancies in your keys"

    [P1, R1, F1m] = measures.per_question_metrics(input, output_map)
    logger.info("Per question measures (i.e. precision-recall per question, then average) ")
    logger.info("\tP: " + str(P1) + " - R: " + str(R1) + " - F1m: " + str(F1m))

    EM0 = measures.exact_match_metrics(input, output_map, 0)
    EM1 = measures.exact_match_metrics(input, output_map, 1)
    logger.info("\tEM0: " + str(EM0))
    logger.info("\tEM1: " + str(EM1))

    [P2, R2, F1a] = measures.per_dataset_metric(input, output_map)

    logger.info("Dataset-wide measures (i.e. precision-recall across all the candidate-answers in the dataset) ")
    logger.info("\tP: " + str(P2) + " - R: " + str(R2) + " - F1a: " + str(F1a))

prediction_save_path="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/MultiRC/version_8_r16_alpha32_lr0.00002/model-A/checkpoint-47000/Fl-llama_prediction_for_eval.json"
official_eval(prediction_save_path,logger)