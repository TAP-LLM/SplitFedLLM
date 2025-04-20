### Evaluation script used for evaluation of baselines for MultiRC dataset
# The evaluation script expects the questions, and predicted answers from separate json files.
# The predicted answers should be 1s and 0s (no real-valued scores)

import json

from Official_eval.multirc_measures import Measures
# from multirc_measures import Measures

# this is the location of your data; has to be downloaded from http://cogcomp.org/multirc/
# inputFile = '/home/zhangzishuai/SplitFederated-LLaMA/Dataset/MultiRC/sample_val.json'
inputFile = '/home/zhangzishuai/SplitFederated-LLaMA/Dataset/MultiRC/val.json'


measures = Measures()

def main():
    eval('baseline-scores/human-01.json')
    # eval('baseline-scores/allOnes.json')
    # eval('baseline-scores/allZeros.json')
    # eval('baseline-scores/simpleLR.json')
    # eval('baseline-scores/lucene_world.json')
    # eval('baseline-scores/lucene_paragraphs.json')

# the input to the `eval` function is the file which contains the binary predictions per question-id
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

# input["data"]
if __name__ == "__main__":
    main()
