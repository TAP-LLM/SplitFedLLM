import math
from functools import reduce


class Measures:

    @staticmethod
    def per_question_metrics(dataset, output_map):
        P = []
        R = []
        for p in dataset:
            for q in p["passage"]["questions"]:
                qIdx=q["idx"]
                id = str(p["idx"]) + "==" + str(qIdx)
                if (id in output_map):
                    predictedAns = output_map.get(id)
                    correctAns = [int(a["label"]) for a in q["answers"]]
                    predictCount = sum(predictedAns)
                    correctCount = sum(correctAns)
                    assert math.ceil(sum(predictedAns)) == sum(predictedAns), "sum of the scores: " + str(sum(predictedAns))
                    agreementCount = sum([a * b for (a, b) in zip(correctAns, predictedAns)])
                    p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
                    r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
                    P.append(p1)
                    R.append(r1)
                else:
                    print("The id " + id + " not found . . . ")

        pAvg = Measures.avg(P)
        rAvg = Measures.avg(R)
        f1Avg = 2 * Measures.avg(R) * Measures.avg(P) / (Measures.avg(P) + Measures.avg(R))
        return [pAvg, rAvg, f1Avg]

    @staticmethod
    def exact_match_metrics(dataset, output_map, delta):
        EM = []
        for p in dataset:
            for q in p["passage"]["questions"]:
                qIdx=q["idx"]
                id = str(p["idx"]) + "==" + str(qIdx)
                if (id in output_map):
                    predictedAns = output_map.get(id)
                    correctAns = [int(a["label"]) for a in q["answers"]]
                    em = 1.0 if sum([abs(i - j) for i, j in zip(correctAns, predictedAns)]) <= delta  else 0.0
                    EM.append(em)
                else:
                    print("The id " + id + " not found . . . ")

        return Measures.avg(EM)

    @staticmethod
    def per_dataset_metric(dataset, output_map):
        agreementCount = 0
        correctCount = 0
        predictCount = 0
        for p in dataset:
            for q in p["passage"]["questions"]:
                qIdx=q["idx"]
                id = str(p["idx"]) + "==" + str(qIdx)
                if (id in output_map):
                    predictedAns = output_map.get(id)
                    correctAns = [int(a["label"]) for a in q["answers"]]
                    predictCount += sum(predictedAns)
                    correctCount += sum(correctAns)
                    agreementCount += sum([a * b for (a, b) in zip(correctAns, predictedAns)])
                else:
                    print("The id " + id + " not found . . . ")

        p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        return [p1, r1, 2 * r1 * p1 / (p1 + r1)]

    @staticmethod
    def avg(l):
        return reduce(lambda x, y: x + y, l) / len(l)
