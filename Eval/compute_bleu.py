from nltk.translate.bleu_score import sentence_bleu

def compute_bleu_scores(reference, hypothesis):
    """
    Computes BLEU score given hypothesis and reference.
    :param reference: list of strings, reference sentences
    :param hypothesis: list of strings, candidate sentences
    :return: float, BLEU_1, BLEU_2, BLEU_3, BLEU_4 scores
    https://towardsdatascience.com/bleu-bilingual-evaluation-understudy-2b4eab9bcfd1
    """
    bleu_1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_1, bleu_2, bleu_3, bleu_4
