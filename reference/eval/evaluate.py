
from word_translation import*
import torch

class Evaluator(object):
    def __init__(self, emb0, emb1, emb0_vocablower2id, emb1_vocablower2id, src_lang, tgt_lang, dico_eval):
        self.src_emb = torch.tensor(emb0)
        self.tgt_emb = torch.tensor(emb1)
        self.src_dico = emb0_vocablower2id
        self.tgt_dico = emb1_vocablower2id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dico_eval = dico_eval

    def word_translation(self):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.src_emb
        tgt_emb = self.tgt_emb

        results = get_word_translation_accuracy(
            self.src_lang, self.src_dico, src_emb,
            self.tgt_lang, self.tgt_dico, tgt_emb,
            dico_eval=self.dico_eval)



        return results

