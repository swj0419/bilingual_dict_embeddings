
from word_translation import*


class Evaluator(object):
    def __init__(self, emb0, emb1):
        self.src_emb = emb0.get_emb()
        self.tgt_emb = emb1.get_emb()
        self.src_dico = emb0.vocablower2id
        self.tgt_dico = emb1.vocablower2id
        self.src_lang = "en"
        self.tgt_lang = "fr"
        self.dico_eval = "default"



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

        print("word translation", results) #log

