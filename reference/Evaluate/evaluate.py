class Evaluator(object):
    def __init__(self, emb0, emb1):
        self.src_emb = emb0.get_emb()
        self.tgt_emb = emb1.get_emb()
        self.src_dico = emb0.vocablower2id
        self.tgt_dico = emb1.vocablower2id


   def word_translation(self, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data


        results = get_word_translation_accuracy(
            self.src_dico.lang, self.src_dico.word2id, src_emb,
            self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
            method=method,
            dico_eval=self.params.dico_eval

            to_log.update([('%s-%s' % (k, method), v) for k, v in results])
