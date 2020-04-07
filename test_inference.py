import logging
import os
import tensorflow as tf
import network
import numpy as np

from config import load_arguments
from vocab import Vocabulary, build_unify_vocab

logger = logging.getLogger(__name__)

def create_model(sess, args, vocab):
  print('NNN', args.network)
  model = eval('network.' + args.network + '.Model')(args, vocab)
  if args.load_model:
    logger.info('-----Loading styler model from: %s.-----' % os.path.join(args.styler_path, 'model'))
    model.saver.restore(sess, os.path.join(args.styler_path, 'model').replace('\\', '/'))
  return model

if __name__ == '__main__':
  args = load_arguments()

  # sentence = "The food at the burger joint was disgusting and staff was impolite"
  # sentence = "The first thing I loved about this movie was that it was the most character driven film in the MCU till date"
  sentence = "Ever since joes has changed hands it's gotten better and better ."

  if not os.path.isfile(args.multi_vocab):
    build_unify_vocab([args.target_train_path, args.source_train_path], args.multi_vocab)
  multi_vocab = Vocabulary(args.multi_vocab)
  logger.info('vocabulary size: %d' % multi_vocab.size)

  print(multi_vocab.word2id('table'), multi_vocab.id2word(multi_vocab.word2id('table')))

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    # create style transfer model
    model = create_model(sess, args, multi_vocab)

    with_go = ['<go>'] + sentence.split()
    sentence_enc = [multi_vocab.word2id(w) for w in sentence.split()]
    sentence_dec = [multi_vocab.word2id(w) for w in with_go]
    
    sentence_enc_len = len(sentence_enc)
    n_missing_characters = max(20 - sentence_enc_len, 0)
    sentence_enc += [0 for i in range(0, n_missing_characters)]


    sentence_dec_len = len(sentence_dec)
    n_missing_characters = max(20 - sentence_dec_len, 0)
    sentence_dec += [0 for i in range(0, n_missing_characters)]

    # mask = [1 for i in range(0, sentence_enc_len)]
    # mask += [0 for i in range(0, n_missing_characters)]

    feed_dict = {
      model.dropout: 1.0,
      # model.sd_batch_len: 1,
      # model.sd_enc_inputs: np.array([sentence_enc]),
      # model.sd_dec_inputs: np.array([sentence_enc]),
      # model.sd_labels: np.array([1]),
      # model.sd_enc_lens: np.array([len(sentence_enc)]),
      # model.sd_targets: np.array([sentence_enc]),
      # model.sd_dec_mask: np.array([mask]),

      model.td_batch_len: 1,
      model.td_enc_inputs: np.array([sentence_enc]),
      model.td_dec_inputs: np.array([sentence_dec]),
      model.td_labels: np.array([1]),
      model.td_enc_lens: np.array([len(sentence_enc)]),
      # model.td_targets: np.array([sentence_enc]),
      # model.td_dec_mask: np.array([mask]),

      # model.td_batch_len: 1,
      # model.td_enc_inputs: np.array([sentence_enc]),
      # model.td_enc_lens: np.array([len(sentence_enc)]),
      # model.td_labels: np.array([1]),
    }

    to_return = {
      'tsf_ids': model.td_tsf_ids,
      'rec_ids': model.td_rec_ids,
    }

    # print('*****', feed_dict, to_return)

    result = sess.run(to_return, feed_dict)

    ids = result['tsf_ids'][0]
    out_sent_tsf = [multi_vocab.id2word(c) for c in ids]

    ids = result['rec_ids'][0]
    out_sent_rec = [multi_vocab.id2word(c) for c in ids]

    print(' '.join(out_sent_tsf))
    print(' '.join(out_sent_rec))
