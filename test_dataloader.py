import logging

import os
import tensorflow as tf
import time
from config import load_arguments
from vocab import Vocabulary, build_unify_vocab
from dataloader.multi_style_dataloader import MultiStyleDataloader
from utils import *
import network

logger = logging.getLogger(__name__)

def as_sentence(array, vocab):
  return ' '.join([vocab.id2word(w) for w in array])

def print_batch(bs, bt, vocab):
  for i in range(0, bs.enc_batch.size-1):
    print('')
    print('---')

    for label, b in [['S', bs], ['T', bt]]:
      print('  *** ', label)
      print('enc_batch: ', as_sentence(b.enc_batch[i], vocab))
      print('labels: ', b.labels[i])
      # print('enc_lens: ', b.enc_lens[i])
      print('dec_batch: ', as_sentence(b.dec_batch[i], vocab))
      print('target_batch: ', as_sentence(b.target_batch[i], vocab))
      # print('dec_padding_mask: ', b.dec_padding_mask[i])

def create_model(sess, args, vocab):
  print('NNN', args.network)
  model = eval('network.' + args.network + '.Model')(args, vocab)
  print('S', model.saver)
  if args.load_model:
    logger.info('-----Loading styler model from: %s.-----' % os.path.join(args.styler_path, 'model'))
    model.saver.restore(sess, os.path.join(args.styler_path, 'model'))
  else:
    logger.info('-----Creating styler model with fresh parameters.-----')
    sess.run(tf.global_variables_initializer())
  if not os.path.exists(args.styler_path):
    os.makedirs(args.styler_path)
  return model

def train(loader, multi_vocab):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  gamma = args.gamma_init

  # use tensorboard
  if args.suffix:
      tensorboard_dir = os.path.join(args.logDir, 'tensorboard', args.suffix)
  else:
      tensorboard_dir = os.path.join(args.logDir, 'tensorboard')
  if not os.path.exists(tensorboard_dir):
      os.makedirs(tensorboard_dir)

  write_dict = {
    'writer': tf.summary.FileWriter(logdir=tensorboard_dir, filename_suffix=args.suffix),
    'step': 0
  }

  with tf.Session(config=config) as sess:
    # create style transfer model
    model = create_model(sess, args, multi_vocab)

    start_time = time.time()
    step = 0
    accumulator = Accumulator(args.train_checkpoint_step, model.get_output_names('all'))
    learning_rate = args.learning_rate

    source_batches = loader.get_batches(domain='source', mode='train')
    target_batches = loader.get_batches(domain='target', mode='train')
    test_batches = loader.get_batches(domain='target', mode='test')

    for epoch in range(1, 1+args.max_epochs):
      logger.info('--------------------epoch %d--------------------' % epoch)
      logger.info('learning_rate: %.4f  gamma: %.4f' % (learning_rate, gamma))

      # multi dataset training
      source_len = len(source_batches)
      target_len = len(target_batches)
      iter_len = max(source_len, target_len)
      for i in range(iter_len):
        model.run_train_step(sess, 
          target_batches[i % target_len], source_batches[i % source_len], accumulator, epoch)
        step += 1
        write_dict['step'] = step

        if step % args.train_checkpoint_step == 0:
          accumulator.output('step %d, time %.0fs,'
              % (step, time.time() - start_time), write_dict, 'train')
          accumulator.clear()
          test_accumulator = Accumulator(len(test_batches), model.get_output_names('target'))
          
          for idx, test_batch in enumerate(test_batches):
            results = model.run_eval_step(sess, test_batch, 'target')
            test_accumulator.add([results[name] for name in test_accumulator.names])

            rec = [[multi_vocab.id2word(i) for i in sent] for sent in results['rec_ids']]
            rec, _ = strip_eos(rec)

            tsf = [[multi_vocab.id2word(i) for i in sent] for sent in results['tsf_ids']]
            tsf, _ = strip_eos(tsf)

            enc_inputs = [[multi_vocab.id2word(i) for i in sent] for sent in results['enc_inputs']]
            enc_inputs, _ = strip_eos(enc_inputs)
            enc_inputs = [[w for w in s if w != '<pad>'] for s in enc_inputs]

            t = [[multi_vocab.id2word(i) for i in sent] for sent in results['targets']]
            t, _ = strip_eos(t)

            def as_sent(x):
              return [' '.join(i) for i in x]

            if idx == len(test_batches) - 1:
              print('---')
              print('REC: ', as_sent(rec[:5]))
              print('TSF: ', as_sent(tsf[:5]))
              print('EI: ', as_sent(enc_inputs[:5]))
              print('T: ', as_sent(t[:5]))

              test_accumulator.output('valid', write_dict, 'valid')

          if args.save_model:
            logger.info('Saving style transfer model...')
            model.saver.save(sess, os.path.join(args.styler_path, 'model'))


if __name__ == '__main__':
  args = load_arguments()

  if not os.path.isfile(args.multi_vocab):
      build_unify_vocab([args.target_train_path, args.source_train_path], args.multi_vocab)
  multi_vocab = Vocabulary(args.multi_vocab)
  logger.info('vocabulary size: %d' % multi_vocab.size)

  print(multi_vocab.word2id('table'), multi_vocab.id2word(multi_vocab.word2id('table')))

  loader = MultiStyleDataloader(args, multi_vocab)

  train(loader, multi_vocab)
  # source_batches = loader.get_batches(domain='source', mode='train')
  # target_batches = loader.get_batches(domain='target', mode='train')

  # sb = source_batches[0]
  # tb = target_batches[0]

  # print_batch(sb, tb, multi_vocab)
  