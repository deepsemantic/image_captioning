#!/usr/bin/env python

from collections import OrderedDict
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import pdb

sys.path.append('./python/')
import caffe

class Captioner():
  def __init__(self, weights_path, image_net_proto, lstm_net_proto,
               vocab_path, device_id=-1):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    # Setup image processing net.
    phase = caffe.TEST
    self.image_net = caffe.Net(image_net_proto, weights_path, phase)
    image_data_shape = self.image_net.blobs['data'].data.shape
    self.transformer = caffe.io.Transformer({'data': image_data_shape})
    #print image_data_shape
    #pdb.set_trace()
    channel_mean = np.zeros(image_data_shape[1:])
    channel_mean_values = [104, 117, 123]
    assert channel_mean.shape[0] == len(channel_mean_values)
    for channel_index, mean_val in enumerate(channel_mean_values):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data', channel_mean)
    self.transformer.set_channel_swap('data', (2, 1, 0))
    self.transformer.set_transpose('data', (2, 0, 1))
    # Setup sentence prediction net.
    self.lstm_net = caffe.Net(lstm_net_proto, weights_path, phase)
    self.vocab = ['<EOS>']
    with open(vocab_path, 'r') as vocab_file:
      self.vocab += [word.strip() for word in vocab_file.readlines()]
    net_vocab_size = self.lstm_net.blobs['predict_forward'].data.shape[2]
    if len(self.vocab) != net_vocab_size:
      raise Exception('Invalid vocab file: contains %d words; '
          'net expects vocab with %d words' % (len(self.vocab), net_vocab_size))

  def set_image_batch_size(self, batch_size):
    self.image_net.blobs['data'].reshape(batch_size,
        *self.image_net.blobs['data'].data.shape[1:])

  def caption_batch_size(self):
    return self.lstm_net.blobs['cont_sentence_forward'].data.shape[1]

  def set_caption_batch_size(self, batch_size):
    self.lstm_net.blobs['cont_sentence_forward'].reshape(1, batch_size)
    self.lstm_net.blobs['input_sentence_forward'].reshape(1, batch_size)
    self.lstm_net.blobs['cont_sentence_backward'].reshape(1, batch_size)
    self.lstm_net.blobs['input_sentence_backward'].reshape(1, batch_size)
    self.lstm_net.blobs['image_features'].reshape(batch_size,
        *self.lstm_net.blobs['image_features'].data.shape[1:])
    self.lstm_net.reshape()

  def preprocess_image(self, image, verbose=False):
    #pdb.set_trace()
    #print image
    if type(image) in (str, unicode):
      image = plt.imread(image)
      #image = plt.imread(image,format='L')
    #pdb.set_trace()
    crop_edge_ratio = (256. - 227.) / 256. / 2
    ch = int(image.shape[0] * crop_edge_ratio + 0.5)
    cw = int(image.shape[1] * crop_edge_ratio + 0.5)
    cropped_image = image[ch:-ch, cw:-cw]
    #print cw,' ',ch,' ',cropped_image.shape
    if len(cropped_image.shape) == 2:
      cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
    preprocessed_image = self.transformer.preprocess('data', cropped_image)
    if verbose:
      print 'Preprocessed image has shape %s, range (%f, %f)' % \
          (preprocessed_image.shape,
           preprocessed_image.min(),
           preprocessed_image.max())
    #print preprocessed_image.shape
    return preprocessed_image

  def preprocessed_image_to_descriptor(self, image, output_name='fc8'):
    net = self.image_net
    if net.blobs['data'].data.shape[0] > 1:
      batch = np.zeros_like(net.blobs['data'].data)
      batch[0] = image[0]
    else:
      batch = image
    net.forward(data=batch)
    descriptor = net.blobs[output_name].data[0].copy()
    return descriptor

  def image_to_descriptor(self, image, output_name='fc8'):
    return self.preprocessed_image_to_descriptor(self.preprocess_image(image))

  def predict_single_word(self, descriptor, previous_word, output='probs'):
    net = self.lstm_net
    cont = 0 if previous_word == 0 else 1
    cont_input = np.array([cont])
    word_input = np.array([previous_word])
    image_features = np.zeros_like(net.blobs['image_features'].data)
    image_features[:] = descriptor
    net.forward(image_features=image_features, cont_sentence=cont_input,
                input_sentence=word_input)
    output_preds = net.blobs[output].data[0, 0, :]
    return output_preds

  def predict_single_word_from_all_previous(self, descriptor, previous_words):
    for word in [0] + previous_words:
      probs = self.predict_single_word(descriptor, word)
    return probs

  # Strategy must be either 'beam' or 'sample'.
  # If 'beam', do a max likelihood beam search with beam size num_samples.
  # Otherwise, sample with temperature temp.
  def predict_caption(self, descriptor, strategy={'type': 'beam'}):
    assert 'type' in strategy
    assert strategy['type'] in ('beam', 'sample')
    if strategy['type'] == 'beam':
      return self.predict_caption_beam_search(descriptor, strategy)
    num_samples = strategy['num'] if 'num' in strategy else 1
    samples_forward = []
    sample_probs_forward = []
    samples_backward = []
    sample_probs_backward = []
    for _ in range(num_samples):
      sample_forward, sample_prob_forward = self.sample_caption(descriptor, strategy,net_output='predict_forward')
      samples_forward.append(sample_forward)
      sample_probs_forward.append(sample_prob_forward)
      
      sample_backward, sample_prob_backward = self.sample_caption(descriptor, strategy,net_output='predict_backward')
      samples_backward.append(sample_forward)
      sample_probs_backward.append(sample_prob_forward)
    return samples_forward, sample_probs_forward,samples_backward, sample_probs_backward

  def sample_caption(self, descriptor, strategy,
                     net_output='predict', max_length=50):
    sentence = []
    probs = []
    eps_prob = 1e-8
    temp = strategy['temp'] if 'temp' in strategy else 1.0
    if max_length < 0: max_length = float('inf')
    while len(sentence) < max_length and (not sentence or sentence[-1] != 0):
      previous_word = sentence[-1] if sentence else 0
      softmax_inputs = self.predict_single_word(descriptor, previous_word,
                                                output=net_output)
      word = random_choice_from_probs(softmax_inputs, temp)
      sentence.append(word)
      probs.append(softmax(softmax_inputs, 1.0)[word])
    return sentence, probs

  def predict_caption_beam_search(self, descriptor, strategy, max_length=50):
    orig_batch_size = self.caption_batch_size()
    if orig_batch_size != 1: self.set_caption_batch_size(1)
    beam_size = strategy['beam_size'] if 'beam_size' in strategy else 1
    assert beam_size >= 1
    beams = [[]]
    beams_complete = 0
    beam_probs = [[]]
    beam_log_probs = [0.]
    while beams_complete < len(beams):
      expansions = []
      for beam_index, beam_log_prob, beam in \
          zip(range(len(beams)), beam_log_probs, beams):
        if beam:
          previous_word = beam[-1]
          if len(beam) >= max_length or previous_word == 0:
            exp = {'prefix_beam_index': beam_index, 'extension': [],
                   'prob_extension': [], 'log_prob': beam_log_prob}
            expansions.append(exp)
            # Don't expand this beam; it was already ended with an EOS,
            # or is the max length.
            continue
        else:
          previous_word = 0  # EOS is first word
        if beam_size == 1:
          probs = self.predict_single_word(descriptor, previous_word)
        else:
          probs = self.predict_single_word_from_all_previous(descriptor, beam)
        assert len(probs.shape) == 1
        assert probs.shape[0] == len(self.vocab)
        expansion_inds = probs.argsort()[-beam_size:]
        for ind in expansion_inds:
          prob = probs[ind]
          extended_beam_log_prob = beam_log_prob + math.log(prob)
          exp = {'prefix_beam_index': beam_index, 'extension': [ind],
                 'prob_extension': [prob], 'log_prob': extended_beam_log_prob}
          expansions.append(exp)
      # Sort expansions in decreasing order of probability.
      expansions.sort(key=lambda expansion: -1 * expansion['log_prob'])
      expansions = expansions[:beam_size]
      new_beams = \
          [beams[e['prefix_beam_index']] + e['extension'] for e in expansions]
      new_beam_probs = \
          [beam_probs[e['prefix_beam_index']] + e['prob_extension'] for e in expansions]
      beam_log_probs = [e['log_prob'] for e in expansions]
      beams_complete = 0
      for beam in new_beams:
        if beam[-1] == 0 or len(beam) >= max_length: beams_complete += 1
      beams, beam_probs = new_beams, new_beam_probs
    if orig_batch_size != 1: self.set_caption_batch_size(orig_batch_size)
    return beams, beam_probs

  def score_caption(self, descriptor, caption, is_gt=True, caption_source='gt'):
    output = {}
    output['caption'] = caption
    output['gt'] = is_gt
    output['source'] = caption_source
    output['prob'] = []
    probs = self.predict_single_word(descriptor, 0)
    for word in caption:
      output['prob'].append(probs[word])
      probs = self.predict_single_word(descriptor, word)
    return output

  def compute_descriptors(self, image_list, output_name='fc8'):
    batch = np.zeros_like(self.image_net.blobs['data'].data)
    batch_shape = batch.shape
    batch_size = batch_shape[0]
    descriptors_shape = (len(image_list), ) + \
        self.image_net.blobs[output_name].data.shape[1:]
    descriptors = np.zeros(descriptors_shape)
    for batch_start_index in range(0, len(image_list), batch_size):
      batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
      for batch_index, image_path in enumerate(batch_list):
        batch[batch_index:(batch_index + 1)] = self.preprocess_image(image_path)
      current_batch_size = min(batch_size, len(image_list) - batch_start_index)
      print 'Computing descriptors for images %d-%d of %d' % \
          (batch_start_index, batch_start_index + current_batch_size - 1,
           len(image_list))
      self.image_net.forward(data=batch)
      descriptors[batch_start_index:(batch_start_index + current_batch_size)] = \
          self.image_net.blobs[output_name].data[:current_batch_size]
    return descriptors

  def score_captions(self, descriptor, captions,
                     output_name='probs_forward', caption_source='gt', verbose=True):
    net = self.lstm_net
    cont_input = np.zeros_like(net.blobs['cont_sentence_forward'].data)
    word_input = np.zeros_like(net.blobs['input_sentence_forward'].data)
    image_features = np.zeros_like(net.blobs['image_features'].data)
    batch_size = image_features.shape[0]
    assert descriptor.shape == image_features.shape[1:]
    for index in range(batch_size):
      image_features[index] = descriptor
    outputs = []
    input_data_initialized = False
    for batch_start_index in range(0, len(captions), batch_size):
      caption_batch = captions[batch_start_index:(batch_start_index + batch_size)]
      current_batch_size = len(caption_batch)
      caption_index = 0
      probs_batch = [[] for b in range(current_batch_size)]
      num_done = 0
      while num_done < current_batch_size:
        if caption_index == 0:
          cont_input[:] = 0
        elif caption_index == 1:
          cont_input[:] = 1
        for index, caption in enumerate(caption_batch):
          word_input[0, index] = \
              caption['caption'][caption_index - 1] if \
              0 < caption_index < len(caption['caption']) else 0
        if input_data_initialized:
          net.forward(start="embedding_forward", image_features=image_features,cont_sentence_forward=cont_input,
                  input_sentence_forward=word_input,cont_sentence_backward=cont_input,
                  input_sentence_backward=word_input)
        else:
          net.forward(cont_sentence_forward=cont_input,
          input_sentence_forward=word_input,cont_sentence_backward=cont_input,
                  input_sentence_backward=word_input,
                      image_features=image_features)
          input_data_initialized = True
        #print output_name
        output_probs = net.blobs[output_name].data
        for index, probs, caption in \
            zip(range(current_batch_size), probs_batch, caption_batch):
          if caption_index == len(caption['caption']) - 1:
            num_done += 1
          if caption_index < len(caption['caption']):
            word = caption['caption'][caption_index]
            probs.append(output_probs[0, index, word].reshape(-1)[0])
        if verbose:
          print 'Computed probs for word %d of captions %d-%d (%d done)' % \
              (caption_index, batch_start_index,
               batch_start_index + current_batch_size - 1, num_done)
        caption_index += 1
      for prob, caption in zip(probs_batch, caption_batch):
        output = {}
        output['caption'] = caption['caption']
        output['prob'] = prob
        output['gt'] = True
        output['source'] = caption_source
        outputs.append(output)
    return outputs

  def sample_captions(self, descriptor, prob_output_name_forward='probs_forward',
                      pred_output_name_forward='predict_forward',prob_output_name_backward='probs_backward',
                      pred_output_name_backward='predict_backward',temp=1, max_length=50):
    descriptor = np.array(descriptor)
    batch_size = descriptor.shape[0]
    self.set_caption_batch_size(batch_size)
    net = self.lstm_net
    #pdb.set_trace()
    #print net
    cont_input_forward = np.zeros_like(net.blobs['cont_sentence_forward'].data)
    word_input_forward = np.zeros_like(net.blobs['input_sentence_forward'].data)
    cont_input_backward = np.zeros_like(net.blobs['cont_sentence_backward'].data)
    word_input_backward = np.zeros_like(net.blobs['input_sentence_backward'].data)
    #pdb.set_trace()
    #print cont_input.shape
    #print word_input.shape
    #print cont_input
    #print word_input
    image_features = np.zeros_like(net.blobs['image_features'].data)
    image_features[:] = descriptor
    
    
    outputs_forward = []
    output_captions_forward = [[] for b in range(batch_size)]
    output_probs_forward = [[] for b in range(batch_size)]
    
    outputs_backward = []
    output_captions_backward = [[] for b in range(batch_size)]
    output_probs_backward = [[] for b in range(batch_size)]
    
    caption_index_forward = 0
    caption_index_backward = 0
    num_done_forward = 0
    num_done_backward = 0
    while num_done_forward < batch_size and caption_index_forward < max_length:
      if caption_index_forward == 0:
        cont_input_forward [:] = 0
      elif caption_index_forward == 1:
        cont_input_forward [:] = 1
      if caption_index_forward == 0:
        word_input_forward [:] = 0
      else:
        for index in range(batch_size):
          word_input_forward[0, index] = \
              output_captions_forward[index][caption_index_forward - 1] if \
              caption_index_forward <= len(output_captions_forward[index]) else 0
      net.forward(image_features=image_features, cont_sentence_forward=cont_input_forward,
                  input_sentence_forward=word_input_forward,cont_sentence_backward=cont_input_backward,
                  input_sentence_backward=word_input_backward)
      if temp == 1.0 or temp == float('inf'):
        net_output_probs_forward = net.blobs[prob_output_name_forward].data[0]
        samples_forward = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True)
            for dist in net_output_probs_forward
        ]
       
      else:
        net_output_preds_forward = net.blobs[pred_output_name_forward].data[0]
        samples_forward = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_preds_forward
        ]
      
      for index, next_word_sample in enumerate(samples_forward):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.

        if not output_captions_forward[index] or output_captions_forward[index][-1] != 0:
          output_captions_forward[index].append(next_word_sample)
          output_probs_forward[index].append(net_output_probs_forward[index, next_word_sample])
          if next_word_sample == 0: num_done_forward += 1
      sys.stdout.write('\r forward sampling: %d/%d done after word %d' %(num_done_forward, batch_size, caption_index_forward))
      sys.stdout.flush()
      caption_index_forward += 1
      
      
    while num_done_backward < batch_size and caption_index_backward < max_length:
      if caption_index_backward == 0:
        cont_input_backward [:] = 0
      elif caption_index_backward == 1:
        cont_input_backward [:] = 1
      if caption_index_backward == 0:
        word_input_backward [:] = 0
      else:
        for index in range(batch_size):
          word_input_backward[0, index] = \
              output_captions_backward[index][caption_index_backward - 1] if \
              caption_index_backward <= len(output_captions_backward[index]) else 0
      net.forward(image_features=image_features, cont_sentence_forward=cont_input_forward,
                  input_sentence_forward=word_input_forward,cont_sentence_backward=cont_input_backward,
                  input_sentence_backward=word_input_backward)
      if temp == 1.0 or temp == float('inf'):
        net_output_probs_backward = net.blobs[prob_output_name_backward].data[0]
        samples_backward = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True)
            for dist in net_output_probs_backward
        ]
       
      else:
        net_output_preds_backward = net.blobs[pred_output_name_backward].data[0]
        samples_backward = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_preds_backward
        ]
      
      for index, next_word_sample in enumerate(samples_backward):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.

        if not output_captions_backward[index] or output_captions_backward[index][-1] != 0:
          output_captions_backward[index].append(next_word_sample)
          output_probs_backward[index].append(net_output_probs_backward[index, next_word_sample])
          if next_word_sample == 0: num_done_backward += 1
      sys.stdout.write('\r backward sampling: %d/%d done after word %d' %(num_done_backward, batch_size, caption_index_backward))
      sys.stdout.flush()
      caption_index_backward += 1
    sys.stdout.write('\n')
    #pdb.set_trace()
    #print output_captions_forward[0]
    #print output_captions_backward[0]
    #print output_probs_forward[0]
    #print output_probs_backward[0] 
    return output_captions_forward, output_probs_forward,output_captions_backward, output_probs_backward

  def sentence(self, vocab_indices):
    sentence = ' '.join([self.vocab[i] for i in vocab_indices])
    if not sentence: return sentence
    sentence = sentence[0].upper() + sentence[1:]
    # If sentence ends with ' <EOS>', remove and replace with '.'
    # Otherwise (doesn't end with '<EOS>' -- maybe was the max length?):
    # append '...'
    suffix = ' ' + self.vocab[0]
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)] + '.'
    else:
      sentence += '...'
    return sentence

def softmax(softmax_inputs, temp):
  shifted_inputs = softmax_inputs - softmax_inputs.max()
  exp_outputs = np.exp(temp * shifted_inputs)
  exp_outputs_sum = exp_outputs.sum()
  if math.isnan(exp_outputs_sum):
    return exp_outputs * float('nan')
  assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False):
  # temperature of infinity == take the max
  if temp == float('inf'):
    return np.argmax(softmax_inputs)
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += math.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  stats['p'] = math.exp(stats['log_p'])
  stats['p_word'] = math.exp(stats['log_p'])
  try:
    stats['perplex'] = math.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = math.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats
