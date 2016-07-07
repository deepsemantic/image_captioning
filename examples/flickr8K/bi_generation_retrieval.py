#!/usr/bin/env python

from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys
import pdb

# seed the RNG so we evaluate on the same subset each time
np.random.seed(seed=0)

from flickr8K_to_hdf5_data_retrieval import *
from bi_captioner import Captioner

COCO_EVAL_PATH = './data/coco/coco-caption-eval'
sys.path.append(COCO_EVAL_PATH)
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.flickr8K_eval import FLICKR8KEvalCap

class CaptionExperiment():
  # captioner is an initialized Captioner (captioner.py)
  # dataset is a dict: image path -> [caption1, caption2, ...]
  def __init__(self, captioner, dataset_forward,dataset_backward, dataset_cache_dir, cache_dir, sg):
    self.captioner = captioner
    self.sg = sg
    self.dataset_cache_dir = dataset_cache_dir
    self.cache_dir = cache_dir
    for d in [dataset_cache_dir, cache_dir]:
      if not os.path.exists(d): os.makedirs(d)
    self.dataset_forward = dataset_forward
    self.dataset_backward = dataset_backward
    self.images = dataset_forward.keys()
    #pdb.set_trace()
    #print dataset_forward[0],len(dataset_backward)
    self.init_caption_list_forward(dataset_forward)
    self.init_caption_list_backward(dataset_backward)
    self.caption_scores = [None] * len(self.images)
    print 'Initialized caption experiment in forward order: %d images, %d captions' % \
        (len(self.images), len(self.captions_forward))
    print 'Initialized caption experiment in backward order: %d images, %d captions' % \
        (len(self.images), len(self.captions_backward))
    #pdb.set_trace()
        
        
  def init_caption_list_forward(self, dataset):
    self.captions_forward = []
    for image, captions in dataset.iteritems():
      for caption, _ in captions:
        self.captions_forward.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    self.captions_forward.sort(key=lambda c: len(c['caption']))


  def init_caption_list_backward(self, dataset):
    self.captions_backward = []
    for image, captions in dataset.iteritems():
      for caption, _ in captions:
        self.captions_backward.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    self.captions_backward.sort(key=lambda c: len(c['caption']))


  def compute_descriptors(self):
    descriptor_filename = '%s/descriptors.npz' % self.dataset_cache_dir
    if os.path.exists(descriptor_filename):
      self.descriptors = np.load(descriptor_filename)['descriptors']
    else:
      self.descriptors = self.captioner.compute_descriptors(self.images)
      np.savez_compressed(descriptor_filename, descriptors=self.descriptors)

  def score_captions(self, image_index, output_name='probs_forward'):
    assert image_index < len(self.images)
    caption_scores_dir = '%s/caption_scores' % self.cache_dir
    if not os.path.exists(caption_scores_dir):
      os.makedirs(caption_scores_dir)
    caption_scores_filename = '%s/scores_image_%06d.pkl' % \
        (caption_scores_dir, image_index)
    if os.path.exists(caption_scores_filename):
      with open(caption_scores_filename, 'rb') as caption_scores_file:
        outputs = pickle.load(caption_scores_file)
    else:
      outputs = self.captioner.score_captions(self.descriptors[image_index],
          self.captions_forward, output_name=output_name, caption_source='gt',
          verbose=False)
      self.caption_stats(image_index, outputs)
      with open(caption_scores_filename, 'wb') as caption_scores_file:
        pickle.dump(outputs, caption_scores_file)
    self.caption_scores[image_index] = outputs

  def caption_stats(self, image_index, caption_scores):
    image_path = self.images[image_index]
    for caption, score in zip(self.captions_forward, caption_scores):
      assert caption['caption'] == score['caption']
      score['stats'] = gen_stats(score['prob'])
      score['correct'] = (image_path == caption['source_image'])

  def eval_image_to_caption(self, image_index, methods=None):
    scores = self.caption_scores[image_index]
    return self.eval_recall(scores, methods=methods)

  def eval_caption_to_image(self, caption_index, methods=None):
    scores = [s[caption_index] for s in self.caption_scores]
    return self.eval_recall(scores, methods=methods)

  def normalize_caption_scores(self, caption_index, stats=['log_p', 'log_p_word']):
    scores = [s[caption_index] for s in self.caption_scores]
    for stat in stats:
      log_stat_scores = np.array([score['stats'][stat] for score in scores])
      stat_scores = np.exp(log_stat_scores)
      mean_stat_score = np.mean(stat_scores)
      log_mean_stat_score = np.log(mean_stat_score)
      for log_stat_score, score in zip(log_stat_scores, scores):
        score['stats']['normalized_' + stat] = log_stat_score - log_mean_stat_score

  def eval_recall(self, scores, methods=None, neg_prefix='negative_'):
    if methods is None:
      # rank on all stats, and all their inverses
      methods = scores[0]['stats'].keys()
      methods += [neg_prefix + method for method in methods]
    correct_ranks = {}
    for method in methods:
      if method.startswith(neg_prefix):
        multiplier = -1
        method_key = method[len(neg_prefix):]
      else:
        multiplier = 1
        method_key = method
      sort_key = lambda s: multiplier * s['stats'][method_key]
      ranked_scores = sorted(scores, key=sort_key)
      for index, score in enumerate(ranked_scores):
        if score['correct']:
          correct_ranks[method] = index
          break
    return correct_ranks

  def recall_results(self, correct_ranks, recall_ranks=[]):
    num_instances = float(len(correct_ranks))
    assert num_instances > 0
    methods = correct_ranks[0].keys()
    results = {}
    for method in methods:
       method_correct_ranks = \
           np.array([correct_rank[method] for correct_rank in correct_ranks])
       r = OrderedDict()
       r['mean'] = np.mean(method_correct_ranks)
       r['median'] = np.median(method_correct_ranks)
       r['mean (1-indexed)'] = r['mean'] + 1
       r['median (1-indexed)'] = r['median'] + 1
       for recall_rank in recall_ranks:
         r['R@%d' % recall_rank] = \
             np.where(method_correct_ranks < recall_rank)[0].shape[0] / num_instances
       results[method] = r
    return results

  def print_recall_results(self, results):
    for method, result in results.iteritems():
      print 'Ranking method:', method
      for metric_name_and_value in result.iteritems():
        print '    %s: %f' % metric_name_and_value

  def retrieval_experiment(self):
    # Compute image descriptors.
    print 'Computing image descriptors'
    self.compute_descriptors()

    num_images, num_captions = len(self.images), len(self.captions_forward)

    # For each image, score all captions.
    for image_index in xrange(num_images):
      sys.stdout.write("\rScoring captions for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
      self.score_captions(image_index)
    sys.stdout.write('\n')

    # Compute global caption statistics for normalization.
    for caption_index in xrange(num_captions):
      self.normalize_caption_scores(caption_index)

    recall_ranks = [1, 5, 10, 50]

    eval_methods = ['negative_normalized_log_p']
    # Evaluate caption-to-image retrieval task.
    self.caption_to_image_ranks = [None] * num_captions
    for caption_index in xrange(num_captions):
      sys.stdout.write("\rCaption-to-image evaluation: "
                       "computing recall for caption %d/%d" %
                       (caption_index, num_captions))
      sys.stdout.flush()
      self.caption_to_image_ranks[caption_index] = \
          self.eval_caption_to_image(caption_index, methods=eval_methods)
    sys.stdout.write('\n')
    self.caption_to_image_recall = \
         self.recall_results(self.caption_to_image_ranks, recall_ranks)
    print 'Caption-to-image retrieval results:'
    self.print_recall_results(self.caption_to_image_recall)

    # Evaluate image-to-caption retrieval task.
    self.image_to_caption_ranks = [None] * num_images
    for image_index in xrange(num_images):
      sys.stdout.write("\rImage-to-caption evaluation: "
                       "computing recall for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
      self.image_to_caption_ranks[image_index] = \
          self.eval_image_to_caption(image_index, methods=eval_methods)
    sys.stdout.write('\n')
    self.image_to_caption_recall = \
        self.recall_results(self.image_to_caption_ranks, recall_ranks)
    print 'Image-to-caption retrieval results:'
    self.print_recall_results(self.image_to_caption_recall)

  def generation_experiment(self, strategy, max_batch_size=1000):
    # Compute image descriptors.
    print 'Computing image descriptors'
    self.compute_descriptors()

    do_batches = (strategy['type'] == 'beam' and strategy['beam_size'] == 1) or \
        (strategy['type'] == 'sample' and
         ('temp' not in strategy or strategy['temp'] in (1, float('inf'))) and
         ('num' not in strategy or strategy['num'] == 1))

    num_images = len(self.images)
    batch_size = min(max_batch_size, num_images) if do_batches else 1
    
    #pdb.set_trace()
    # Generate captions for all images.
    all_captions_forward = [None] * num_images
    all_captions_backward = [None] * num_images
    for image_index in xrange(0, num_images, batch_size):
      batch_end_index = min(image_index + batch_size, num_images)
      sys.stdout.write("\rGenerating captions for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
      if do_batches:
        if strategy['type'] == 'beam' or \
            ('temp' in strategy and strategy['temp'] == float('inf')):
          temp = float('inf')
        else:
          temp = strategy['temp'] if 'temp' in strategy else 1
        output_captions_forward, output_probs_forward,output_captions_backward, output_probs_backward = self.captioner.sample_captions(self.descriptors[image_index:batch_end_index], temp=temp)
        for batch_index, output in zip(range(image_index, batch_end_index),
                                       output_captions_forward):
          all_captions_forward[batch_index] = output
        for batch_index, output in zip(range(image_index, batch_end_index),
                                       output_captions_backward):
          all_captions_backward[batch_index] = output
      else:
        for batch_image_index in xrange(image_index, batch_end_index):
          captions_forward, caption_probs_forward,captions_backward, caption_probs_backward = self.captioner.predict_caption(self.descriptors[batch_image_index], strategy=strategy)
          best_caption, max_log_prob = None, None
          for caption, probs in zip(captions_forward, caption_probs_forward):
            log_prob = gen_stats(probs)['log_p']
            if best_caption is None or \
                (best_caption is not None and log_prob > max_log_prob):
              best_caption, max_log_prob = caption, log_prob
          all_captions_forward[batch_image_index] = best_caption
          
          best_caption, max_log_prob = None, None
          for caption, probs in zip(captions_backward, caption_probs_backward):
            log_prob = gen_stats(probs)['log_p']
            if best_caption is None or \
                (best_caption is not None and log_prob > max_log_prob):
              best_caption, max_log_prob = caption, log_prob
          all_captions_backward[batch_image_index] = best_caption
    sys.stdout.write('\n')
    #pdb.set_trace()
    #print all_captions_forward
    #print all_captions_backward
    #print len(output_probs_forward)
    #print output_probs_backward
    
    ##Compute the final caption according to forward and backward order.
    all_captions_final = [None] * num_images
    cap_index=0
    
    print 'Compute the final caption according to forward and backward order.. '
    for cap_f, cap_b, pros_f, pros_b in zip(all_captions_forward,all_captions_backward,output_probs_forward,output_probs_backward):
        #print sum(pros_f), ' ',sum(pros_b)
        if sum(pros_f)>sum(pros_b):
            all_captions_final[cap_index]=cap_f
        else:
            #pdb.set_trace()
            #print cap_index
            all_captions_final[cap_index]=cap_b[0:-1][::-1]+[0]
        #print all_captions_forward[cap_index]
        #print all_captions_backward[cap_index]
        #print all_captions_final[cap_index]
        cap_index+=1
    
    
    # Compute the number of reference files as the maximum number of ground
    # truth captions of any image in the dataset.
    num_reference_files = 0
    for captions in self.dataset_forward.values():
      if len(captions) > num_reference_files:
        num_reference_files = len(captions)
    if num_reference_files <= 0:
      raise Exception('No reference captions.')

    # Collect model/reference captions, formatting the model's captions and
    # each set of reference captions as a list of len(self.images) strings.
    exp_dir = '%s/generation' % self.cache_dir
    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    # For each image, write out the highest probability caption.
    model_captions = [''] * len(self.images)
    reference_captions = [([''] * len(self.images)) for _ in xrange(num_reference_files)]
    for image_index, image in enumerate(self.images):
      #pdb.set_trace()
      print image_index,' ',image
      caption = self.captioner.sentence(all_captions_final[image_index])
      print caption
      model_captions[image_index] = caption
      for reference_index, (_, caption) in enumerate(self.dataset_forward[image]):
        caption = ' '.join(caption)
        reference_captions[reference_index][image_index] = caption

    coco_image_ids = [self.sg.image_path_to_id[image_path]
                      for image_path in self.images]
    
    #print coco_image_ids
  
    #print self.sg.image_path_to_id
  
    generation_result = [{
      'image_id':self.sg.image_path_to_id[image_path],
      'caption': model_captions[image_index]
    } for (image_index, image_path) in enumerate(self.images)]
    
    print 'generation_result...len   ',generation_result[0]['image_id']
    json_filename = '%s/generation_result.json' % self.cache_dir
    print 'Dumping result to file: %s' % json_filename
    with open(json_filename, 'w') as json_file:
      json.dump(generation_result, json_file)
  
  
  #get the ground truth from val
    image_id_caption_pairs=[]
    DATASET_NAME='val'
    with open(CAPTION_SPLIT%DATASET_NAME, 'r') as split_file:
     for line in split_file.readlines():
        image_caption=line.strip().split('#')
        image_id=image_caption[0].replace('.jpg','')
        caption=image_caption[1].strip()
        #print image_id,'   ',caption    
        image_id_caption_pairs.append((image_id, caption))
        #print image_id_caption_pairs
    groundtruth_result = [{
      'image_id':image_id,
      'caption': caption
    } for image_id, caption in image_id_caption_pairs]
##    
##    [{u'image_id': 63647, u'id': 95101, u'caption': u'A living room with red leather furniture in it'}, 
##    {u'image_id': 63647, u'id': 105070, u'caption': u'A burgundy leather couch in a living room.'}, 
##    {u'image_id': 63647, u'id': 111685, u'caption': u'A living room decorated with red couch and chair.'}, 
##    {u'image_id': 63647, u'id': 112537, u'caption': u'The furniture in the small living room is mostly red.'}, 
##    {u'image_id': 63647, u'id': 115666, u'caption': u'A living room has a red sofa and chair.'}]   
##    [{u'image_id': 63647, 'id': 1, u'caption': u'A living room with a couch and a couch.'}]

   #prepare to the fromat that coco evalution tool accept
    evl_groundtruth_result={}
    evl_generation_result={}
    evl_image_id=[]
    for generate_item in generation_result:
      sub_groundtruth_result=[]
      sub_generation_result=[]
      for ground_item in groundtruth_result:
        if ground_item['image_id']==generate_item['image_id']:
          sub_groundtruth_result.append(ground_item)
      sub_generation_result.append(generate_item)
      evl_groundtruth_result[generate_item['image_id']]=sub_groundtruth_result
      evl_generation_result[generate_item['image_id']]=sub_generation_result
      evl_image_id.append(generate_item['image_id'])
    #pdb.set_trace()
    #print evl_groundtruth_result,'  ',evl_generation_result
          
        
    #generation_result = self.sg.coco.loadRes(json_filename) #get generated results
    #coco_evaluator = COCOEvalCap(self.sg.coco, generation_result) # compare two results
    coco_evaluator=FLICKR8KEvalCap( evl_groundtruth_result,evl_generation_result,np.array(evl_image_id))
    #coco_evaluator.params['image_id'] = coco_image_ids
    coco_evaluator.evaluate()

def gen_stats(prob):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += np.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = np.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = np.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  return stats

def main():
  MAX_IMAGES =1000  # -1 to use all images
  TAG = 'flickr 8K'
  if MAX_IMAGES >= 0:
    TAG += '_%dimages' % MAX_IMAGES
  eval_on_test = False
  if eval_on_test:
    ITER = 100000
    MODEL_FILENAME = 'lrcn_finetune_trainval_stepsize40k_iter_%d' % ITER
    DATASET_NAME = 'test'
  else:  # eval on val
    ITER =15000
    MODEL_FILENAME = 'multi_Bi_LSTM_iter_%d' % ITER  
    DATASET_NAME = 'val'
  TAG += '_%s' % DATASET_NAME
  MODEL_DIR = './examples/flickr8K/multi_Bi_LSTM_trained_models'
  MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
  #IMAGE_NET_FILE = './models/vgg16/vgg16_deply.prototxt'
  IMAGE_NET_FILE = './models/bvlc_reference_caffenet/deploy.prototxt'
  LSTM_NET_FILE = './examples/flickr8K/multi_Bi_LSTM.deploy.prototxt'
  #LSTM_NET_FILE = './examples/flickr8K/bi_lrcn_word_to_preds.deploy_multi_task.prototxt'
  NET_TAG = '%s_%s' % (TAG, MODEL_FILENAME)
  DATASET_SUBDIR = '%s/%s_ims' % (DATASET_NAME,
      str(MAX_IMAGES) if MAX_IMAGES >= 0 else 'all')
  DATASET_CACHE_DIR = './examples/flickr8K/bi_flicrk8K_generation_retrieval_cache/%s/%s' % (DATASET_SUBDIR, MODEL_FILENAME)
  VOCAB_FILE = './examples/flickr8K/h5_data_forward/buffer_150/vocabulary.txt'
  #VOCAB_FILE = './examples/mscoco_flickr30/h5_data_forward/buffer_200/vocabulary.txt'
  SPLITS_PATTERN = './data/flickr8K/texts/Flickr_8k.%s.txt' # used for generate image id in retrieval experiment
  DEVICE_ID = 0
  with open(VOCAB_FILE, 'r') as vocab_file:
    vocab = [line.strip() for line in vocab_file.readlines()]
  coco = COCO() # for retrieval experiments
  image_root = COCO_IMAGE_PATTERN % DATASET_NAME
  print image_root
  
  with open(SPLITS_PATTERN % DATASET_NAME, 'r') as split_file:
    split_image_ids = [line.strip().replace('.jpg','') for line in split_file.readlines()] # line.strip(): remove the '\n' in each line
    
  sg = CocoSequenceGenerator(coco, DATASET_NAME,BUFFER_SIZE, image_root, split_ids=split_image_ids,vocab=vocab,
                             align=False, shuffle=False)
  dataset_forward = {}
  dataset_backward = {}
  #print len(sg.image_sentence_pairs)
  #pdb.set_trace()
  for image_path, sentence_forward,sentence_backward in sg.image_sentence_pairs:
    #print image_path," ",sentence_forward," ",sentence_backward
   # pdb.set_trace()
    if image_path not in dataset_forward:
      dataset_forward[image_path] = []
      dataset_backward[image_path] = []
    dataset_forward[image_path].append((sg.line_to_stream(sentence_forward), sentence_forward))
    dataset_backward[image_path].append((sg.line_to_stream(sentence_backward), sentence_backward))
  #pdb.set_trace()
  #print len(dataset_forward)
  #print dataset_backward
  print 'Original dataset contains %d images' % len(dataset_forward.keys())
  
  
  
  if 0 <= MAX_IMAGES < len(dataset_forward.keys()):
    all_keys = dataset_forward.keys()
    perm = np.random.permutation(len(all_keys))[:MAX_IMAGES]
    chosen_keys = set([all_keys[p] for p in perm])
    for key in all_keys:
      if key not in chosen_keys:
        del dataset_forward[key]
        del dataset_backward[key]
    print 'Reduced dataset to %d images' % len(dataset_forward.keys())
  if MAX_IMAGES < 0: MAX_IMAGES = len(dataset_forward.keys())
  captioner = Captioner(MODEL_FILE, IMAGE_NET_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)
  beam_size = 1
  generation_strategy = {'type': 'beam', 'beam_size': beam_size}
  if generation_strategy['type'] == 'beam':
    strategy_name = 'beam%d' % generation_strategy['beam_size']
  elif generation_strategy['type'] == 'sample':
    strategy_name = 'sample%f' % generation_strategy['temp']
  else:
    raise Exception('Unknown generation strategy type: %s' % generation_strategy['type'])
  CACHE_DIR = '%s/%s' % (DATASET_CACHE_DIR, strategy_name)
  experimenter = CaptionExperiment(captioner, dataset_forward,dataset_backward, DATASET_CACHE_DIR, CACHE_DIR, sg)
  captioner.set_image_batch_size(min(100, MAX_IMAGES))
  experimenter.generation_experiment(generation_strategy)
  captioner.set_caption_batch_size(min(MAX_IMAGES * 5, 1000))
  experimenter.retrieval_experiment()

if __name__ == "__main__":
  main()
