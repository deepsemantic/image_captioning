#!/usr/bin/env python

from hashlib import sha1
import os
import random
random.seed(3)
import re
import sys

import pdb

sys.path.append('./examples/flickr8K/')
COCO_PATH = './data/coco/coco'
DATA_PATH = './data/flickr8K'
COCO_TOOL_PATH = '%s/PythonAPI/pycocotools' % COCO_PATH
COCO_IMAGE_ROOT = '%s/images' % DATA_PATH
IMAGE_CAPTION='%s/texts/Flickr8k.lemma.token.txt'% DATA_PATH
CAPTION_SPLIT='./data/flickr8K/texts/Flickr_8k_caption.%s.txt'

MAX_HASH = 100000
sys.path.append(COCO_TOOL_PATH)
from coco import COCO

from hdf5_sequence_generator import SequenceGenerator, HDF5SequenceWriter


# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<unk>'

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  #print 'sentence...1 : ', sentence
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  # remove the '.' from the end of the sentence
  if sentence[-1] != '.':
    # print "Warning: sentence doesn't end with '.'; ends with: %s" % sentence[-1]
    #pdb.set_trace()
    #print 'sentence...2 : ', sentence
    return sentence

  return sentence[:-1]

MAX_WORDS = 20

def split_image_captions(split_name,image_root):
  image_sentence_pairs=[]
  with open(CAPTION_SPLIT%split_name, 'r') as split_file:
    for line in split_file.readlines():
      image_caption=line.strip().split('#')
      image=image_caption[0]
      caption=image_caption[1].replace('.','')
      #print image,'   ',caption
      image_path = '%s/%s' % (image_root,image)      
      sentence=split_sentence(caption)
      #pdb.set_trace()
      #print sentence
      #print sentence[::-1]
      image_sentence_pairs.append((image_path, sentence,sentence[::-1]))
      #print image_sentence_pairs
  return image_sentence_pairs
      

class CocoSequenceGenerator(SequenceGenerator):
  def __init__(self, coco,split_name,batch_num_streams, image_root, vocab=None,
               max_words=MAX_WORDS, align=True, shuffle=True, gt_captions=True,
               pad=True, truncate=True, split_ids=None):
            
    #split_ids: image list e.g. '2801146217_03a0b59ccb.jpg\n', '1321723162_9d4c78b8af.jpg\n'
    #
    #
    self.max_words = max_words
    num_empty_lines = 0
    self.images = []
    num_total = 0
    num_missing = 0
    num_captions = 0
    known_images = {}
    #self.coco = coco
  
    self.image_path_to_id = {}
    self.image_sentence_pairs = []
    
    ###
    ##  generate image_id list, which will be used in retrieval experiments
    ###
    if split_ids is None:
      split_ids = coco.imgs.keys()
    self.image_path_to_id = {}
   # pdb.set_trace()
   # print split_ids
    for image_id in split_ids:
      image_path = '%s/%s.jpg' % (image_root, image_id)
      
     # print 'image_info ',image_info
      #print 'image_path ',image_path
      self.image_path_to_id[image_path] = image_id
      #print self.image_path_to_id
      ###'./data/flickr8K/images/train/143688895_e837c3bc76.jpg': '143688895_e837c3bc76', 
    
    self.image_sentence_pairs=split_image_captions(split_name,image_root)    
    #print image_sentence_pairs
  
    #generate word vocabulary based on image caption sentences
    if vocab is None:
      self.init_vocabulary(self.image_sentence_pairs)
    else:
      self.vocabulary_inverted = vocab
      self.vocabulary = {}
      for index, word in enumerate(self.vocabulary_inverted):
        self.vocabulary[word] = index
         
    self.index = 0
    self.num_resets = 0
    self.num_truncates = 0
    self.num_pads = 0
    self.num_outs = 0
    self.image_list = []
    SequenceGenerator.__init__(self)
    self.batch_num_streams = batch_num_streams
    # make the number of image/sentence pairs a multiple of the buffer size
    # so each timestep of each batch is useful and we can align the images
    if align:
      num_pairs = len(self.image_sentence_pairs) 
      #pdb.set_trace()
      print 'number of pairs: ', num_pairs
      remainder = num_pairs % batch_num_streams
      if remainder > 0:
        num_needed = batch_num_streams - remainder
        for i in range(num_needed):
          choice = random.randint(0, num_pairs - 1)
          self.image_sentence_pairs.append(self.image_sentence_pairs[choice])
      assert len(self.image_sentence_pairs) % batch_num_streams == 0
    if shuffle:
      random.shuffle(self.image_sentence_pairs)
    self.pad = pad
    self.truncate = truncate
    self.negative_one_padded_streams = frozenset(('input_sentence', 'target_sentence'))
    
  def line_to_stream(self, sentence):
    stream = []
    for word in sentence:
      word = word.strip()
      if word in self.vocabulary:
        stream.append(self.vocabulary[word])
      else:  # unknown word; append UNK
        stream.append(self.vocabulary[UNK_IDENTIFIER])
    # increment the stream -- 0 will be the EOS character
    stream = [s + 1 for s in stream]
    return stream

BUFFER_SIZE = 5
COCO_IMAGE_PATTERN = '%s/images/%%s' % DATA_PATH

SPLITS_PATTERN = './data/flickr8K/texts/Flickr_8k.%s.txt'

if __name__ == "__main__":
  process_coco(include_trainval=False)
