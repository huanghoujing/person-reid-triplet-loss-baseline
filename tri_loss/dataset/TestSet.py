from __future__ import print_function
import sys
import time
import os.path as osp
from PIL import Image
import numpy as np
from collections import defaultdict

from .Dataset import Dataset

from ..utils.utils import measure_time
from ..utils.re_ranking import re_ranking
from ..utils.metric import cmc, mean_ap
from ..utils.dataset_utils import parse_im_name
from ..utils.distance import normalize
from ..utils.distance import compute_dist


class TestSet(Dataset):
  """
  Args:
    extract_feat_func: a function to extract features. It takes a batch of
      images and returns a batch of features.
    marks: a list, each element e denoting whether the image is from 
      query (e == 0), or
      gallery (e == 1), or 
      multi query (e == 2) set
  """

  def __init__(
      self,
      im_dir=None,
      im_names=None,
      marks=None,
      extract_feat_func=None,
      separate_camera_set=None,
      single_gallery_shot=None,
      first_match_break=None,
      **kwargs):

    super(TestSet, self).__init__(dataset_size=len(im_names), **kwargs)

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.marks = marks
    self.extract_feat_func = extract_feat_func
    self.separate_camera_set = separate_camera_set
    self.single_gallery_shot = single_gallery_shot
    self.first_match_break = first_match_break

  def set_feat_func(self, extract_feat_func):
    self.extract_feat_func = extract_feat_func

  def get_sample(self, ptr):
    im_name = self.im_names[ptr]
    im_path = osp.join(self.im_dir, im_name)
    im = np.asarray(Image.open(im_path))
    im, _ = self.pre_process_im(im)
    id = parse_im_name(self.im_names[ptr], 'id')
    cam = parse_im_name(self.im_names[ptr], 'cam')
    # denoting whether the im is from query, gallery, or multi query set
    mark = self.marks[ptr]
    return im, id, cam, im_name, mark

  def next_batch(self):
    if self.epoch_done and self.shuffle:
      self.prng.shuffle(self.im_names)
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, ids, cams, im_names, marks = zip(*samples)
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(im_list, axis=0)
    ids = np.array(ids)
    cams = np.array(cams)
    im_names = np.array(im_names)
    marks = np.array(marks)
    return ims, ids, cams, im_names, marks, self.epoch_done

  def extract_feat(self, normalize_feat, verbose=True):
    """Extract the features of the whole image set.
    Args:
      normalize_feat: True or False, whether to normalize feature to unit length
      verbose: whether to print the progress of extracting feature
    Returns:
      feat: numpy array with shape [N, C]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
    feat, ids, cams, im_names, marks = [], [], [], [], []
    done = False
    step = 0
    printed = False
    st = time.time()
    last_time = time.time()
    while not done:
      ims_, ids_, cams_, im_names_, marks_, done = self.next_batch()
      feat_ = self.extract_feat_func(ims_)
      feat.append(feat_)
      ids.append(ids_)
      cams.append(cams_)
      im_names.append(im_names_)
      marks.append(marks_)

      if verbose:
        # Print the progress of extracting feature
        total_batches = (self.prefetcher.dataset_size
                         // self.prefetcher.batch_size + 1)
        step += 1
        if step % 20 == 0:
          if not printed:
            printed = True
          else:
            # Clean the current line
            sys.stdout.write("\033[F\033[K")
          print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
                .format(step, total_batches,
                        time.time() - last_time, time.time() - st))
          last_time = time.time()

    feat = np.vstack(feat)
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    im_names = np.hstack(im_names)
    marks = np.hstack(marks)
    if normalize_feat:
      feat = normalize(feat, axis=1)
    return feat, ids, cams, im_names, marks

  def eval(
      self,
      normalize_feat=True,
      to_re_rank=True,
      pool_type='average',
      verbose=True):

    """Evaluate using metric CMC and mAP.
    Args:
      normalize_feat: whether to normalize features before computing distance
      to_re_rank: whether to also report re-ranking scores
      pool_type: 'average' or 'max', only for multi-query case
      verbose: whether to print the intermediate information
    """

    with measure_time('Extracting feature...', verbose=verbose):
      feat, ids, cams, im_names, marks = self.extract_feat(
        normalize_feat, verbose)

    # query, gallery, multi-query indices
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    # A helper function just for avoiding code duplication.
    def compute_score(
        dist_mat,
        query_ids=ids[q_inds],
        gallery_ids=ids[g_inds],
        query_cams=cams[q_inds],
        gallery_cams=cams[g_inds]):
      # Compute mean AP
      mAP = mean_ap(
        distmat=dist_mat,
        query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams)
      # Compute CMC scores
      cmc_scores = cmc(
        distmat=dist_mat,
        query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams,
        separate_camera_set=self.separate_camera_set,
        single_gallery_shot=self.single_gallery_shot,
        first_match_break=self.first_match_break,
        topk=10)
      return mAP, cmc_scores

    def print_scores(mAP, cmc_scores):
      print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
            .format(mAP, *cmc_scores[[0, 4, 9]]))

    ################
    # Single Query #
    ################

    with measure_time('Computing distance...', verbose=verbose):
      # query-gallery distance
      q_g_dist = compute_dist(feat[q_inds], feat[g_inds], type='euclidean')

    with measure_time('Computing scores...', verbose=verbose):
      mAP, cmc_scores = compute_score(q_g_dist)

    print('{:<30}'.format('Single Query:'), end='')
    print_scores(mAP, cmc_scores)

    ###############
    # Multi Query #
    ###############

    mq_mAP, mq_cmc_scores = None, None
    if any(mq_inds):
      mq_ids = ids[mq_inds]
      mq_cams = cams[mq_inds]
      mq_feat = feat[mq_inds]
      unique_mq_ids_cams = defaultdict(list)
      for ind, (id, cam) in enumerate(zip(mq_ids, mq_cams)):
        unique_mq_ids_cams[(id, cam)].append(ind)
      keys = unique_mq_ids_cams.keys()
      assert pool_type in ['average', 'max']
      pool = np.mean if pool_type == 'average' else np.max
      mq_feat = np.stack([pool(mq_feat[unique_mq_ids_cams[k]], axis=0)
                          for k in keys])

      with measure_time('Multi Query, Computing distance...', verbose=verbose):
        # multi_query-gallery distance
        mq_g_dist = compute_dist(mq_feat, feat[g_inds], type='euclidean')

      with measure_time('Multi Query, Computing scores...', verbose=verbose):
        mq_mAP, mq_cmc_scores = compute_score(
          mq_g_dist,
          query_ids=np.array(zip(*keys)[0]),
          gallery_ids=ids[g_inds],
          query_cams=np.array(zip(*keys)[1]),
          gallery_cams=cams[g_inds]
        )

      print('{:<30}'.format('Multi Query:'), end='')
      print_scores(mq_mAP, mq_cmc_scores)

    if to_re_rank:

      ##########################
      # Re-ranked Single Query #
      ##########################

      with measure_time('Re-ranking distance...', verbose=verbose):
        # query-query distance
        q_q_dist = compute_dist(feat[q_inds], feat[q_inds], type='euclidean')
        # gallery-gallery distance
        g_g_dist = compute_dist(feat[g_inds], feat[g_inds], type='euclidean')
        # re-ranked query-gallery distance
        re_r_q_g_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

      with measure_time('Computing scores for re-ranked distance...',
                        verbose=verbose):
        mAP, cmc_scores = compute_score(re_r_q_g_dist)

      print('{:<30}'.format('Re-ranked Single Query:'), end='')
      print_scores(mAP, cmc_scores)

      #########################
      # Re-ranked Multi Query #
      #########################

      if any(mq_inds):
        with measure_time('Multi Query, Re-ranking distance...',
                          verbose=verbose):
          # multi_query-multi_query distance
          mq_mq_dist = compute_dist(mq_feat, mq_feat, type='euclidean')
          # re-ranked multi_query-gallery distance
          re_r_mq_g_dist = re_ranking(mq_g_dist, mq_mq_dist, g_g_dist)

        with measure_time(
            'Multi Query, Computing scores for re-ranked distance...',
            verbose=verbose):
          mq_mAP, mq_cmc_scores = compute_score(
            re_r_mq_g_dist,
            query_ids=np.array(zip(*keys)[0]),
            gallery_ids=ids[g_inds],
            query_cams=np.array(zip(*keys)[1]),
            gallery_cams=cams[g_inds]
          )

        print('{:<30}'.format('Re-ranked Multi Query:'), end='')
        print_scores(mq_mAP, mq_cmc_scores)

    return mAP, cmc_scores, mq_mAP, mq_cmc_scores
