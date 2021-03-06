import numpy as np
import pandas as pd 
import os
from pathlib import Path
import tensorflow.compat.v1 as tf


def get_image_paths(directory):
    """
    Takes in directory, returns list of image paths
    """
    image_paths = list(Path(directory).glob("*.jpg")) + list(Path(directory).glob("*.png"))#os.listdir(directory)
    #image_paths = [os.path.join(directory, i) for i in image_names if i.split('.')[1] in ['jpg','png']]
    if len(image_paths) == 0:
        return 'no jpg or png images found in specified directory'
    return image_paths

def get_image_number(image_path):
    """
    Extracts img number from path 
    """
    image_num = image_path.name.split('.')[0]
    return image_num

def load_image_into_numpy_array(image):
    """
    Image to numpy array
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  """
  Run trained object detection model on image
  Tensor flow object detection API: github.com/tensorflow/models/tree/master/research/object_detection
  """
  with graph.as_default():
     with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def compute_viability(df,image_path):
    """
    Computes percent viability for output df
    """
    class_counts = df['detection_classes'].value_counts()
    total_count  = class_counts.sum()
    viability = class_counts[1]/total_count*100
    viability = {'germinated_count':class_counts[1],
                 'ungerminated_count':class_counts[2],
                 'total_count': total_count,
                 'percent_viability': viability,
                 'parent_image': image_path.name.split('_')[0],
                 'image_path': image_path}
    return viability