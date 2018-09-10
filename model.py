""" --------------------------------------------------
    author: arthur meyer
    email: arthur.meyer.38@gmail.com  
    status: final
    version: v2.0
    --------------------------------------------------"""



from __future__ import division
import tensorflow as tf
import numpy as np



class MODEL(object):
  """ 
  Model description:
    conv          :   vgg
    deconv        :   vgg + 1 more
    fc layer      :   2
    loss          :   flexible
    direct 
     connections  :   flexible (if yes then 111 110)
    edge contrast :   flexible
  """
  
  def __init__(self, name, batch_size, learning_rate, wd, concat, l2_loss, penalty, coef):
    """
    Args:
       name             : name of the model (used to create a specific folder to save/load parameters)
       batch_size       : batch size
       learning_rate    : learning_rate
       wd               : weight decay factor
       concat           : does this model include direct connections?
       l2_loss          : does this model use l2 loss (if not then cross entropy)
       penalty          : whether to use the edge contrast penalty
       coef             : coef for the edge contrast penalty
    """

    self.name                   =        'saliency_' + name
    self.losses                 =        'loss_of_' + self.name
    self.losses_decay           =        'loss_of_' + self.name +'_decay'
    self.batch_size             =        batch_size
    self.learning_rate          =        learning_rate
    self.wd                     =        wd
    self.moving_avg_decay       =        0.9999
    self.concat                 =        concat
    self.smooth                 =        1.
    self.l2_loss                =        l2_loss
    self.penalty                =        penalty
    self.coef                   =        coef
    self.parameters_conv        =        []    
    self.parameters_deconv      =        []  
    self.deconv                 =        []  
    
    with tf.device('/cpu:0'):

      # conv0_0  0
      with tf.variable_scope(self.name + '_' + 'conv0_0') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 3, 48), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [48], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]


      # conv1_1 1.1
      with tf.variable_scope(self.name + '_' + 'db1_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 48, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # conv1_1 1.2
      with tf.variable_scope(self.name + '_' + 'db1_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 64, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 1.3
      with tf.variable_scope(self.name + '_' + 'db1_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 80, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 1.4
      with tf.variable_scope(self.name + '_' + 'db1_4') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 96, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      with tf.variable_scope(self.name + '_' + 'td1') as scope:
        kernel        = tf.get_variable('kernel', (1, 1, 112, 112), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [112], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 2.1
      with tf.variable_scope(self.name + '_' + 'db2_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 112, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 2.2
      with tf.variable_scope(self.name + '_' + 'db2_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 128, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 2.3
      with tf.variable_scope(self.name + '_' + 'db2_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 144, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 2.4
      with tf.variable_scope(self.name + '_' + 'db2_4') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 160, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 2.5
      with tf.variable_scope(self.name + '_' + 'db2_5') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 176, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      with tf.variable_scope(self.name + '_' + 'td2') as scope:
        kernel        = tf.get_variable('kernel', (1, 1, 192, 192), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB  3.1
      with tf.variable_scope(self.name + '_' + 'db3_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 192, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 3.2
      with tf.variable_scope(self.name + '_' + 'db3_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 208, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB  3.3
      with tf.variable_scope(self.name + '_' + 'db3_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 224, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]


      # DB 3.4
      with tf.variable_scope(self.name + '_' + 'db3_4') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 240, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 3.5
      with tf.variable_scope(self.name + '_' + 'db3_5') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 256, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 3.6
      with tf.variable_scope(self.name + '_' + 'db3_6') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 272, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 3.7
      with tf.variable_scope(self.name + '_' + 'db3_7') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 288, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      with tf.variable_scope(self.name + '_' + 'td3') as scope:
        kernel        = tf.get_variable('kernel', (1, 1, 304, 304), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [304], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]
      # DB  4.1
      with tf.variable_scope(self.name + '_' + 'db4_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 304, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.2
      with tf.variable_scope(self.name + '_' + 'db4_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 320, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB  4.3
      with tf.variable_scope(self.name + '_' + 'db4_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 336, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.4
      with tf.variable_scope(self.name + '_' + 'db4_4') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 352, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.5
      with tf.variable_scope(self.name + '_' + 'db4_5') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 368, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.6
      with tf.variable_scope(self.name + '_' + 'db4_6') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 384, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.7
      with tf.variable_scope(self.name + '_' + 'db4_7') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 400, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.8
      with tf.variable_scope(self.name + '_' + 'db4_8') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 416, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.9
      with tf.variable_scope(self.name + '_' + 'db4_9') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 432, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 4.10
      with tf.variable_scope(self.name + '_' + 'db4_10') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 448, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      with tf.variable_scope(self.name + '_' + 'td4') as scope:
        kernel        = tf.get_variable('kernel', (1, 1, 464, 464), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [464], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]
      # DB  5.1
      with tf.variable_scope(self.name + '_' + 'db5_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 464, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.2
      with tf.variable_scope(self.name + '_' + 'db5_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 480, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB  5.3
      with tf.variable_scope(self.name + '_' + 'db5_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 496, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.4
      with tf.variable_scope(self.name + '_' + 'db5_4') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 512, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.5
      with tf.variable_scope(self.name + '_' + 'db5_5') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 528, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.6
      with tf.variable_scope(self.name + '_' + 'db5_6') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 544, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.7
      with tf.variable_scope(self.name + '_' + 'db5_7') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 560, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.8
      with tf.variable_scope(self.name + '_' + 'db5_8') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 576, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.9
      with tf.variable_scope(self.name + '_' + 'db5_9') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 592, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.10
      with tf.variable_scope(self.name + '_' + 'db5_10') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 608, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]


      # DB 5.11
      with tf.variable_scope(self.name + '_' + 'db5_11') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 624, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 5.12
      with tf.variable_scope(self.name + '_' + 'db5_12') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 640, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      with tf.variable_scope(self.name + '_' + 'td5') as scope:
        kernel        = tf.get_variable('kernel', (1, 1, 656, 656), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [656], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB  6.1
      with tf.variable_scope(self.name + '_' + 'db6_1') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 656, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.2
      with tf.variable_scope(self.name + '_' + 'db6_2') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 672, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB  6.3
      with tf.variable_scope(self.name + '_' + 'db6_3') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 688, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.4
      with tf.variable_scope(self.name + '_' + 'db6_4') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 704, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses_decay, weight_decay)
        tf.add_to_collection(self.losses, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.5
      with tf.variable_scope(self.name + '_' + 'db6_5') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 720, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.6
      with tf.variable_scope(self.name + '_' + 'db6_6') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 736, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.7
      with tf.variable_scope(self.name + '_' + 'db6_7') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 752, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.8
      with tf.variable_scope(self.name + '_' + 'db6_8') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 768, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.9
      with tf.variable_scope(self.name + '_' + 'db6_9') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 784, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.10
      with tf.variable_scope(self.name + '_' + 'db6_10') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 800, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]


      # DB 6.11
      with tf.variable_scope(self.name + '_' + 'db6_11') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 816, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.12
      with tf.variable_scope(self.name + '_' + 'db6_12') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 832, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.13
      with tf.variable_scope(self.name + '_' + 'db6_13') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 848, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.14
      with tf.variable_scope(self.name + '_' + 'db6_14') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 864, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      # DB 6.15
      with tf.variable_scope(self.name + '_' + 'db6_15') as scope:
        kernel        = tf.get_variable('kernel', (3, 3, 880, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      with tf.variable_scope(self.name + '_' + 'tu1') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 240, 240), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [240], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]


      #DDB 1.1
      with tf.variable_scope(self.name + '_' + 'ddb1_1') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 896, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.2
      with tf.variable_scope(self.name + '_' + 'ddb1_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 912, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.3
      with tf.variable_scope(self.name + '_' + 'ddb1_3') as scope:
        kernel = tf.get_variable('kernel', (3, 3, 928, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.4
      with tf.variable_scope(self.name + '_' + 'ddb1_4') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 944, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.5
      with tf.variable_scope(self.name + '_' + 'ddb1_5') as scope:
        kernel       = tf.get_variable('kernel', (3, 3,960, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.6
      with tf.variable_scope(self.name + '_' + 'ddb1_6') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 976, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.7
      with tf.variable_scope(self.name + '_' + 'ddb1_7') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 992, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.8
      with tf.variable_scope(self.name + '_' + 'ddb1_8') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 1008, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.9
      with tf.variable_scope(self.name + '_' + 'ddb1_9') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 1024, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.10
      with tf.variable_scope(self.name + '_' + 'ddb1_10') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 1040, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.11
      with tf.variable_scope(self.name + '_' + 'ddb1_11') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 1056, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 1.12
      with tf.variable_scope(self.name + '_' + 'ddb1_12') as scope:
        kernel       = tf.get_variable('kernel', (3, 3,1072, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      with tf.variable_scope(self.name + '_' + 'tu2') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 192, 192), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]



      #DDB  2.1
      with tf.variable_scope(self.name + '_' + 'ddb2_1') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 656, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.2
      with tf.variable_scope(self.name + '_' + 'ddb2_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 672, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.3
      with tf.variable_scope(self.name + '_' + 'ddb2_3') as scope:
        kernel = tf.get_variable('kernel', (3, 3, 688, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.4
      with tf.variable_scope(self.name + '_' + 'ddb2_4') as scope:
        kernel       = tf.get_variable('kernel', (3, 3,704, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.5
      with tf.variable_scope(self.name + '_' + 'ddb2_5') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 720, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.6
      with tf.variable_scope(self.name + '_' + 'ddb2_6') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 736, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.7
      with tf.variable_scope(self.name + '_' + 'ddb2_7') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 752, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.8
      with tf.variable_scope(self.name + '_' + 'ddb2_8') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 768, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.9
      with tf.variable_scope(self.name + '_' + 'ddb2_9') as scope:
        kernel       = tf.get_variable('kernel', (3, 3,784, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 2.10
      with tf.variable_scope(self.name + '_' + 'ddb2_10') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 800, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      with tf.variable_scope(self.name + '_' + 'tu3') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 160, 160), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [160], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #3.1
      with tf.variable_scope(self.name + '_' + 'ddb3_1') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 464, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 3.2
      with tf.variable_scope(self.name + '_' + 'ddb3_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 480, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 3.3
      with tf.variable_scope(self.name + '_' + 'ddb3_3') as scope:
        kernel = tf.get_variable('kernel', (3, 3, 496, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 3.4
      with tf.variable_scope(self.name + '_' + 'ddb3_4') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 512, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 3.5
      with tf.variable_scope(self.name + '_' + 'ddb3_5') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 528, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 3.6
      with tf.variable_scope(self.name + '_' + 'ddb3_6') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 544, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 3.7
      with tf.variable_scope(self.name + '_' + 'ddb3_7') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 560, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      with tf.variable_scope(self.name + '_' + 'tu4') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 112, 112), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [112], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
      #DDB  4.1
      with tf.variable_scope(self.name + '_' + 'ddb4_1') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 304, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 4.2
      with tf.variable_scope(self.name + '_' + 'ddb4_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3,320, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 4.3
      with tf.variable_scope(self.name + '_' + 'ddb4_3') as scope:
        kernel = tf.get_variable('kernel', (3, 3, 336, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 4.4
      with tf.variable_scope(self.name + '_' + 'ddb4_4') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 352, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 4.5
      with tf.variable_scope(self.name + '_' + 'ddb4_5') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 368, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
        
      with tf.variable_scope(self.name + '_' + 'tu5') as scope:
        kernel     = tf.get_variable('kernel', (3, 3, 80, 80), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [80], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
      #DDB 5.1
      with tf.variable_scope(self.name + '_' + 'ddb5_1') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 192, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 5.2
      with tf.variable_scope(self.name + '_' + 'ddb5_2') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 208, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]


      #DDB 5.3
      with tf.variable_scope(self.name + '_' + 'ddb5_3') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 224, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]

      #DDB 5.4
      with tf.variable_scope(self.name + '_' + 'ddb5_4') as scope:
        kernel       = tf.get_variable('kernel', (3, 3, 240, 16), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases       = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
      # conv1_1  6
      with tf.variable_scope(self.name + '_' + 'conv1_1') as scope:
        kernel        = tf.get_variable('kernel', (1, 1, 256, 1), initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32), dtype=tf.float32)
        biases        = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0), dtype=tf.float32)
        weight_decay  = tf.multiply(tf.nn.l2_loss(kernel), self.wd)
        tf.add_to_collection(self.losses, weight_decay)
        tf.add_to_collection(self.losses_decay, weight_decay)
        self.parameters_conv += [kernel, biases]
        
  def display_info(self, verbosity):
    """
    Display information about this model
    
    Args:
      verbosity : level of details to display
    """
    
    print('------------------------------------------------------')  
    print('This model is %s' % (self.name))
    print('------------------------------------------------------')
    if verbosity > 0:
      print('Learning rate: %0.8f -- Weight decay: %0.8f -- Cross entropy loss: %r' % (self.learning_rate , self.wd, not self.l2_loss))
      print('------------------------------------------------------')
    print('Direct connections: %r' % (self.concat))
    print('------------------------------------------------------')
    print('Edge contrast penalty: %r -- coefficient %0.5f' % (self.penalty, self.coef))
    print('------------------------------------------------------\n')
  
  
    
    
      
  def infer(self, images, inter_layer = False, arithmetic = None, debug = False):
    """
    Return saliency map from given images
    
    Args:
      images          : input images
      inter_layer     : whether we want to return the middle layer code
      arithmetic      : type of special operation on the middle layer encoding (1 is add, 2 subtract, 3 is linear combination)
      debug           : whether to return a extra value use for debug (control value)
      
    Returns:
      out             : saliency maps of the input
      control_value   : some value used to debug training
      inter_layer_out : value of the middle layer
    """

    control_value   = None
    inter_layer_out = None 

    if self.concat:
      detail      =  []
      detail_bis  =  []
      detail      += [tf.image.resize_images(images,[112,112])]
      detail_bis  += [images]
    
    # conv0_0
    with tf.variable_scope(self.name + '_' + 'conv0_0') as scope:
      conv = tf.nn.conv2d(images, self.parameters_conv[0], [1, 1, 1, 1], padding='SAME')
      out  = tf.nn.bias_add(conv, self.parameters_conv[1])      
      
    # DB 1.1
    with tf.variable_scope(self.name + '_' + 'db1_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[2], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[3])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db1_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[4], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[5])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db1_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[6], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[7])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db1_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[8], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[9])
      out=tf.concat([out,l],3)
    
    skip1=out

    with tf.variable_scope(self.name + '_' + 'td1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[10], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[11])

    out = tf.nn.max_pool(l,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

    # DB 2.1
    with tf.variable_scope(self.name + '_' + 'db2_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[12], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[13])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db2_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[14], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[15])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db2_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[16], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[17])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db2_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[18], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[19])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db2_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[20], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[21])
      out=tf.concat([out,l],3)

    skip2=out

    with tf.variable_scope(self.name + '_' + 'td2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[22], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[23])

    out = tf.nn.max_pool(l,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')

    # DB 3.1
    with tf.variable_scope(self.name + '_' + 'db3_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[24], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[25])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db3_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[26], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[27])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db3_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[28], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[29])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db3_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[30], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[31])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db3_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[32], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[33])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db3_6') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[34], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[35])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db3_7') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[36], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[37])
      out=tf.concat([out,l],3)

    skip3=out

    with tf.variable_scope(self.name + '_' + 'td3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[38], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[39])

    out = tf.nn.max_pool(l,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool3')

    # DB 4.1
    with tf.variable_scope(self.name + '_' + 'db4_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[40], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[41])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[42], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[43])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[44], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[45])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[46], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[47])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[48], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[49])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_6') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[50], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[51])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_7') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[52], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[53])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_8') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[54], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[55])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_9') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[56], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[57])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db4_10') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[58], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[59])
      out=tf.concat([out,l],3)

    skip4=out

    with tf.variable_scope(self.name + '_' + 'td4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[60], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[61])

    out = tf.nn.max_pool(l,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')


    # DB 5.1
    with tf.variable_scope(self.name + '_' + 'db5_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[62], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[63])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[64], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[65])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[66], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[67])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[68], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[69])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[70], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[71])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_6') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[72], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[73])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_7') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[74], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[75])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_8') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[76], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[77])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_9') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[78], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[79])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_10') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[80], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[81])
      out=tf.concat([out,l],3)
    with tf.variable_scope(self.name + '_' + 'db5_11') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[82], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[83])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'db5_12') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[84], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[85])
      out=tf.concat([out,l],3)

    skip5=out

    with tf.variable_scope(self.name + '_' + 'td5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[86], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[87])

    out = tf.nn.max_pool(l,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool5')

    # DB 6.1
    with tf.variable_scope(self.name + '_' + 'db6_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[88], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[89])
      out=tf.concat([out,l],3)
      new=l

    with tf.variable_scope(self.name + '_' + 'db6_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[90], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[91])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[92], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[93])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[94], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[95])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[96], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[97])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_6') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[98], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[99])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_7') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[100], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[101])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_8') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[102], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[103])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_9') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[104], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[105])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_10') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[106], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[107])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_11') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[108], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[109])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_12') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[110], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[111])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_13') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[112], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[113])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_14') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[114], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[115])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'db6_15') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[116], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[117])
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'tu1') as scope:
      deconv =  tf.nn.conv2d_transpose(new, self.parameters_conv[118], (self.batch_size,14,14,240), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_conv[119])

    out=tf.concat([skip5,bias],3)

    # DDB 1.1
    with tf.variable_scope(self.name + '_' + 'ddb1_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[120], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[121])
      out=tf.concat([out,l],3)
      new=l

    with tf.variable_scope(self.name + '_' + 'ddb1_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[122], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[123])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)


    with tf.variable_scope(self.name + '_' + 'ddb1_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[124], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[125])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[126], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[127])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[128], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[129])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_6') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[130], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[131])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_7') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[132], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[133])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_8') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[134], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[135])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_9') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[136], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[137])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_10') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[138], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[139])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_11') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[140], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[141])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb1_12') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[142], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[143])
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'tu2') as scope:
      deconv =  tf.nn.conv2d_transpose(new, self.parameters_conv[144], (self.batch_size,28,28,192), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_conv[145])

    out=tf.concat([skip4,bias],3)

    # DDB 2.1
    with tf.variable_scope(self.name + '_' + 'ddb2_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[146], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[147])
      out=tf.concat([out,l],3)
      new=l

    with tf.variable_scope(self.name + '_' + 'ddb2_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[148], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[149])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[150], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[151])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[152], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[153])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[154], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[155])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_6') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[156], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[157])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_7') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[158], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[159])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_8') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[160], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[161])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_9') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[162], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[163])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb2_10') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[164], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[165])
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'tu3') as scope:
      deconv =  tf.nn.conv2d_transpose(new, self.parameters_conv[166], (self.batch_size,56,56,160), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_conv[167])

    out=tf.concat([skip3,bias],3)


    # DDB 3.1
    with tf.variable_scope(self.name + '_' + 'ddb3_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[168], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[169])
      out=tf.concat([out,l],3)
      new=l

    with tf.variable_scope(self.name + '_' + 'ddb3_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[170], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[171])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb3_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[172], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[173])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb3_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[174], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[175])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb3_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[176], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[177])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb3_6') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[178], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[179])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb3_7') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[180], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[181])
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'tu4') as scope:
      deconv =  tf.nn.conv2d_transpose(new, self.parameters_conv[182], (self.batch_size,112,112,112), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_conv[183])

    out=tf.concat([skip2,bias],3)

    # DDB 4.1
    with tf.variable_scope(self.name + '_' + 'ddb4_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[184], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[185])
      out=tf.concat([out,l],3)
      new=l

    with tf.variable_scope(self.name + '_' + 'ddb4_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[186], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[187])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb4_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[188], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[189])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb4_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[190], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[191])
      out=tf.concat([out,l],3)
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb4_5') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[192], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[193])
      new=tf.concat([new,l],3)

    with tf.variable_scope(self.name + '_' + 'tu5') as scope:
      deconv =  tf.nn.conv2d_transpose(new, self.parameters_conv[194], (self.batch_size,224,224,80), strides= [1, 2, 2, 1], padding='SAME')
      bias   =  tf.nn.bias_add(deconv, self.parameters_conv[195])

    out=tf.concat([skip1,bias],3)

    # DDB 5.1
    with tf.variable_scope(self.name + '_' + 'ddb5_1') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[196], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[197])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb5_2') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[198], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[199])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb5_3') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[200], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[201])
      out=tf.concat([out,l],3)

    with tf.variable_scope(self.name + '_' + 'ddb5_4') as scope:
      norm = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
      relu = tf.nn.relu(norm)
      conv = tf.nn.conv2d(relu, self.parameters_conv[202], [1, 1, 1, 1], padding='SAME')
      l  = tf.nn.bias_add(conv, self.parameters_conv[203])
      out=tf.concat([out,l],3)

    #conv1
    with tf.variable_scope(self.name + '_' + 'conv1_1') as scope:
      conv =  tf.nn.conv2d(out, self.parameters_conv[204], [1, 1, 1, 1], padding='SAME')
      l    = tf.nn.bias_add(conv, self.parameters_conv[205])
      relu   =  tf.sigmoid(l)
      out = tf.squeeze(relu)

      
    if debug:
      control_value = tf.reduce_mean(relu)
      
    return out, control_value, inter_layer_out



  
  
  def loss(self, guess, labels, loss_bis = False):
    """
    Return the loss for given saliency map with corresponding ground truth
    
    Args:
      guess    :    input saliency map
      labels   :    corresponding ground truth
      loss_bis :    is it the main loss or the auxiliary one (for validation while training)
      
    Returns:
      loss_out :    the loss value
    """
    
    if self.l2_loss:
      reconstruction      = tf.reduce_sum(tf.square(guess - labels), [1,2])
      reconstruction_mean = tf.reduce_mean(reconstruction)
      if not loss_bis:
        tf.add_to_collection(self.losses, reconstruction_mean)
    else:
      guess_flat  = tf.reshape(guess,  [self.batch_size, -1])
      labels_flat = tf.reshape(labels, [self.batch_size, -1])
      zero        = tf.fill(tf.shape(guess_flat), 1e-7)
      one         = tf.fill(tf.shape(guess_flat), 1 - 1e-7)
      ret_1       = tf.where(guess_flat > 1e-7, guess_flat, zero)
      ret_2       = tf.where(ret_1 < 1 - 1e-7, ret_1, one)
      product     = tf.multiply(ret_2,labels_flat)
      intersection= tf.reduce_sum(product)
      coefficient = (0.3*intersection +self.smooth) / (tf.reduce_sum(ret_2)+tf.reduce_sum(labels_flat)*0.3 +self.smooth)
      loss        = 1.- coefficient
      # loss        = tf.reduce_mean(- labels_flat * tf.log(ret_2) - (1. - labels_flat) * tf.log(1. - ret_2))
      if not loss_bis:
        tf.add_to_collection(self.losses, loss)
      elif loss_bis:
        tf.add_to_collection(self.losses_decay, loss)
      
    if self.penalty and not loss_bis:
      labels_new   = tf.reshape(labels, [self.batch_size, 224, 224, 1])
      guess_new    = tf.reshape(guess, [self.batch_size, 224, 224, 1])
      filter_x     = tf.constant(np.array([[0,0,0] , [-1,2,-1], [0,0,0]]).reshape((3,3,1,1)), dtype=tf.float32)
      filter_y     = tf.constant(np.array([[0,-1,0] , [0,2,0], [0,-1,0]]).reshape((3,3,1,1)), dtype=tf.float32)
      gradient_x   = tf.nn.conv2d(labels_new, filter_x, [1,1,1,1], padding = "SAME")
      gradient_y   = tf.nn.conv2d(labels_new, filter_y, [1,1,1,1], padding = "SAME")
      result_x     = tf.greater(gradient_x,0)
      result_y     = tf.greater(gradient_y,0)
      keep         = tf.cast(tf.logical_or(result_x,result_y), tf.float32)  #edges

      filter_neighboor_1 = tf.constant(np.array([[0,0,0], [0,1,-1], [0,0,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor_2 = tf.constant(np.array([[0,-1,0], [0,1,0], [0,0,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor_3 = tf.constant(np.array([[0,0,0], [-1,1,0], [0,0,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor_4 = tf.constant(np.array([[0,0,0], [0,1,0], [0,-1,0]]).reshape((3,3,1)), dtype=tf.float32)
      filter_neighboor   = tf.stack([filter_neighboor_1,filter_neighboor_2,filter_neighboor_3,filter_neighboor_4], axis = 3)
      compare            = tf.square(keep * tf.nn.conv2d(guess_new, filter_neighboor, [1,1,1,1], padding = "SAME"))

      compare_m       = tf.nn.conv2d(labels_new, filter_neighboor, [1,1,1,1], padding = "SAME")
      new_compare_m   = tf.where(tf.equal(compare_m, 0), tf.ones([self.batch_size,224,224,4]), -1*tf.ones([self.batch_size,224,224,4])) #0 mean same so want to minimize and if not then diff so want to maximize
      final_compare_m = keep * new_compare_m
      
      score_ret = tf.reduce_sum(final_compare_m * compare, [1,2,3]) / (4*(tf.reduce_sum(keep,[1,2,3])+1e-7))
      score     = self.coef * tf.reduce_mean(score_ret)
      tf.add_to_collection(self.losses, score)
    
    if loss_bis:
      loss_out = tf.add_n(tf.get_collection(self.losses_decay))
    else:
      loss_out = tf.add_n(tf.get_collection(self.losses))
    
    return loss_out

  

  
  
  def train(self, loss, global_step):
    """
    Return a training step for the tensorflow graph
    
    Args:
      loss                   : loss to do sgd on
      global_step            : which step are we at
    """

    opt = tf.train.AdamOptimizer(self.learning_rate)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    variable_averages = tf.train.ExponentialMovingAverage(self.moving_avg_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')
  
    return train_op