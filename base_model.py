# Copied from https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/base_model.py

# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

from magenta.contrib import training as contrib_training
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tf_slim

ds = tfp.distributions


class BaseEncoder(object, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def output_depth(self):
        pass

    @abc.abstractmethod
    def build(self, hparams, is_training=True):
        pass

    @abc.abstractmethod
    def encode(self, sequence, sequence_length):
        pass


class BaseDecoder(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build(self, hparams, output_depth, is_training=True):
        pass

    @abc.abstractmethod
    def reconstruction_loss(self, x_input, x_target, x_length, z=None, c_input=None):
        pass

    @abc.abstractmethod
    def sample(self, n, max_length=None, z=None, c_input=None):
        pass


class MusicVAE(object):

    def __init__(self, encoder, decoder):
        self._encoder = encoder
        self._decoder = decoder

    def build(self, hparams, output_depth, is_training):
        tf.logging.info('Building MusicVAE model with %s, %s, and hparams:\n%s',
                        self.encoder.__class__.__name__,
                        self.decoder.__class__.__name__, hparams.values())
        self.global_step = tf.train.get_or_create_global_step()
        self._hparams = hparams
        self._encoder.build(hparams, is_training)
        self._decoder.build(hparams, output_depth, is_training)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def hparams(self):
        return self._hparams

    def encode(self, sequence, sequence_length, control_sequence=None):
        """인코더가 입력 데이터를 입력받아 mu와 sigma를 계산하는 함수"""
        hparams = self.hparams
        z_size = hparams.z_size

        sequence = tf.to_float(sequence)
        if control_sequence is not None:
            control_sequence = tf.to_float(control_sequence)
            sequence = tf.concat([sequence, control_sequence], axis=-1)
        encoder_output = self.encoder.encode(sequence, sequence_length)

        mu = tf.layers.dense(
            encoder_output,
            z_size,
            name='encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))
        sigma = tf.layers.dense(
            encoder_output,
            z_size,
            activation=tf.nn.softplus,
            name='encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))

        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def _compute_model_loss(self, input_sequence, output_sequence, sequence_length, control_sequence):
        """forward 후 loss 계산"""
        hparams = self.hparams
        batch_size = hparams.batch_size

        input_sequence = tf.to_float(input_sequence)
        output_sequence = tf.to_float(output_sequence)

        max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)
        input_sequence = input_sequence[:, :max_seq_len]

        if control_sequence is not None:
            control_depth = control_sequence.shape[-1]
            control_sequence = tf.to_float(control_sequence)
            control_sequence = control_sequence[:, :max_seq_len]
            control_sequence.set_shape([batch_size, None, control_depth])

        # decoder input, target 전처리
        x_target = output_sequence[:, :max_seq_len]
        x_input = tf.pad(output_sequence[:, :max_seq_len - 1], [(0, 0), (1, 0), (0, 0)])
        x_length = tf.minimum(sequence_length, max_seq_len)


        if hparams.z_size:
            # latent vector 추출
            q_z = self.encode(input_sequence, x_length, control_sequence)
            z = q_z.sample() # 

            # 사전분포 p(z)
            p_z = ds.MultivariateNormalDiag(loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

            # regularization term in ELBO 
            kl_div = ds.kl_divergence(q_z, p_z)


        else:
            kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
            z = None

        # decoder forward 및 loss 계산
        # reconstruction term in ELBO
        r_loss, metric_map = self.decoder.reconstruction_loss(x_input, x_target, x_length, z, control_sequence)[0:2]

        # free bits in Eq.4
        # threshold 이상인 경우만 kl_loss 적용
        free_nats = hparams.free_bits * tf.math.log(2.0)
        kl_cost = tf.maximum(kl_div - free_nats, 0)

        # beta-VAE in Eq.3
        # beta(beta < 1)를 곱함으로써 kl_loss 비중 감소
        beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step))) * hparams.max_beta)
        self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)

        scalars_to_summarize = {
            'loss': self.loss,
            'losses/r_loss': r_loss,
            'losses/kl_loss': kl_cost,
            'losses/kl_bits': kl_div / tf.math.log(2.0),
            'losses/kl_beta': beta,
        }
        return metric_map, scalars_to_summarize


    def train(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        _, scalars_to_summarize = self._compute_model_loss(input_sequence, output_sequence, sequence_length, control_sequence)

        hparams = self.hparams
        lr = ((hparams.learning_rate - hparams.min_learning_rate) *
              tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
              hparams.min_learning_rate)

        optimizer = tf.train.AdamOptimizer(lr)

        tf.summary.scalar('learning_rate', lr)
        for n, t in scalars_to_summarize.items():
            tf.summary.scalar(n, tf.reduce_mean(t))

        return optimizer


    def eval(self, input_sequence, output_sequence, sequence_length, control_sequence=None):
        metric_map, scalars_to_summarize = self._compute_model_loss(input_sequence, output_sequence, sequence_length, control_sequence)

        for n, t in scalars_to_summarize.items():
            metric_map[n] = tf.metrics.mean(t)

        metrics_to_values, metrics_to_updates = (
            tf_slim.metrics.aggregate_metric_map(metric_map))

        for metric_name, metric_value in metrics_to_values.items():
            tf.summary.scalar(metric_name, metric_value)

        return list(metrics_to_updates.values())


    def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
        if z is not None and int(z.shape[0]) != n:
            raise ValueError(
                '`z` must have a first dimension that equals `n` when given. '
                'Got: %d vs %d' % (z.shape[0], n))

        if self.hparams.z_size and z is None:
            tf.logging.warning('Sampling from conditional model without `z`. Using random `z`.')
            normal_shape = [n, self.hparams.z_size]
            normal_dist = tfp.distributions.Normal(
                loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape))
            z = normal_dist.sample()

        return self.decoder.sample(n, max_length, z, c_input, **kwargs)


def get_default_hparams():
    return contrib_training.HParams(
        max_seq_len=32,  # Maximum sequence length. Others will be truncated.
        z_size=32,  # Size of latent vector z.
        free_bits=0.0,  # Bits to exclude from KL loss per dimension.
        max_beta=1.0,  # Maximum KL cost weight, or cost if not annealing.
        beta_rate=0.0,  # Exponential rate at which to anneal KL cost.
        batch_size=512,  # Minibatch size.
        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
        clip_mode='global_norm',  # value or global_norm.
        grad_norm_clip_to_zero=10000,
        learning_rate=0.001,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.00001,  # Minimum learning rate.
    )
