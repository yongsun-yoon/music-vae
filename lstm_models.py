# Copied from https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/lstm_models.py

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

"""LSTM-based encoders and decoders for MusicVAE."""
import abc

from magenta.common import flatten_maybe_padded_sequences
from magenta.common import Nade
import magenta.contrib.rnn as contrib_rnn
import magenta.contrib.seq2seq as contrib_seq2seq
import magenta.contrib.training as contrib_training
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import base_model
import lstm_utils


# ENCODERS
class BidirectionalLstmEncoder(base_model.BaseEncoder):
    """Bidirectional LSTM Encoder."""

    @property
    def output_depth(self):
        return self._cells[0][-1].output_size + self._cells[1][-1].output_size

    def build(self, hparams, is_training=True, name_or_scope='encoder'):
        self._is_training = is_training
        self._name_or_scope = name_or_scope
        if hparams.use_cudnn:
            tf.logging.warning(
                'cuDNN LSTM no longer supported. Using regular LSTM.')

        tf.logging.info('\nEncoder Cells (bidirectional):\n'
                        '  units: %s\n',
                        hparams.enc_rnn_size)

        self._cells = lstm_utils.build_bidirectional_lstm(
            layer_sizes=hparams.enc_rnn_size,
            dropout_keep_prob=hparams.dropout_keep_prob,
            residual=hparams.residual_encoder,
            is_training=is_training)

    def encode(self, sequence, sequence_length):
        cells_fw, cells_bw = self._cells

        _, states_fw, states_bw = contrib_rnn.stack_bidirectional_dynamic_rnn(
            cells_fw,
            cells_bw,
            sequence,
            sequence_length=sequence_length,
            time_major=False,
            dtype=tf.float32,
            scope=self._name_or_scope)
        # Note we access the outputs (h) from the states since the backward
        # ouputs are reversed to the input order in the returned outputs.
        last_h_fw = states_fw[-1][-1].h
        last_h_bw = states_bw[-1][-1].h

        return tf.concat([last_h_fw, last_h_bw], 1)


# DECODERS
class BaseLstmDecoder(base_model.BaseDecoder):
    """Abstract LSTM Decoder class.

    Implementations must define the following abstract methods:
        -`_sample`
        -`_flat_reconstruction_loss`
    """

    def build(self, hparams, output_depth, is_training=True):
        if hparams.use_cudnn:
            tf.logging.warning(
                'cuDNN LSTM no longer supported. Using regular LSTM.')

        self._is_training = is_training

        tf.logging.info('\nDecoder Cells:\n'
                        '  units: %s\n',
                        hparams.dec_rnn_size)

        self._sampling_probability = lstm_utils.get_sampling_probability(
            hparams, is_training)
        self._output_depth = output_depth
        self._output_layer = tf.layers.Dense(
            output_depth, name='output_projection')
        self._dec_cell = lstm_utils.rnn_cell(
            hparams.dec_rnn_size, hparams.dropout_keep_prob,
            hparams.residual_decoder, is_training)

    @property
    def state_size(self):
        return self._dec_cell.state_size

    @abc.abstractmethod
    def _sample(self, rnn_output, temperature):
        """Core sampling method for a single time step.

        Args:
          rnn_output: The output from a single timestep of the RNN, sized
              `[batch_size, rnn_output_size]`.
          temperature: A scalar float specifying a sampling temperature.
        Returns:
          A batch of samples from the model.
        """
        pass

    @abc.abstractmethod
    def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
        """Core loss calculation method for flattened outputs.

        Args:
          flat_x_target: The flattened ground truth vectors, sized
            `[sum(x_length), self._output_depth]`.
          flat_rnn_output: The flattened output from all timeputs of the RNN,
            sized `[sum(x_length), rnn_output_size]`.
        Returns:
          r_loss: The unreduced reconstruction losses, sized `[sum(x_length)]`.
          metric_map: A map of metric names to tuples, each of which contain the
            pair of (value_tensor, update_op) from a tf.metrics streaming metric.
        """
        pass

    def _decode(self, z, helper, input_shape, max_length=None):
        """Decodes the given batch of latent vectors vectors, which may be 0-length.

        Args:
          z: Batch of latent vectors, sized `[batch_size, z_size]`, where `z_size`
            may be 0 for unconditioned decoding.
          helper: A seq2seq.Helper to use.
          input_shape: The shape of each model input vector passed to the decoder.
          max_length: (Optional) The maximum iterations to decode.

        Returns:
          results: The LstmDecodeResults.
        """
        initial_state = lstm_utils.initial_cell_state_from_embedding(
            self._dec_cell, z, name='decoder/z_to_initial_state')

        decoder = lstm_utils.Seq2SeqLstmDecoder(
            self._dec_cell,
            helper,
            initial_state=initial_state,
            input_shape=input_shape,
            output_layer=self._output_layer)
        final_output, final_state, final_lengths = contrib_seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=max_length,
            swap_memory=True,
            scope='decoder')
        results = lstm_utils.LstmDecodeResults(
            rnn_input=final_output.rnn_input[:, :, :self._output_depth],
            rnn_output=final_output.rnn_output,
            samples=final_output.sample_id,
            final_state=final_state,
            final_sequence_lengths=final_lengths)

        return results

    def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                            c_input=None):
        """Reconstruction loss calculation.

        Args:
          x_input: Batch of decoder input sequences for teacher forcing, sized
            `[batch_size, max(x_length), output_depth]`.
          x_target: Batch of expected output sequences to compute loss against,
            sized `[batch_size, max(x_length), output_depth]`.
          x_length: Length of input/output sequences, sized `[batch_size]`.
          z: (Optional) Latent vectors. Required if model is conditional. Sized
            `[n, z_size]`.
          c_input: (Optional) Batch of control sequences, sized
              `[batch_size, max(x_length), control_depth]`. Required if conditioning
              on control sequences.

        Returns:
          r_loss: The reconstruction loss for each sequence in the batch.
          metric_map: Map from metric name to tf.metrics return values for logging.
          decode_results: The LstmDecodeResults.
        """
        batch_size = int(x_input.shape[0])

        has_z = z is not None
        z = tf.zeros([batch_size, 0]) if z is None else z
        repeated_z = tf.tile(
            tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

        has_control = c_input is not None
        if c_input is None:
            c_input = tf.zeros([batch_size, tf.shape(x_input)[1], 0])

        sampling_probability_static = tf.get_static_value(
            self._sampling_probability)
        if sampling_probability_static == 0.0:
            # Use teacher forcing.
            x_input = tf.concat([x_input, repeated_z, c_input], axis=2)
            helper = contrib_seq2seq.TrainingHelper(x_input, x_length)
        else:
            # Use scheduled sampling.
            if has_z or has_control:
                auxiliary_inputs = tf.zeros(
                    [batch_size, tf.shape(x_input)[1], 0])
                if has_z:
                    auxiliary_inputs = tf.concat(
                        [auxiliary_inputs, repeated_z], axis=2)
                if has_control:
                    auxiliary_inputs = tf.concat(
                        [auxiliary_inputs, c_input], axis=2)
            else:
                auxiliary_inputs = None
            helper = contrib_seq2seq.ScheduledOutputTrainingHelper(
                inputs=x_input,
                sequence_length=x_length,
                auxiliary_inputs=auxiliary_inputs,
                sampling_probability=self._sampling_probability,
                next_inputs_fn=self._sample)

        decode_results = self._decode(
            z, helper=helper, input_shape=helper.inputs.shape[2:])
        flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
        flat_rnn_output = flatten_maybe_padded_sequences(
            decode_results.rnn_output, x_length)
        r_loss, metric_map = self._flat_reconstruction_loss(
            flat_x_target, flat_rnn_output)

        # Sum loss over sequences.
        cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)
        r_losses = []
        for i in range(batch_size):
            b, e = cum_x_len[i], cum_x_len[i + 1]
            r_losses.append(tf.reduce_sum(r_loss[b:e]))
        r_loss = tf.stack(r_losses)

        return r_loss, metric_map, decode_results

    def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
               start_inputs=None, end_fn=None):
        """Sample from decoder with an optional conditional latent vector `z`.

        Args:
          n: Scalar number of samples to return.
          max_length: (Optional) Scalar maximum sample length to return. Required if
            data representation does not include end tokens.
          z: (Optional) Latent vectors to sample from. Required if model is
            conditional. Sized `[n, z_size]`.
          c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
          temperature: (Optional) The softmax temperature to use when sampling, if
            applicable.
          start_inputs: (Optional) Initial inputs to use for batch.
            Sized `[n, output_depth]`.
          end_fn: (Optional) A callable that takes a batch of samples (sized
            `[n, output_depth]` and emits a `bool` vector
            shaped `[batch_size]` indicating whether each sample is an end token.
        Returns:
          samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
          final_state: The final states of the decoder.
        Raises:
          ValueError: If `z` is provided and its first dimension does not equal `n`.
        """
        if z is not None and int(z.shape[0]) != n:
            raise ValueError(
                '`z` must have a first dimension that equals `n` when given. '
                'Got: %d vs %d' % (z.shape[0], n))

        # Use a dummy Z in unconditional case.
        z = tf.zeros((n, 0), tf.float32) if z is None else z

        if c_input is not None:
            # Tile control sequence across samples.
            c_input = tf.tile(tf.expand_dims(c_input, 1), [1, n, 1])

        # If not given, start with zeros.
        if start_inputs is None:
            start_inputs = tf.zeros([n, self._output_depth], dtype=tf.float32)
        # In the conditional case, also concatenate the Z.
        start_inputs = tf.concat([start_inputs, z], axis=-1)
        if c_input is not None:
            start_inputs = tf.concat([start_inputs, c_input[0]], axis=-1)

        def initialize_fn(): return (tf.zeros([n], tf.bool), start_inputs)

        def sample_fn(time, outputs, state): return self._sample(
            outputs, temperature)
        end_fn = end_fn or (lambda x: False)

        def next_inputs_fn(time, outputs, state, sample_ids):
            del outputs
            finished = end_fn(sample_ids)
            next_inputs = tf.concat([sample_ids, z], axis=-1)
            if c_input is not None:
                # We need to stop if we've run out of control input.
                finished = tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                                   lambda: finished,
                                   lambda: True)
                next_inputs = tf.concat([
                    next_inputs,
                    tf.cond(tf.less(time, tf.shape(c_input)[0] - 1),
                            lambda: c_input[time + 1],
                            lambda: tf.zeros_like(c_input[0]))  # should be unused
                ], axis=-1)
            return (finished, next_inputs, state)

        sampler = contrib_seq2seq.CustomHelper(
            initialize_fn=initialize_fn, sample_fn=sample_fn,
            next_inputs_fn=next_inputs_fn, sample_ids_shape=[
                self._output_depth],
            sample_ids_dtype=tf.float32)

        decode_results = self._decode(
            z, helper=sampler, input_shape=start_inputs.shape[1:],
            max_length=max_length)

        return decode_results.samples, decode_results


class GrooveLstmDecoder(BaseLstmDecoder):
    """Groove LSTM decoder with MSE loss for continuous values.

    At each timestep, this decoder outputs a vector of length (N_INSTRUMENTS*3).
    The default number of drum instruments is 9, with drum categories defined in
    drums_encoder_decoder.py

    For each instrument, the model outputs a triple of (on/off, velocity, offset),
    with a binary representation for on/off, continuous values between 0 and 1
    for velocity, and continuous values between -0.5 and 0.5 for offset.
    """

    def _activate_outputs(self, flat_rnn_output):
        output_hits, output_velocities, output_offsets = tf.split(
            flat_rnn_output, 3, axis=1)

        output_hits = tf.nn.sigmoid(output_hits)
        output_velocities = tf.nn.sigmoid(output_velocities)
        output_offsets = tf.nn.tanh(output_offsets)

        return output_hits, output_velocities, output_offsets

    def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
        # flat_x_target is by default shape (1,27), [on/offs... vels...offsets...]
        # split into 3 equal length vectors
        target_hits, target_velocities, target_offsets = tf.split(
            flat_x_target, 3, axis=1)

        output_hits, output_velocities, output_offsets = self._activate_outputs(
            flat_rnn_output)

        hits_loss = tf.reduce_sum(tf.losses.log_loss(
            labels=target_hits, predictions=output_hits,
            reduction=tf.losses.Reduction.NONE), axis=1)

        velocities_loss = tf.reduce_sum(tf.losses.mean_squared_error(
            target_velocities, output_velocities,
            reduction=tf.losses.Reduction.NONE), axis=1)

        offsets_loss = tf.reduce_sum(tf.losses.mean_squared_error(
            target_offsets, output_offsets,
            reduction=tf.losses.Reduction.NONE), axis=1)

        loss = hits_loss + velocities_loss + offsets_loss

        metric_map = {
            'metrics/hits_loss':
                tf.metrics.mean(hits_loss),
            'metrics/velocities_loss':
                tf.metrics.mean(velocities_loss),
            'metrics/offsets_loss':
                tf.metrics.mean(offsets_loss)
        }

        return loss, metric_map

    def _sample(self, rnn_output, temperature=1.0):
        output_hits, output_velocities, output_offsets = tf.split(
            rnn_output, 3, axis=1)

        output_velocities = tf.nn.sigmoid(output_velocities)
        output_offsets = tf.nn.tanh(output_offsets)

        hits_sampler = tfp.distributions.Bernoulli(
            logits=output_hits / temperature, dtype=tf.float32)

        output_hits = hits_sampler.sample()
        return tf.concat([output_hits, output_velocities, output_offsets], axis=1)



class HierarchicalLstmDecoder(base_model.BaseDecoder):
    """Hierarchical LSTM decoder."""

    def __init__(
        self,
        core_decoder,
        level_lengths,
        disable_autoregression=False,
        hierarchical_encoder=None
    ):

        if disable_autoregression is True:
            disable_autoregression = list(range(len(level_lengths)))
        elif disable_autoregression is False:
            disable_autoregression = []
        if (hierarchical_encoder and
            (tuple(hierarchical_encoder.level_lengths[-1::-1]) !=
             tuple(level_lengths))):
            raise ValueError(
                'Incompatible hierarchical encoder level output lengths: ',
                hierarchical_encoder.level_lengths, level_lengths)

        self._core_decoder = core_decoder
        self._level_lengths = level_lengths
        self._disable_autoregression = disable_autoregression
        self._hierarchical_encoder = hierarchical_encoder

    def build(self, hparams, output_depth, is_training=True):
        self.hparams = hparams
        self._output_depth = output_depth
        self._total_length = hparams.max_seq_len
        if self._total_length != np.prod(self._level_lengths):
            raise ValueError(
                'The product of the HierarchicalLstmDecoder level lengths (%d) must '
                'equal the padded input sequence length (%d).' % (
                    np.prod(self._level_lengths), self._total_length))
        tf.logging.info('\nHierarchical Decoder:\n'
                        '  input length: %d\n'
                        '  level output lengths: %s\n',
                        self._total_length,
                        self._level_lengths)

        self._hier_cells = [
            lstm_utils.rnn_cell(
                hparams.dec_rnn_size,
                dropout_keep_prob=hparams.dropout_keep_prob,
                residual=hparams.residual_decoder)
            # Subtract 1 for the core decoder level
            for _ in range(len(self._level_lengths) - 1)]

        with tf.variable_scope('core_decoder', reuse=tf.AUTO_REUSE):
            self._core_decoder.build(hparams, output_depth, is_training)

    @property
    def state_size(self):
        return self._core_decoder.state_size

    def _merge_decode_results(self, decode_results):
        assert decode_results
        time_axis = 1
        zipped_results = lstm_utils.LstmDecodeResults(
            *list(zip(*decode_results)))
        if zipped_results.rnn_output[0] is None:
            rnn_output = None
            rnn_input = None
        else:
            rnn_output = tf.concat(zipped_results.rnn_output, axis=time_axis)
            rnn_input = tf.concat(zipped_results.rnn_input, axis=time_axis)
        return lstm_utils.LstmDecodeResults(
            rnn_output=rnn_output,
            rnn_input=rnn_input,
            samples=tf.concat(zipped_results.samples, axis=time_axis),
            final_state=zipped_results.final_state[-1],
            final_sequence_lengths=tf.stack(
                zipped_results.final_sequence_lengths, axis=time_axis))

    def _hierarchical_decode(self, z, base_decode_fn):
        """Depth first decoding from `z`, passing final embeddings to base fn."""
        batch_size = z.shape[0]
        # Subtract 1 for the core decoder level.
        num_levels = len(self._level_lengths) - 1

        hparams = self.hparams
        batch_size = hparams.batch_size

        def recursive_decode(initial_input, path=None):
            """Recursive hierarchical decode function."""
            path = path or []
            level = len(path)

            if level == num_levels:
                with tf.variable_scope('core_decoder', reuse=tf.AUTO_REUSE):
                    return base_decode_fn(initial_input, path)

            scope = tf.VariableScope(
                tf.AUTO_REUSE, 'decoder/hierarchical_level_%d' % level)
            num_steps = self._level_lengths[level]
            with tf.variable_scope(scope):
                state = lstm_utils.initial_cell_state_from_embedding(
                    self._hier_cells[level], initial_input, name='initial_state')
            if level not in self._disable_autoregression:
                # The initial input should be the same size as the tensors returned by
                # next level.
                if self._hierarchical_encoder:
                    input_size = self._hierarchical_encoder.level(
                        0).output_depth
                elif level == num_levels - 1:
                    input_size = sum(tf.nest.flatten(
                        self._core_decoder.state_size))
                else:
                    input_size = sum(
                        tf.nest.flatten(self._hier_cells[level + 1].state_size))
                next_input = tf.zeros([batch_size, input_size])
            lower_level_embeddings = []
            for i in range(num_steps):
                if level in self._disable_autoregression:
                    next_input = tf.zeros([batch_size, 1])
                else:
                    next_input = tf.concat([next_input, initial_input], axis=1)
                with tf.variable_scope(scope):
                    output, state = self._hier_cells[level](
                        next_input, state, scope)
                next_input = recursive_decode(output, path + [i])
                lower_level_embeddings.append(next_input)
            if self._hierarchical_encoder:
                # Return the encoding of the outputs using the appropriate level of the
                # hierarchical encoder.
                enc_level = num_levels - level
                return self._hierarchical_encoder.level(enc_level).encode(
                    sequence=tf.stack(lower_level_embeddings, axis=1),
                    sequence_length=tf.fill([batch_size], num_steps))
            else:
                # Return the final state.
                return tf.concat(tf.nest.flatten(state), axis=-1)

        return recursive_decode(z)

    def _reshape_to_hierarchy(self, t):
        """Reshapes `t` so that its initial dimensions match the hierarchy."""
        # Exclude the final, core decoder length.
        level_lengths = self._level_lengths[:-1]
        t_shape = t.shape.as_list()
        t_rank = len(t_shape)
        batch_size = t_shape[0]
        hier_shape = [batch_size] + level_lengths
        if t_rank == 3:
            hier_shape += [-1] + t_shape[2:]
        elif t_rank != 2:
            # We only expect rank-2 for lengths and rank-3 for sequences.
            raise ValueError('Unexpected shape for tensor: %s' % t)
        hier_t = tf.reshape(t, hier_shape)
        # Move the batch dimension to after the hierarchical dimensions.
        num_levels = len(level_lengths)
        perm = list(range(len(hier_shape)))
        perm.insert(num_levels, perm.pop(0))
        return tf.transpose(hier_t, perm)

    def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                            c_input=None):
        """Reconstruction loss calculation.
        Args:
          x_input: Batch of decoder input sequences of concatenated segmeents for
            teacher forcing, sized `[batch_size, max_seq_len, output_depth]`.
          x_target: Batch of expected output sequences to compute loss against,
            sized `[batch_size, max_seq_len, output_depth]`.
          x_length: Length of input/output sequences, sized
            `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
            each length must either equal `max_seq_len` or 0. In this case, the
            segment lengths are assumed to be constant and the total length will be
            evenly divided amongst the segments.
          z: (Optional) Latent vectors. Required if model is conditional. Sized
            `[n, z_size]`.
          c_input: (Optional) Batch of control sequences, sized
            `[batch_size, max_seq_len, control_depth]`. Required if conditioning on
            control sequences.
        Returns:
          r_loss: The reconstruction loss for each sequence in the batch.
          metric_map: Map from metric name to tf.metrics return values for logging.
          decode_results: The LstmDecodeResults.
        Raises:
          ValueError: If `c_input` is provided in re-encoder mode.
        """
        if self._hierarchical_encoder and c_input is not None:
            raise ValueError(
                'Re-encoder mode unsupported when conditioning on controls.')

        batch_size = int(x_input.shape[0])

        x_length = lstm_utils.maybe_split_sequence_lengths(
            x_length, np.prod(self._level_lengths[:-1]), self._total_length)

        hier_input = self._reshape_to_hierarchy(x_input)
        hier_target = self._reshape_to_hierarchy(x_target)
        hier_length = self._reshape_to_hierarchy(x_length)
        hier_control = (
            self._reshape_to_hierarchy(c_input) if c_input is not None else None)

        loss_outputs = []

        def base_train_fn(embedding, hier_index):
            """Base function for training hierarchical decoder."""
            split_size = self._level_lengths[-1]
            split_input = hier_input[hier_index]
            split_target = hier_target[hier_index]
            split_length = hier_length[hier_index]
            split_control = (
                hier_control[hier_index] if hier_control is not None else None)

            res = self._core_decoder.reconstruction_loss(
                split_input, split_target, split_length, embedding, split_control)
            loss_outputs.append(res)
            decode_results = res[-1]

            if self._hierarchical_encoder:
                # Get the approximate "sample" from the model.
                # Start with the inputs the RNN saw (excluding the start token).
                samples = decode_results.rnn_input[:, 1:]
                # Pad to be the max length.
                samples = tf.pad(
                    samples,
                    [(0, 0), (0, split_size - tf.shape(samples)[1]), (0, 0)])
                samples.set_shape([batch_size, split_size, self._output_depth])
                # Set the final value based on the target, since the scheduled sampling
                # helper does not sample the final value.
                samples = lstm_utils.set_final(
                    samples,
                    split_length,
                    lstm_utils.get_final(
                        split_target, split_length, time_major=False),
                    time_major=False)
                # Return the re-encoded sample.
                return self._hierarchical_encoder.level(0).encode(
                    sequence=samples,
                    sequence_length=split_length)
            elif self._disable_autoregression:
                return None
            else:
                return tf.concat(tf.nest.flatten(decode_results.final_state), axis=-1)

        z = tf.zeros([batch_size, 0]) if z is None else z
        self._hierarchical_decode(z, base_train_fn)

        # Accumulate the split sequence losses.
        r_losses, metric_maps, decode_results = list(zip(*loss_outputs))

        # Merge the metric maps by passing through renamed values and taking the
        # mean across the splits.
        merged_metric_map = {}
        for metric_name in metric_maps[0]:
            metric_values = []
            for i, m in enumerate(metric_maps):
                merged_metric_map['segment/%03d/%s' %
                                  (i, metric_name)] = m[metric_name]
                metric_values.append(m[metric_name][0])
            merged_metric_map[metric_name] = (
                tf.reduce_mean(metric_values), tf.no_op())

        return (tf.reduce_sum(r_losses, axis=0),
                merged_metric_map,
                self._merge_decode_results(decode_results))

    def sample(self, n, max_length=None, z=None, c_input=None,
               **core_sampler_kwargs):
        """Sample from decoder with an optional conditional latent vector `z`.
        Args:
          n: Scalar number of samples to return.
          max_length: (Optional) maximum total length of samples. If given, must
            match `hparams.max_seq_len`.
          z: (Optional) Latent vectors to sample from. Required if model is
            conditional. Sized `[n, z_size]`.
          c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
          **core_sampler_kwargs: (Optional) Additional keyword arguments to pass to
            core sampler.
        Returns:
          samples: Sampled sequences with concenated, possibly padded segments.
             Sized `[n, max_length, output_depth]`.
          decoder_results: The merged LstmDecodeResults from sampling.
        Raises:
          ValueError: If `z` is provided and its first dimension does not equal `n`,
            or if `c_input` is provided in re-encoder mode.
        """
        if z is not None and int(z.shape[0]) != n:
            raise ValueError(
                '`z` must have a first dimension that equals `n` when given. '
                'Got: %d vs %d' % (z.shape[0], n))
        z = tf.zeros([n, 0]) if z is None else z

        if self._hierarchical_encoder and c_input is not None:
            raise ValueError(
                'Re-encoder mode unsupported when conditioning on controls.')

        if max_length is not None:
            with tf.control_dependencies([
                tf.assert_equal(
                    max_length, self._total_length,
                    message='`max_length` must equal `hparams.max_seq_len` if given.')
            ]):
                max_length = tf.identity(max_length)

        if c_input is not None:
            # Reshape control sequence to hierarchy.
            c_input = tf.squeeze(
                self._reshape_to_hierarchy(tf.expand_dims(c_input, 0)),
                axis=len(self._level_lengths) - 1)

        core_max_length = self._level_lengths[-1]
        all_samples = []
        all_decode_results = []

        def base_sample_fn(embedding, hier_index):
            """Base function for sampling hierarchical decoder."""
            samples, decode_results = self._core_decoder.sample(
                n,
                max_length=core_max_length,
                z=embedding,
                c_input=c_input[hier_index] if c_input is not None else None,
                start_inputs=all_samples[-1][:, -1] if all_samples else None,
                **core_sampler_kwargs)
            all_samples.append(samples)
            all_decode_results.append(decode_results)
            if self._hierarchical_encoder:
                return self._hierarchical_encoder.level(0).encode(
                    samples,
                    decode_results.final_sequence_lengths)
            else:
                return tf.concat(tf.nest.flatten(decode_results.final_state), axis=-1)

        # Populate `all_sample_ids`.
        self._hierarchical_decode(z, base_sample_fn)

        all_samples = tf.concat(
            [tf.pad(s, [(0, 0), (0, core_max_length - tf.shape(s)[1]), (0, 0)])
             for s in all_samples],
            axis=1)
        return all_samples, self._merge_decode_results(all_decode_results)




    
def get_default_hparams():
    """Returns copy of default HParams for LSTM models."""
    hparams_map = base_model.get_default_hparams().values()
    hparams_map.update({
        'conditional': True,
        'dec_rnn_size': [512],  # Decoder RNN: number of units per layer.
        # Encoder RNN: number of units per layer per dir.
        'enc_rnn_size': [256],
        'dropout_keep_prob': 1.0,  # Probability all dropout keep.
        'sampling_schedule': 'constant',  # constant, exponential, inverse_sigmoid
        # Interpretation is based on `sampling_schedule`.
        'sampling_rate': 0.0,
        'use_cudnn': False,  # DEPRECATED
        'residual_encoder': False,  # Use residual connections in encoder.
        'residual_decoder': False,  # Use residual connections in decoder.
        # Decoder control preprocessing.
        'control_preprocessing_rnn_size': [256],
    })
    return contrib_training.HParams(**hparams_map)

