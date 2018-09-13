"""Generates an ONNX graph for sentiment analysis."""

import chainer
import numpy as np
import onnx

import oniku_script

F = chainer.functions


def _gen_random_sequence(batch_size, sequence_length, num_vocabs):
    lengths = np.random.randint(2, sequence_length, size=batch_size)
    lengths = np.flip(np.sort(lengths), axis=0)
    labels = np.random.randint(
        2, num_vocabs, size=(batch_size, sequence_length))
    return labels, lengths


def gen_rnn_sentiment_test(cell_type,
                           num_vocabs=10,
                           num_hidden=5,
                           batch_size=3,
                           sequence_length=4,
                           output_loss_only=False,
                           param_initializer=np.random.random):
    def fn(test_name):
        gb = oniku_script.GraphBuilder(test_name)
        if cell_type == 'LSTM':
            wr = 8
            perm = [0, 2, 1, 3, 4, 6, 5, 7]
            num_direction = 1
            direction = 'forward'
        elif cell_type == 'BiLSTM':
            wr = 16
            perm = np.tile([0, 2, 1, 3], 4) + np.repeat(np.arange(4), 4) * 4
            num_direction = 2
            direction = 'bidirectional'
        elif cell_type == 'GRU':
            wr = 6
            perm = [1, 0, 2, 4, 3, 5]
            num_direction = 1
            direction = 'forward'
        elif cell_type == 'BiGRU':
            wr = 12
            perm = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11]
            num_direction = 2
            direction = 'bidirectional'
        else:
            raise RuntimeError('Unknown cell_type: %s' % cell_type)
        embed_size = num_hidden
        np.random.seed(42)
        if batch_size == 3 and sequence_length == 4:
            labels = np.array([[1, 2, 3, 7], [4, 5, 0, 0], [6, 0, 0, 0]])
            lengths = np.array([4, 2, 1])
            targets = np.array([1, 0, 1])
        else:
            labels, lengths = _gen_random_sequence(
                batch_size, sequence_length, num_vocabs)
            targets = np.random.randint(2, size=batch_size)

        embed = param_initializer(
            size=(num_vocabs, embed_size)).astype(np.float32)
        weight = param_initializer(
            size=(embed_size, num_hidden * wr)).astype(np.float32)
        bias = param_initializer(
            size=(num_hidden * wr,)).astype(np.float32)
        linear = param_initializer(
            size=(num_direction * num_hidden, 2)).astype(np.float32)

        x = F.embed_id(labels, embed)
        state = np.zeros(
            (num_direction, len(labels), num_hidden)).astype(np.float32)
        xs = F.transpose_sequence([v[:l] for v, l in zip(x, lengths)])
        ch_weight = np.split(weight, wr, axis=1)
        ch_weight = [ch_weight[i] for i in perm]
        ch_bias = np.split(bias, wr, axis=0)
        ch_bias = [ch_bias[i] for i in perm]
        if cell_type == 'LSTM':
            h, _, rnn_outputs = F.n_step_lstm(1, 0.0,
                                              state,
                                              state,
                                              [ch_weight],
                                              [ch_bias],
                                              xs)
        elif cell_type == 'BiLSTM':
            h, _, rnn_outputs = F.n_step_bilstm(1, 0.0,
                                                state,
                                                state,
                                                [ch_weight[:8], ch_weight[8:]],
                                                [ch_bias[:8], ch_bias[8:]],
                                                xs)
        elif cell_type == 'GRU':
            h, rnn_outputs = F.n_step_gru(1, 0.0,
                                          state,
                                          [ch_weight],
                                          [ch_bias],
                                          xs)
        elif cell_type == 'BiGRU':
            h, rnn_outputs = F.n_step_bigru(1, 0.0,
                                            state,
                                            [ch_weight[:6], ch_weight[6:]],
                                            [ch_bias[:6], ch_bias[6:]],
                                            xs)
        shape = (len(labels), num_hidden * num_direction)
        h = F.reshape(h, shape)
        rnn_outputs = F.pad_sequence(rnn_outputs)
        rnn_outputs = F.reshape(rnn_outputs,
                                (-1, len(labels), num_direction, num_hidden))
        rnn_outputs = F.transpose(rnn_outputs, axes=[0, 2, 1, 3])
        result = F.linear(h, np.transpose(linear))
        loss = F.softmax_cross_entropy(result, targets)

        weight_w, weight_r = np.split(weight, 2, axis=1)
        labels_v = gb.input('labels', labels)
        lengths_v = gb.input('lengths', lengths)
        targets_v = gb.input('targets', targets)
        embed_v = gb.param('embed', embed)
        weight_w_v = gb.param(
            'weight_w',
            np.reshape(np.transpose(weight_w),
                       (num_direction, -1, embed_size)))
        weight_r_v = gb.param(
            'weight_r',
            np.reshape(np.transpose(weight_r),
                       (num_direction, -1, num_hidden)))
        bias_v = gb.param('bias', np.reshape(bias, (num_direction, -1)))
        linear_v = gb.param('linear', linear)

        x = gb.Gather([embed_v, labels_v])
        x = gb.Transpose([x], perm=[1, 0, 2])
        if cell_type in ['LSTM', 'BiLSTM']:
            rnn_outputs_v, h = gb.LSTM(
                [x, weight_w_v, weight_r_v, bias_v, lengths_v],
                outputs=['rnn_outputs', 'last_state'],
                direction=direction)
        elif cell_type in ['GRU', 'BiGRU']:
            rnn_outputs_v, h = gb.GRU(
                [x, weight_w_v, weight_r_v, bias_v, lengths_v],
                outputs=['rnn_outputs', 'last_state'],
                direction=direction)
        shape_v = gb.const(onnx.TensorProto.INT64, shape)
        h = gb.Reshape([h, shape_v])
        result_v = gb.MatMul([h, linear_v])
        loss_v = gb.OnikuxSoftmaxCrossEntropy([result_v, targets_v])

        if not output_loss_only:
            gb.output(rnn_outputs_v, rnn_outputs.array)
            gb.output(result_v, result.array)
        gb.output(loss_v, loss.array)

        gb.gen_test()

    return fn


if __name__ == '__main__':
    # https://github.com/ilkarman/DeepLearningFrameworks/blob/master/notebooks/common/params_lstm.py
    fn = gen_rnn_sentiment_test('LSTM',
                                num_vocabs=30000,
                                num_hidden=100,
                                batch_size=64,
                                sequence_length=150,
                                output_loss_only=False,
                                param_initializer=np.random.normal)
    fn('sentiment_lstm')
