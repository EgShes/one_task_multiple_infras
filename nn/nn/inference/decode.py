from typing import List

import numpy as np


def greedy_decode(predicted_seq, chars_list: List[str]):
    full_pred_labels = []
    labels = []
    for i in range(predicted_seq.shape[0]):
        single_prediction = predicted_seq[i, :, :]
        predicted_labels = []
        for j in range(single_prediction.shape[1]):
            predicted_labels.append(np.argmax(single_prediction[:, j], axis=0))

        without_repeating = []
        current_char = predicted_labels[0]
        if current_char != len(chars_list) - 1:
            without_repeating.append(current_char)
        for c in predicted_labels:
            if (current_char == c) or (c == len(chars_list) - 1):
                if c == len(chars_list) - 1:
                    current_char = c
                continue
            without_repeating.append(c)
            current_char = c

        full_pred_labels.append(without_repeating)

    for i, label in enumerate(full_pred_labels):
        decoded_label = ""
        for j in label:
            decoded_label += chars_list[j]
        labels.append(decoded_label)

    return labels, full_pred_labels


def beam_decode(predicted_seq, chars_list: List[str]):
    labels = []
    final_labels = []
    final_prob = []
    k = 1
    for i in range(predicted_seq.shape[0]):
        sequences = [[list(), 0.0]]
        all_seq = []
        single_prediction = predicted_seq[i, :, :]
        for j in range(single_prediction.shape[1]):
            single_seq = []
            for char in single_prediction[:, j]:
                single_seq.append(char)
            all_seq.append(single_seq)

        for row in all_seq:
            all_candidates = []
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - row[j]]

                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            # select k best
            sequences = ordered[:k]

        full_pred_labels = []
        probs = []
        for i in sequences:

            predicted_labels = i[0]
            without_repeating = []
            current_char = predicted_labels[0]
            if current_char != len(chars_list) - 1:
                without_repeating.append(current_char)
            for c in predicted_labels:
                if (current_char == c) or (c == len(chars_list) - 1):
                    if c == len(chars_list) - 1:
                        current_char = c
                    continue
                without_repeating.append(c)
                current_char = c

            full_pred_labels.append(without_repeating)
            probs.append(i[1])
        for i, label in enumerate(full_pred_labels):
            decoded_label = ""
            for j in label:
                decoded_label += chars_list[j]
            labels.append(decoded_label)
            final_prob.append(probs[i])
            final_labels.append(full_pred_labels[i])

    return labels, final_prob, final_labels
