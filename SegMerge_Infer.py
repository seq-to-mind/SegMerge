import os
import torch
import numpy as np
import argparse
import os
import config
import re
from transformers import AutoTokenizer, AutoModel
from model_depth import ParsingNet
from nltk import pos_tag, word_tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)
min_span_len = 5


def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument('--ModelPath', type=str, default='depth_mode/Savings/multi_all_checkpoint.torchsave', help='pre-trained model')
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, _, predict_EDU_breaks = model.encoder(input_sen_batch, EDU_breaks=None, is_test=True)

            all_segmentation_pred.extend(predict_EDU_breaks)
    return input_sentences, all_segmentation_pred


def recover_EDU_span(one_tokenized_text, one_segmentation_pred, subword_tokenizer):
    one_EDU_list = []
    assert one_segmentation_pred[-1] == len(one_tokenized_text) - 1
    tmp_split_list = [-1, ] + one_segmentation_pred
    for j in range(1, len(tmp_split_list)):
        tmp_str = " " + subword_tokenizer.convert_tokens_to_string(one_tokenized_text[tmp_split_list[j - 1] + 1:tmp_split_list[j] + 1]).strip() + " "
        one_EDU_list.append(tmp_str)
    return one_EDU_list


def decide_segment(cur_segment, in_idx, in_lines):
    tmp_merged = cur_segment + in_lines[in_idx]

    tmp_merged_len = len(tmp_merged.strip().split())
    tmp_cur_segment_len = len(cur_segment.strip().split())
    tmp_next_seg_len = len(in_lines[in_idx].strip().split())

    if in_idx == len(in_lines) - 1:
        return True

    if in_lines[in_idx].startswith("`s ") or in_lines[in_idx].startswith("\'s ") or len(in_lines[in_idx]) < 3:
        return False

    if tmp_next_seg_len < min_span_len and in_lines[in_idx].endswith(". ") and (not cur_segment.endswith(". ")):
        return False

    if tmp_merged.endswith(": ") and len(tmp_merged) > 4 and tmp_merged.strip()[-3].isdigit() and in_idx + 1 < len(in_lines) and in_lines[in_idx + 1].strip()[0].isdigit():
        return False

    if in_lines[in_idx].endswith("said. ") or in_lines[in_idx].endswith("says. "):
        return True

    if len(re.findall("\`\s\`", tmp_merged)) > 0 and len(re.findall("\'\s\'", tmp_merged)) < 1:
        if (" says " not in cur_segment) and (" said " not in cur_segment):
            return False

    if tmp_merged.endswith(". "):
        return True

    if tmp_merged_len < min_span_len + 1:
        return False

    """ you can comment this POS tagging step to higher processing speed, if needed """
    pos_list = pos_tag(word_tokenize(tmp_merged))
    pos_list = " ".join([i[1] for i in pos_list])
    if "VB" not in pos_list:
        return False

    return True


def EDU_level_merge_segments(input_lines):
    input_lines = [i.replace("\n", " ").replace("` '", "' '") for i in input_lines]

    new_lines = []
    seg_idx = 0
    current_merged_span = ""

    while seg_idx < len(input_lines):
        current_merged_span = re.sub("\s+", " ", current_merged_span)
        current_one_EDU = input_lines[seg_idx]

        if len(current_one_EDU.strip().split()) < min_span_len and current_one_EDU.endswith(". ") and len(new_lines) > 1 and (not (new_lines[-1] + current_merged_span).endswith(". ")):
            new_lines[-1] = new_lines[-1] + current_merged_span + current_one_EDU
            current_merged_span = ""

        else:
            if decide_segment(current_merged_span, seg_idx, input_lines) is False:
                current_merged_span = current_merged_span + current_one_EDU

            else:
                new_lines.append(current_merged_span + current_one_EDU)
                current_merged_span = ""

        seg_idx += 1

    return new_lines


if __name__ == '__main__':

    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size
    save_path = args.savepath

    """ BERT tokenizer and model """
    bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("xlm-roberta-base")

    bert_model = bert_model.cuda()

    for name, param in bert_model.named_parameters():
        param.requires_grad = False

    model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    Test_InputSentences = open("./data/text_for_inference.txt").readlines()

    input_sentences, all_segmentation_pred = inference(model, bert_tokenizer, Test_InputSentences, batch_size)

    """ process examples, you can customize this part for your own case """
    for tmp_i in range(len(input_sentences)):
        print(input_sentences[tmp_i])
        print(len(input_sentences[tmp_i]), all_segmentation_pred[tmp_i])

        """ get the EDU segmentation, recovered from subword tokenization """
        tmp_EDU_list = recover_EDU_span(one_tokenized_text=input_sentences[tmp_i], one_segmentation_pred=all_segmentation_pred[tmp_i], subword_tokenizer=bert_tokenizer)
        print(tmp_EDU_list)

        """ get the merged EDU segments """
        merged_EDU_list = EDU_level_merge_segments(tmp_EDU_list)
        print(merged_EDU_list)

        assert "".join(tmp_EDU_list).replace(" ", "") == "".join(merged_EDU_list).replace(" ", "")
