## Introduction
One implementation of the paper "Exploiting Discourse-Level Segmentation for Extractive Summarization".

* Users can apply it to process the input text, and obtain the EDU segments, as well as the merged segments for extractive summarization. <br>
* This repo and the pre-trained model are only for research use. Please cite the papers if they are helpful. <br>

## Package Requirements
The model training and inference scripts were tested on following libraries and versions:
1. pytorch==1.7.1
2. transformers==4.8.2

## Inference: How to use it for parsing
* Download the code from the repo [DMRST_Parser](https://github.com/seq-to-mind/DMRST_Parser).
* Put the `SegMerge_Infer.py` to the repo folder.
* Put your text paragraph to the file `./data/text_for_inference.txt`. <br>
* Pre-trained model checkpoint should be downloaded and saved at `./depth_mode/Savings/` (see repo `DMRST_Parser`). <br>
* Run the script `SegMerge_Infer.py` to obtain the EDU segmentationg and merge result. See the script for detailed model output. <br>
* We recommend users to run the parser on a GPU-equipped environment. <br>

## Citation
If the work is helpful, please cite our papers in your publications, reports, slides, and docs.

```
@inproceedings{liu2019exploiting,
  title={Exploiting discourse-level segmentation for extractive summarization},
  author={Liu, Zhengyuan and Chen, Nancy},
  booktitle={Proceedings of the 2nd Workshop on New Frontiers in Summarization},
  pages={116--121},
  year={2019}
}
```

```
@inproceedings{liu-etal-2021-dmrst,
    title = "{DMRST}: A Joint Framework for Document-Level Multilingual {RST} Discourse Segmentation and Parsing",
    author = "Liu, Zhengyuan and Shi, Ke and Chen, Nancy",
    booktitle = "Proceedings of the 2nd Workshop on Computational Approaches to Discourse",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.codi-main.15",
    pages = "154--164",
}
```

