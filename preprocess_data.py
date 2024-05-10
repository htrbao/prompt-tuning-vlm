from datasets import CaptioningDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("model/beit3.spm")

CaptioningDataset.make_coco_captioning_dataset_index_from_origin(
    data_index_file="data/annotations_trainval2014/annotations",
    data_path="data",
    tokenizer=tokenizer,
    year="2014"
)