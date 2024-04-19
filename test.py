import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from torchvision import transforms
from transformers import XLMRobertaTokenizer

import utils
import modeling_finetune

# Get current workdir of this file
CWD = Path(__file__).parent
print(CWD)


# String
s = "<mask>"


class Preprocess:
    def __init__(self, tokenizer):
        self.max_len = 64
        self.input_size = 480

        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def preprocess(self, input: Union[str, Image.Image]):
        if isinstance(input, str):
            tokens = self.tokenizer.tokenize(input)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
            num_tokens = len(tokens)
            padding_mask = [0] * num_tokens + [1] * (self.max_len - num_tokens)

            return (
                torch.LongTensor(
                    tokens + [self.pad_token_id] * (self.max_len - num_tokens)
                ).unsqueeze(0),
                torch.Tensor(padding_mask).unsqueeze(0),
                num_tokens,
            )
        elif isinstance(input, Image.Image):
            return self.transform(input).unsqueeze(0)
        else:
            raise Exception("Invalid input type")


class Beit3Model:
    def __init__(
        self,
        model_name: str = "beit3_base_patch16_480_captioning_with_gott",
        model_path: str = os.path.join(
            CWD,
            "model/beit3_base_patch16_480_coco_captioning.pth",
        ),
        device: str = "cuda",
    ):
        self._load_model(model_name, model_path, device)
        self.device = device

    # @lru_cache(maxsize=1)
    def _load_model(self, model_name, model_path, device: str = "cpu"):
        kwargs = {
            "ori_ctx_init": "this is a",
            "ctx_init": torch.IntTensor([[0, 127,  18,  10,   2]]),
        }
        self.model = create_model(
            model_name,
            pretrained=False,
            drop_path_rate=0.1,
            vocab_size=64010,
            checkpoint_activations=False,
            **kwargs
        )

        if model_name:
            utils.load_model_and_may_interpolate(model_path, self.model, "model|module", "")

        self.preprocessor = Preprocess(
            XLMRobertaTokenizer(os.path.join(CWD, "model/beit3.spm"))
        )

        # print(self.preprocessor.preprocess("This is a"))
        self.model.to(device)

    def get_answer(self, input_img: Image.Image, intput_ques: str):
        token_ids, padding_mask, _ = self.preprocessor.preprocess(intput_ques)

        image_input = self.preprocessor.preprocess(input_img)
        image_input = image_input.to(self.device)
        
        ans, _ = self.model(image_input, token_ids, padding_mask, None)
        ans = ans[:, len(s.split()) + 1, :]
        ids = torch.argmax(F.log_softmax(ans, dim=-1), dim=1)
        return self.preprocessor.tokenizer.decode(ids)
        
model = Beit3Model(device="cpu")
print(model.get_answer(Image.open("data/val2014/COCO_val2014_000000581655.jpg"),
                       s))




