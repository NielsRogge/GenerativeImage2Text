# import json
# import os.path as op
# from .common import qd_tqdm as tqdm
# from .common import json_dump
# from .common import pilimg_from_base64
# from .common import get_mpi_rank, get_mpi_size, get_mpi_local_rank

# from .tsv_io import TSVFile, tsv_writer, tsv_reader
# from .common import write_to_file
import argparse

import torch
import PIL
from pprint import pformat
import logging
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from azfuse import File

# from .common import init_logging
# from .common import parse_general_args
# from .tsv_io import load_from_yaml_file
from .torch_common import torch_load
from .torch_common import load_state_dict
from .process_image import load_image_by_pil
from .model import get_git_model



class MinMaxResizeForTest(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size

        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __repr__(self):
        return 'MinMaxResizeForTest({}, {})'.format(
            self.min_size, self.max_size)

    def __call__(self, img):
        size = self.get_size(img.size)
        import torchvision.transforms.functional as F
        image = F.resize(img, size, interpolation=PIL.Image.BICUBIC)
        return image


def test_git_inference_single_image(image_path, model_name, prefix):
    param = {}
    # if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
    #     param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if isinstance(image_path, str):
        image_path = [image_path]
    # if it is more than 1 image, it is normally a video with multiple image
    # frames
    img = [load_image_by_pil(i) for i in image_path]

    transforms = get_image_transform(param)
    img = [transforms(i) for i in img]

    # model
    model = get_git_model(tokenizer, param)
    # pretrained = f'output/{model_name}/snapshot/model.pt'
    # checkpoint = torch_load(pretrained)['model']
    checkpoint = torch.hub.load_state_dict_from_url(f"https://publicgit.blob.core.windows.net/data/output/{model_name}/snapshot/model.pt",
                                                map_location="cpu", file_name=model_name)["model"]
    load_state_dict(model, checkpoint)
    model.cuda()
    model.eval()
    img = [i.unsqueeze(0).cuda() for i in img]

    print("Loaded model!")

    print("Preparing cats image...")
    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    img = [transforms(image).unsqueeze(0).cuda()]

    # prefix
    max_text_len = 40
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=max_text_len)
    payload = prefix_encoding['input_ids']
    if len(payload) > max_text_len - 2:
        payload = payload[-(max_text_len - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    print("Decoding of prompt:", tokenizer.decode(input_ids))

    with torch.no_grad():
        result = model({
            'image': img,
            'prefix': torch.tensor(input_ids).unsqueeze(0).cuda(),
        })

    cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
    print("Caption:", cap)
    logging.info('output: {}'.format(cap))

def get_image_transform(param):
    crop_size = param.get('test_crop_size', 224)
    if 'test_respect_ratio_max' in param:
        trans = [
            MinMaxResizeForTest(crop_size, param['test_respect_ratio_max'])
        ]
    else:
        trans = [
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
            lambda image: image.convert("RGB"),

        ]
    trans.extend([
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    transforms = Compose(trans)
    return transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="GIT_BASE",
        type=str,
        help="Name of the model you'd like to try.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="Prefix for generating text.",
    )

    args = parser.parse_args()
    test_git_inference_single_image(image_path="aux_data/images/1.jpg", model_name=args.model_name, prefix=args.prefix)
    # init_logging()
    # kwargs = parse_general_args()
    # logging.info('param:\n{}'.format(pformat(kwargs)))
    # function_name = kwargs['type']
    # del kwargs['type']
    # locals()[function_name](**kwargs)