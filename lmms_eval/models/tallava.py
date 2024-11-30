import os
import uuid
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context
# from llava.model import load_pretrained_model
from llava.model import load_pretrained_model

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer


@register_model("ta-llava")
class TALlava(lmms):
    """
    TA-LLaVA Model
    https://github.com/ToviTu/TA-LLaVA/tree/main
    """

    def __init__(self, pretrained: str, device: str = "cuda", **kwargs) -> None:
        super().__init__()
        # assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self.batch_size_per_gpu = kwargs.pop("batch_size", 1)
        
        # Load the model, tokenizer, etc.
        self._tokenizer, self._model, self._image_processor, self._context_len, self._vision_priori = load_pretrained_model(
            model_path=pretrained,
            model_base=None,  # Set if you use a base model
            model_name="tallava_gemma",
            device=device,
        )
        self._model.eval()
        self._device = device
        self._model.to(self._device)
        self._model.vision_priori = self._vision_priori
        self.batch_size_per_gpu = 1
        
        print(hasattr(self._model, "vision_priori"))

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eod_id

    @property
    def max_length(self):
        return self._max_length

    # should be deleted since max_new_tokens is decided by gen_kwargs not a model property
    # @property
    # def max_new_tokens(self) -> int:
    #     return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), desc="Calculating log probabilities", unit="request")

        for req in requests:
            context, target, doc_to_visual, doc_id, task, split = req.args

            context_tokens = self.tokenizer(
                context, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            target_tokens = self.tokenizer(
                target, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )

            input_ids = context_tokens.input_ids.to(self.device)
            attention_mask = context_tokens.attention_mask.to(self.device)
            target_ids = target_tokens.input_ids.to(self.device)

            if doc_to_visual:
                doc = self.task_dict[task][split][doc_id]
                raw_image = doc_to_visual(doc)

                image_tensor = self._image_processor(raw_image, return_tensors="pt")["pixel_values"].squeeze(0)
                image_tensor = torch.clamp(image_tensor, 0.0, 1.0).to(dtype=torch.bfloat16, device=self.device)

                images = torch.stack([image_tensor])

                output = self.model.vis_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    labels=target_ids
                )
            else:
                output = self.model.text_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_ids
                )

            log_probs = -output.loss.item()
            greedy_match = (output.logits.argmax(dim=-1) == target_ids).all().item()

            res.append((log_probs, greedy_match))
            pbar.update(1)

        pbar.close()
        return res


    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
    def pope_doc_to_visual(doc):
        print(f"Received doc: {doc}")
        return [doc["image"].convert("RGB")]

    # uncomment this, and run example-tallava.py
    '''
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        for req in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.arguments
            
            tokens = self.tokenizer(
                context, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            if doc_to_visual:
                print("Using visual input")
                images = [doc_to_visual(doc_id)]

                if isinstance(images[0], torch.Tensor):
                    print("Using torch.Tensor")
                    images = [torch.clamp(image, 0.0, 1.0).to(dtype=torch.bfloat16) for image in images]
                else:
                    print("load Image")
                    images = [
                        image["pixel_values"] if isinstance(image, dict) else image
                        for image in images
                    ]
                
                images = torch.stack(images).to(self.device, dtype=torch.bfloat16)

                output_ids = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    images=images, 
                    **gen_kwargs
                )
            else:
                output_ids = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    **gen_kwargs
                )
            
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            # print(output_text) 
            res.append(output_text)
        return res
    
    '''


    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), desc="Processing requests", unit="request")

        for req in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.arguments
            
            tokens = self.tokenizer(
                context, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            if doc_to_visual:
                # print("Using visual input")
                doc = self.task_dict[task][split][doc_id]
                raw_image = doc_to_visual(doc) 
            
                # torch.Tensor
                image_tensor = self._image_processor(raw_image, return_tensors="pt")["pixel_values"].squeeze(0)
                image_tensor = torch.clamp(image_tensor, 0.0, 1.0).to(dtype=torch.bfloat16, device=self.device)
                
                # stack the image_tensor
                images = torch.stack([image_tensor])
                
                # images is list of torch.Tensor
                # images = torch.stack(images).to(self.device, dtype=torch.bfloat16)
                
                if "until" in gen_kwargs:
                    del gen_kwargs["until"]

                output_ids = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    images=images, 
                    **gen_kwargs
                )
            else:
                if "until" in gen_kwargs:
                    del gen_kwargs["until"]
                output_ids = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    **gen_kwargs
                )
            
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            # print(output_text) 
            pbar.update(1)
            res.append(output_text)
            
        pbar.close()
        return res





    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
