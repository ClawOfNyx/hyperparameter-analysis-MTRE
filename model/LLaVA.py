import sys
import os
import json

import torch
from torch import nn
import numpy as np
from io import BytesIO
from transformers import TextStreamer
from transformers.generation import BeamSearchDecoderOnlyOutput

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from model.base import LargeMultimodalModel

class LLaVA(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaVA, self).__init__()
        load_8bit = False
        load_4bit = False
        
        # Load Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        if "finetune-lora" in args.model_path:
            model_base = "liuhaotian/llava-v1.5-7b"
        elif "lora" in args.model_path:
            model_base = "lmsys/vicuna-7b-v1.5"
        else:
            model_base = None
        print(model_base)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, model_base, model_name, load_8bit, load_4bit)
        print(model_name)
        self.conv_mode = "llava_v1"
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
    
    def refresh_chat(self):
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles
    
    def _basic_forward(self, image, prompt, return_dict=False):
        self.refresh_chat()
        
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to(self.device)

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        conv_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = ["###"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                
                max_new_tokens=100,
                streamer=streamer,
                use_cache=True,
                
                stopping_criteria=[stopping_criteria],
                
                return_dict_in_generate=return_dict,
                output_attentions=return_dict,
                output_hidden_states=return_dict,
                output_scores=return_dict)
            
        return input_ids, outputs
    
    def forward_with_probs(self, image, prompt):
        input_ids, outputs = self._basic_forward(image, prompt, return_dict=True)
        
        if isinstance(outputs, BeamSearchDecoderOnlyOutput):
            beam_indices = outputs.beam_indices[0].cpu()
            beam_indices = [i for i in beam_indices if i != -1]
            logits = None
            probs = float(outputs.sequences_scores.cpu().item())
            output_ids = outputs.sequences[0][-len(beam_indices):]
        else:
            is_tuple = isinstance(outputs, tuple)
            out_scores = outputs[1] if is_tuple else outputs.scores
            out_seq = outputs[0] if is_tuple else outputs.sequences

            output_ids = out_seq[0][-len(out_scores):]

            logits = torch.stack(out_scores, dim=0).squeeze(1).cpu().numpy()
            
            probs_list = [nn.functional.softmax(s, dim=-1) for s in out_scores]
            probs = torch.stack(probs_list, dim=0).squeeze(1).cpu().numpy()
            
        raw_response = self.tokenizer.decode(output_ids)
        response = raw_response.replace("</s>", "").replace("###", "").strip()
        output_ids = output_ids.cpu().numpy()
        
        return response, output_ids, logits, probs
        
    def get_p_true(self, image, prompt, response=None):
        self.refresh_chat()
        
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to("cuda")

        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        
        context_prompt = self.conv.get_prompt()
        
        if response is None:
            response = "A" 

        full_prompt = context_prompt + response + "</s>"
        
        context_ids = tokenizer_image_token(context_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        full_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        response_start_idx = len(context_ids)
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=full_ids,
                images=image_tensor,
                return_dict=True
            )
            
        logits = outputs.logits.squeeze(0)
        
        shift_logits = logits[response_start_idx - 1 : -1, :]
        shift_labels = full_ids[0, response_start_idx:]
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        return gathered_log_probs.sum().item()