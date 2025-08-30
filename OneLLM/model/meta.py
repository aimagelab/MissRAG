from typing import List
import torch
import torch.nn as nn
import json
import os
from .tokenizer import Tokenizer
from . import LLM

from fairscale.nn.model_parallel import initialize as fs_init


class MetaModel(nn.Module):

    def __init__(self, llama_type, llama_config, llama_ckpt_dir=None, tokenizer_path=None):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        ModelArgs = LLM.__dict__[llama_type].ModelArgs
        Transformer = LLM.__dict__[llama_type].Transformer

        with open(llama_config, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=2048, max_batch_size=32, **params
        )
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words

        model = Transformer(model_args)
        mp_rank = fs_init.get_model_parallel_rank()
        if llama_ckpt_dir is not None:
            ckpt_path = os.path.join(llama_ckpt_dir, f"consolidated.{mp_rank:02d}.pth")
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                msg = model.load_state_dict(checkpoint, strict=False)
                print(msg)
            else:
                print(f'Checkpoint not found at {ckpt_path}')
        self.llma = model
        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameter count : {count}")

    def forward(self, examples, labels, image=None, modal='image'):
        output = self.llma(examples, image=image, modal=modal)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())

        return c_loss

    def generate(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
    ) -> List[str]:
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal=modal)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
    
    def generate_multimodal(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
        allowed_token_ids = [], # if empty, all tokens are allowed - if used use batch_size = 1
        use_normal_attention = False,
        return_tokens=False,
    ):
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        
        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            if len(t) > params.max_seq_len:
                print(f"Warning: Truncating prompt from {len(t)} to {params.max_seq_len} tokens.")
            tokens[k, : min(len(t), params.max_seq_len)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_multimodal_inference(tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, use_normal_attention=use_normal_attention)
            if torch.isnan(logits).any():
                print("logits Contiene NaN")
            if torch.isinf(logits).any():
                print("logits Contiene Inf")
            if len(allowed_token_ids) > 0:
                logit_mask = torch.full_like(logits, -float('inf'))
                logit_mask[:, allowed_token_ids] = 0.0
                logits = logits + logit_mask

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if torch.isnan(probs).any():
                    print("probs Contiene NaN")
                if torch.isinf(probs).any():
                    print("probs Contiene Inf")
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated -> If condition is True, the value from x is selected; otherwise, the value from y is taken.
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            if return_tokens:
                out = []
                for token in t[len(prompt_tokens[i]):]:
                    if token == 835 or token == 2277: # '###', '##'
                        break
                    out.append(token)
                decoded.append(out)
            else:
                decoded.append(self.tokenizer.decode(t))
        return decoded

    def generate_multimodal_with_attention_blocking(
        self,
        prompts: List[str],
        answers: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
        block_types: List[str] = None,
        k: int = 9,
        use_normal_attention = False,
    ) -> List[int]:
        
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        """
        prompt_tokens = []
        answer_tokens=[]
        for x, y in zip(prompts, answers):
            prompt_toks = self.tokenizer.encode(x, bos=True, eos=False) 
            answer_toks = self.tokenizer.encode(y, bos=False, eos=False) 
            if len(answer_toks)>2 or len(answer_toks)==0:
                print("Tokens della risposta devono essere maggiori di zero e non devono essere maggiori di due")
                raise ValueError
            if len(answer_toks)==2:
                if answer_toks[0]==29871:
                    prompt_toks.append(29871)
                else:
                    print("Primo Token sbagliato")
                    raise ValueError
                answer_toks = answer_toks[1:2]
            prompt_tokens.append(prompt_toks)
            answer_tokens.append(answer_toks)
            """

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        answer_tokens = [self.tokenizer.encode(x, bos=False, eos=False) for x in answers]
        answer_tokens = torch.tensor(answer_tokens, device=images[0].device).unsqueeze(0)  # (1, B, n_tokens)

        """
        answer_tokens_capitalized = torch.tensor(
            [self.tokenizer.encode(x.capitalize(), bos=False, eos=False) for x in answers], 
            device=images[0].device
        ).unsqueeze(0)  # (1, B, n_tokens=1)
        assert answer_tokens_capitalized.shape[2] == 1, "Capitalized tokens della risposta devono essere uno"
        """

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        
        word_to_digit = {
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9'
        }
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        
        for j, t in enumerate(prompt_tokens):
            tokens[j, : min(len(t), params.max_seq_len)] = torch.tensor(t).long()

        #input_text_mask = tokens != self.tokenizer.pad_id
        prob_layers_type = {}
        if block_types is not None:
            for block_type in block_types:
                start_pos = max_prompt_size
                prev_pos = 0
                prompt_lengths = torch.tensor([len(t)-1 for t in prompt_tokens]).to(images[0].device) #bos not included, (B, )
                
                if block_type in ['last_to_last', 'video_to_last', 'audio_to_last', 'question_to_last', 'video_to_question', 'audio_to_question', 'video_to_audio', 'full_attention', 'audio_to_video']:
                    for i in range(answer_tokens.shape[2]):
                        answer_token = answer_tokens[:, :, i].unsqueeze(-1)                                                     # (1, B, 1)
                        layers_logits = self.llma.forward_multimodal_inference_with_attention_blocking(tokens[:, prev_pos:start_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, block_type=block_type, k=k, prompt_lengths=prompt_lengths, use_normal_attention=use_normal_attention)  # (n_layers, B, vocab_size)
                    
                        if block_type in ['last_to_last', 'video_to_last', 'audio_to_last', 'question_to_last', 'video_to_question', 'audio_to_question', 'video_to_audio', 'audio_to_video']:
                            probs = torch.softmax(layers_logits / temperature if temperature > 0 else layers_logits, dim=-1)    # (n_layers, B, vocab_size)
                            indices = answer_token.expand_as(layers_logits[..., :answer_token.shape[2]])                        # (n_layers, B, 1)
                            new_prob = probs.gather(dim=2, index=indices).squeeze(-1).transpose(0, 1)                           # (n_layers, B, 1) -> (B, n_layers)
                            
                            if block_type in prob_layers_type:
                                prob_layers_type[block_type] = prob_layers_type[block_type] * new_prob
                            else:
                                prob_layers_type[block_type] = new_prob  
                        
                        elif block_type == 'full_attention':
                            probs = torch.softmax(layers_logits / temperature if temperature > 0 else layers_logits, dim=-1)    # (B, vocab_size)
                            indices = answer_token.squeeze(0)                                                                   # (B, n_tokens=1)
                            new_prob = probs.gather(dim=1, index=indices).squeeze(-1)                                           # (B, 1) -> (B)
                            
                            if block_type in prob_layers_type:
                                prob_layers_type[block_type] = prob_layers_type[block_type] * new_prob
                            else:
                                prob_layers_type[block_type] = new_prob
                        else:
                            raise NotImplementedError

                        tokens[:, start_pos] = answer_tokens[:, 0, i] #(B, tot_len)
                        start_pos += 1 
                        prompt_lengths += 1     


                elif block_type == 'cap_to_uncap':
                    answer_tokens_capitalized = torch.tensor(
                        [self.tokenizer.encode(x.capitalize(), bos=False, eos=False) for x in answers], 
                        device=images[0].device
                    ).unsqueeze(0)  # (1, B, n_tokens=1)
                    assert answer_tokens_capitalized.shape[2] == 1, "Capitalized tokens della risposta devono essere uno"
                    
                    layers_logits = self.llma.forward_multimodal_inference_string_number(tokens[:, prev_pos:start_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, prompt_lengths=prompt_lengths, use_normal_attention=use_normal_attention)  # (n_layers, B, vocab_size)
                    probs = torch.softmax(layers_logits / temperature if temperature > 0 else layers_logits, dim=-1) # (n_layers, B, vocab_size)
                    
                    indices_uncapitalized = answer_tokens.expand_as(layers_logits[..., :answer_tokens.shape[2]])  # (n_layers, B, n_tokens=1)
                    indices_capitalized = answer_tokens_capitalized.expand_as(layers_logits[..., :answer_tokens_capitalized.shape[2]])  # (n_layers, B, n_tokens=1)
                    
                    prob_layers_type['capitalized'] = probs.gather(dim=2, index=indices_capitalized).squeeze(-1).transpose(0, 1)  # (n_layers, B, 1) -> (B, n_layers)
                    prob_layers_type['uncapitalized'] = probs.gather(dim=2, index=indices_uncapitalized).squeeze(-1).transpose(0, 1)  # (n_layers, B, 1) -> (B, n_layers)

                elif block_type == 'digit_to_word':
                    answer_tokens_number = torch.tensor(
                        [self.tokenizer.encode(word_to_digit.get(x.lower()), bos=False, eos=False) for x in answers], 
                        device=images[0].device
                    ).unsqueeze(0)  # (1, B, n_tokens=1)
                    
                    layers_logits = self.llma.forward_multimodal_inference_string_number(tokens[:, prev_pos:start_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, prompt_lengths=prompt_lengths, use_normal_attention=use_normal_attention)  # (n_layers, B, vocab_size)
                    probs = torch.softmax(layers_logits / temperature if temperature > 0 else layers_logits, dim=-1) # (n_layers, B, vocab_size)
                    
                    indices_string = answer_tokens.expand_as(layers_logits[..., :answer_tokens.shape[2]])  # (n_layers, B, n_tokens=1)
                    prob_layers_type['string_number'] = probs.gather(dim=2, index=indices_string).squeeze(-1).transpose(0, 1)  # (n_layers, B, 1) -> (B, n_layers)

                    answer_token_number = answer_tokens_number[:, :, 0].unsqueeze(0)
                    indices_number = answer_token_number.expand_as(layers_logits[..., :answer_token_number.shape[2]])  # (n_layers, B, n_tokens=1)
                    prob_layers_type['digit_number'] = probs.gather(dim=2, index=indices_number).squeeze(-1).transpose(0, 1)  # (n_layers, B, 1) -> (B, n_layers)

                    tokens[:, start_pos] = answer_tokens_number[:, 0, 0] #(B, tot_len)
                    start_pos += 1 
                    prompt_lengths += 1  #(B, )  

                    layers_logits = self.llma.forward_multimodal_inference_string_number(tokens[:, prev_pos:start_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, prompt_lengths=prompt_lengths, use_normal_attention=use_normal_attention)  # (n_layers, B, vocab_size)
                    probs = torch.softmax(layers_logits / temperature if temperature > 0 else layers_logits, dim=-1) # (n_layers, B, vocab_size)
                    
                    answer_token_number = answer_tokens_number[:, :, 1].unsqueeze(0)
                    indices_number = answer_token_number.expand_as(layers_logits[..., :answer_token_number.shape[2]])  # (n_layers, B, n_tokens=1)
                    new_prob = probs.gather(dim=2, index=indices_number).squeeze(-1).transpose(0, 1)  # (n_layers, B, 1) -> (B, n_layers)
                    prob_layers_type['digit_number'] = prob_layers_type['digit_number'] * new_prob
                else:
                    raise NotImplementedError
                           

        print(list(prob_layers_type.keys()))
        return prob_layers_type 

    def generate_multimodal_oneshot(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
        allowed_token_ids = [], # if empty, all tokens are allowed - if used use batch_size = 1
        use_normal_attention=False
    ) -> List[str]:
        bsz = len(prompts)
        assert bsz == 1 # batch_size must be 1
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : min(len(t), params.max_seq_len)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # for cur_pos in range(start_pos, total_len):
        logits = self.llma.forward_multimodal_inference(tokens[:, prev_pos:start_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, use_normal_attention=use_normal_attention) #torch.randn(bsz, 32000).cuda() 
        if len(allowed_token_ids) > 0:
            logit_mask = torch.full_like(logits, -float('inf'))
            logit_mask[:, allowed_token_ids] = 0.0
            logits = logits + logit_mask

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated -> If condition is True, the value from x is selected; otherwise, the value from y is taken.
        next_token = torch.where(
            input_text_mask[:, start_pos], tokens[:, start_pos], next_token
        )
        tokens[:, start_pos] = next_token
        tokens[:, start_pos+1:] = self.tokenizer.eos_id
        prev_pos = start_pos
        # print(tokens)

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded    

    def generate_multimodal_allow_earlystop(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
        allowed_token_ids = [], # if empty, all tokens are allowed - if used use batch_size = 1
        use_normal_attention=False
    ) -> List[str]:
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # Initialize a boolean mask for the batch
        initial_mask = torch.ones([bsz], dtype=torch.bool)
        counter = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_multimodal_inference(tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, use_normal_attention=use_normal_attention)
            if len(allowed_token_ids) > 0:
                logit_mask = torch.full_like(logits, -float('inf'))
                logit_mask[:, allowed_token_ids] = 0.0
                logits = logits + logit_mask

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            

            # Determine which sequences have produced the EOS token at the last position
            actual_mask = next_token == self.tokenizer.eos_id
            # Increase counter by the number of sequences that just ended
            counter += (initial_mask & actual_mask).sum()
            # Update the mask: for sequences that ended, mark them as finished
            initial_mask = initial_mask & (~actual_mask)            

            # only replace token if prompt has already been generated -> If condition is True, the value from x is selected; otherwise, the value from y is taken.
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            # If all sequences have ended, break
            if counter == bsz:
                break

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded        
    
    def generate_multimodal_retrieve(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
        retrieved_tokens = {},
        use_normal_attention=False
    ) -> List[str]:
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_multimodal_inference_retrieve(tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, retrieved_tokens=retrieved_tokens, use_normal_attention=use_normal_attention)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded        
    
    def generate_multimodal_retrieve_oneshot(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
        retrieved_tokens = {},
        allowed_token_ids = [],
        use_normal_attention = False
    ) -> List[str]:
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : min(len(t), params.max_seq_len)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        # for cur_pos in range(start_pos, total_len):
        logits = self.llma.forward_multimodal_inference_retrieve(tokens[:, prev_pos:start_pos], prev_pos, images if prev_pos == 0 else None, modal=modal, retrieved_tokens=retrieved_tokens, use_normal_attention=use_normal_attention)
        if len(allowed_token_ids) > 0:
            logit_mask = torch.full_like(logits, -float('inf'))
            logit_mask[:, allowed_token_ids] = 0.0
            logits = logits + logit_mask
        
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, start_pos], tokens[:, start_pos], next_token
        )
        tokens[:, start_pos] = next_token
        tokens[:, start_pos+1:] = self.tokenizer.eos_id
        prev_pos = start_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded         
  
    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
    ):
        params = self.llma.params

        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        # truncate from the left. leave some space for generation.
        max_seq_len = params.max_seq_len
        if images is not None:
            max_seq_len -= self.llma.image_words

        max_prompt_size = max_seq_len - max_gen_len
        prompt_tokens = prompt_tokens[-max_prompt_size:]

        prompt_size = len(prompt_tokens)

        total_len = min(max_seq_len, max_gen_len + prompt_size)

        tokens = torch.full([total_len], 0).cuda().long()

        tokens[:len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        start_pos = prompt_size
        prev_pos = 0
        generate_until = start_pos
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(tokens[None, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal = modal)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.item()

            if next_token == self.tokenizer.eos_id:
                break

            tokens[cur_pos] = next_token
            prev_pos = cur_pos
            generate_until = cur_pos + 1
            yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": False}

        yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": True}

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def get_image_words(self):
        return self.llma.image_words