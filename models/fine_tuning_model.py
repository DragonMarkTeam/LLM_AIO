import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel

class SharedModel(nn.Module):
    def __init__(self, pretrain_model):
        if pretrain_model == "gpt2":
            self.pretrain_model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese')
        elif pretrain_model == "llama-2":
            self.pretrain_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        for param in self.pretrain_model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

            self.pretrain_model.gradient_checkpointing_enable()  # reduce number of stored activations
            self.pretrain_model.enable_input_require_grads()

    def forward(self, input, ):
       return 
   
    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )