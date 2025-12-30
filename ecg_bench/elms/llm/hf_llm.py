from torch import nn


### HuggingFace LLM Wrapper
class HuggingFaceLLM(nn.Module):
    def __init__(self, llm, pad_token_id, eos_token_id, output_hidden_states=False, max_new_tokens=512):
        super(HuggingFaceLLM, self).__init__()
        self.llm = llm
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_hidden_states = output_hidden_states
        self.max_new_tokens = max_new_tokens

    def forward(self, batch):
        device = self.llm.device
        kwargs = {
            "attention_mask": batch["elm_attention_mask"].to(device),
            "output_hidden_states": self.output_hidden_states,
        }

        if "elm_inputs_embeds" in batch and batch["elm_inputs_embeds"] is not None:
            kwargs["inputs_embeds"] = batch["elm_inputs_embeds"].to(device)
        else:
            kwargs["input_ids"] = batch["elm_input_ids"].to(device)

        if "elm_labels" in batch:
            kwargs["labels"] = batch["elm_labels"].to(device)
        return self.llm(**kwargs)

    def get_llm_embeddings(self, elm_input_ids):
        out = self.llm.get_input_embeddings()(elm_input_ids.to(self.llm.device))
        return out

    def generate(self, batch, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if "elm_inputs_embeds" in batch and batch["elm_inputs_embeds"] is not None:
            out = self.llm.generate(
                inputs_embeds=batch["elm_inputs_embeds"].to(self.llm.device),
                attention_mask=batch["elm_attention_mask"].to(self.llm.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        else:
            out = self.llm.generate(
                input_ids=batch["elm_input_ids"].to(self.llm.device),
                attention_mask=batch["elm_attention_mask"].to(self.llm.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        return out
