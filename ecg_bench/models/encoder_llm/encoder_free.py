import torch
import torch.nn as nn

class EncoderFree(nn.Module):
    def __init__(self, llm, projection_dim, llm_tokenizer, patch_size=None):
        super(EncoderFree, self).__init__()
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.num_leads = 12
        
        # linear projection from flattened entire signal to LLM hidden size
        self.signal_projection = nn.Linear(
            projection_dim, 
            self.llm.llm.config.hidden_size
        ).to(dtype=self.llm.llm.dtype, device=self.llm.llm.device)
            
    def forward(self, batch):
        # Flatten entire signal and project to single embedding
        signal_flat = batch['signal'].reshape(batch['signal'].shape[0], -1)  # (batch_size, 12*seq_len)
        projected_embed = self.signal_projection(signal_flat.to(dtype=self.llm.llm.dtype, device=self.llm.llm.device))  # (batch_size, hidden_size)
        projected_embed = projected_embed.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Get LLM embeddings and replace signal token
        llm_embeddings = self.llm.get_llm_embeddings(batch['input_ids'])
        
        # Insert projected signal embedding at the signal token position
        signal_start_idx = batch['signal_start_idx']
        
        # signal_start_idx is a tensor of shape (batch_size,) 
        for i in range(projected_embed.shape[0]):  # batch dimension
            start_idx = signal_start_idx[i].item()
            llm_embeddings[i, start_idx:start_idx+1, :] = projected_embed[i]
        
        batch['inputs_embeds'] = llm_embeddings
        out = self.llm(batch)
        return out
    
    
    def generate_chat(self, input_ids, attention_mask, tokenizer, signal=None, signal_start_idx=None):
        # Flatten entire signal and project to single embedding
        signal_flat = signal.reshape(signal.shape[0], -1)  # (batch_size, 12*seq_len)
        projected_embed = self.signal_projection(signal_flat.to(dtype=self.llm.llm.dtype, device=self.llm.llm.device))  # (batch_size, hidden_size)
        projected_embed = projected_embed.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Get LLM embeddings and replace signal token
        llm_embeddings = self.llm.get_llm_embeddings(input_ids)
        
        # Insert projected signal embedding
        if isinstance(signal_start_idx, int):
            # Single sample case for inference
            llm_embeddings[0, signal_start_idx:signal_start_idx+1, :] = projected_embed[0]
        else:
            # Batch case 
            for i in range(projected_embed.shape[0]):
                start_idx = signal_start_idx[i].item() if torch.is_tensor(signal_start_idx[i]) else signal_start_idx[i]
                llm_embeddings[i, start_idx:start_idx+1, :] = projected_embed[i]
        
        out = self.llm.generate_chat(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            inputs_embeds=llm_embeddings
        )
        return out