import torch
import torch.nn as nn

class EncoderFree(nn.Module):
    def __init__(self, llm, projection_dim, llm_tokenizer):
        super(EncoderFree, self).__init__()
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        
        # linear projection from flattened entire signal to LLM hidden size
        self.signal_projection = nn.Linear(
            projection_dim, 
            self.llm.llm.config.hidden_size
        ).to(dtype=self.llm.llm.dtype, device=self.llm.llm.device)
        
        # LayerNorm to match LLM's embedding distribution
        self.layer_norm = nn.LayerNorm(
            self.llm.llm.config.hidden_size
        ).to(dtype=self.llm.llm.dtype, device=self.llm.llm.device)
            
    def forward(self, batch):
        # Flatten entire signal: (B, 12, seq_len) -> (B, 12*seq_len)
        # Confirmed reshaping correctly
        signal_flat = batch['signal'].reshape(batch['signal'].shape[0], -1)  # (batch_size, 12*seq_len)
        projected_embed = self.signal_projection(signal_flat.to(dtype=self.llm.llm.dtype, device=self.llm.llm.device))  # (batch_size, hidden_size)
        projected_embed = self.layer_norm(projected_embed)  # Normalize to match LLM embedding distribution
        
        # Get LLM embeddings and replace signal token
        llm_embeddings = self.llm.get_llm_embeddings(batch['input_ids'])
        
        # Insert projected signal embedding at the signal token position
        batch_indices = torch.arange(projected_embed.shape[0], device=projected_embed.device)
        llm_embeddings[batch_indices, batch['signal_id_index'], :] = projected_embed
        
        batch['inputs_embeds'] = llm_embeddings
        out = self.llm(batch)
        return out
    
    def generate_chat(self, input_ids, attention_mask, tokenizer, signal=None, signal_id_index=None):
        # Flatten entire signal: (B, 12, seq_len) -> (B, 12*seq_len)
        # Confirmed reshaping correctly
        signal_flat = signal.reshape(signal.shape[0], -1)  # (batch_size, 12*seq_len)
        projected_embed = self.signal_projection(signal_flat.to(dtype=self.llm.llm.dtype, device=self.llm.llm.device))  # (batch_size, hidden_size)
        projected_embed = self.layer_norm(projected_embed)  # Normalize to match LLM embedding distribution
        
        # Get LLM embeddings and replace signal token
        llm_embeddings = self.llm.get_llm_embeddings(input_ids)
        
        # Insert projected signal embedding (like second-stage)
        batch_indices = torch.arange(projected_embed.shape[0], device=projected_embed.device)
        llm_embeddings[batch_indices, signal_id_index, :] = projected_embed
        
        out = self.llm.generate_chat(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            inputs_embeds=llm_embeddings
        )
        return out