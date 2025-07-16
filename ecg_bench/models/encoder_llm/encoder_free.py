import torch
import torch.nn as nn

class EncoderFree(nn.Module):
    def __init__(self, llm, projection_dim, llm_tokenizer, patch_size=10):
        super(EncoderFree, self).__init__()
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.patch_size = patch_size
        self.num_leads = 12
        
        # linear projection from flattened patch to LLM hidden size
        self.signal_projection = nn.Linear(
            self.num_leads * self.patch_size, 
            self.llm.llm.config.hidden_size
        ).to(dtype=self.llm.llm.dtype, device=self.llm.llm.device)
            
    def forward(self, batch):
        # Get signal patches and project them
        signal_patches = self.create_patches(batch['signal'])  # (batch_size, num_patches, 12, patch_size)
        projected_embeds = self.project_patches(signal_patches)  # (batch_size, num_patches, hidden_size)
        
        # Get LLM embeddings and replace signal tokens
        llm_embeddings = self.llm.get_llm_embeddings(batch['input_ids'])
        
        # Insert projected signal embeddings at the signal token positions
        signal_start_idx = batch['signal_start_idx']
        num_patches = projected_embeds.shape[1]
        
        # signal_start_idx is a tensor of shape (batch_size,) 
        for i in range(projected_embeds.shape[0]):  # batch dimension
            start_idx = signal_start_idx[i].item()
            llm_embeddings[i, start_idx:start_idx+num_patches, :] = projected_embeds[i]
        
        batch['inputs_embeds'] = llm_embeddings
        out = self.llm(batch)
        return out
    
    def create_patches(self, signal):
        """
        Create non-overlapping patches from the signal
        signal: (batch_size, 12, seq_len)
        returns: (batch_size, num_patches, 12, patch_size)
        """
        batch_size, num_leads, seq_len = signal.shape
        num_patches = seq_len // self.patch_size
        
        # Truncate signal to fit exact number of patches
        truncated_len = num_patches * self.patch_size
        signal = signal[:, :, :truncated_len]
        
        # Reshape to create patches
        patches = signal.reshape(batch_size, num_leads, num_patches, self.patch_size)
        patches = patches.permute(0, 2, 1, 3)  # (batch_size, num_patches, num_leads, patch_size)
        
        return patches
    
    def project_patches(self, patches):
        """
        Project each patch to LLM embedding dimension
        patches: (batch_size, num_patches, 12, patch_size)
        returns: (batch_size, num_patches, hidden_size)
        """
        batch_size, num_patches, num_leads, patch_size = patches.shape
        
        # Flatten each patch
        patches_flat = patches.reshape(batch_size, num_patches, num_leads * patch_size)
        
        # Project to LLM dimension
        projected = self.signal_projection(patches_flat.to(dtype=self.llm.llm.dtype, device=self.llm.llm.device))
        
        return projected
    
    def generate_chat(self, input_ids, attention_mask, tokenizer, signal=None, signal_start_idx=None):
        # Get signal patches and project them
        signal_patches = self.create_patches(signal)
        projected_embeds = self.project_patches(signal_patches)
        
        # Get LLM embeddings and replace signal tokens
        llm_embeddings = self.llm.get_llm_embeddings(input_ids)
        
        # Insert projected signal embeddings
        num_patches = projected_embeds.shape[1]
        if isinstance(signal_start_idx, int):
            # Single sample case for inference
            llm_embeddings[0, signal_start_idx:signal_start_idx+num_patches, :] = projected_embeds[0]
        else:
            # Batch case 
            for i in range(projected_embeds.shape[0]):
                start_idx = signal_start_idx[i].item() if torch.is_tensor(signal_start_idx[i]) else signal_start_idx[i]
                llm_embeddings[i, start_idx:start_idx+num_patches, :] = projected_embeds[i]
        
        out = self.llm.generate_chat(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            inputs_embeds=llm_embeddings
        )
        return out