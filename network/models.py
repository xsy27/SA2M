import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


class MotionDiffusion(nn.Module):
    def __init__(self, pose_vec, vec_len, audio_dim, clip_len=240,
                 latent_dim=512, ff_size=1024, num_layers=4, num_heads=8, dropout=0.2,
                 ablation=None, activation="gelu", legacy=False, 
                 arch='trans_enc', cond_mask_prob=0.15, device=None):
        super().__init__()

        self.legacy = legacy
        self.training = True
        
        self.pose_vec = pose_vec
        self.vec_len = vec_len
        self.audio_dim = audio_dim
        self.clip_len = clip_len

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.cond_mask_prob = cond_mask_prob
        self.arch = arch
        
        # self.traj_trans_process = TrajProcess(2, self.latent_dim)
        # self.traj_pose_process = TrajProcess(6, self.latent_dim)
        # self.style_feature_process = nn.Linear(512, self.latent_dim)
        # self.embed_style = EmbedStyle(nstyles, self.latent_dim)
        self.motion_process = MotionProcess(self.vec_len, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_audio = AudioEmbedder(self.audio_dim, self.latent_dim)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqEncoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == 'gru':
            print("GRU init")
            self.seqEncoder = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
      
        self.output_process = OutputProcess(self.vec_len, self.latent_dim)


    def forward(self, x, timesteps, music_feature=None, y=None):
        bs, vec_len, nframes = x.shape
        if y != None: music_feature = y['music']

        time_emb = self.embed_timestep(timesteps)  # [1, bs, L]
        audio_emb = self.embed_audio(music_feature) # [nframes, bs, L]
        motion_emb = self.motion_process(x) # [nframes, bs, L]
        
        xseq = torch.cat((time_emb, audio_emb, motion_emb), axis=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:]
        output = self.output_process(output)
        return output
        

    def interface(self, x, timesteps, y=None):
        """
            x: [batch_size, vec_len, nframes], denoted x_t in the paper 
            timesteps: [batch_size] (int)
            y: a dictionary containing conditions
        """
        bs, vec_len, nframes = x.shape

        music_feature = y['music']

        # style_idx = y['style_idx'] 
        # traj_pose = y['traj_pose']
        # traj_trans = y['traj_trans']
        
        # CFG on past motion
        # keep_batch_idx = torch.rand(bs, device=past_motion.device) < (1-self.cond_mask_prob)
        # past_motion = past_motion * keep_batch_idx.view((bs, 1, 1, 1))
        
        return self.forward(x, timesteps, music_feature=music_feature)

class MotionProcess(nn.Module):
    def __init__(self, vec_len, latent_dim):
        super().__init__()
        self.vec_len = vec_len
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.vec_len, self.latent_dim)

    def forward(self, x):
        bs, vec_len, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.poseEmbedding(x)  
        return x

class AudioEmbedder(nn.Module):
    def __init__(self, audio_dim, latent_dim):
        super().__init__()
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.audioEmbedding = nn.Linear(self.audio_dim, self.latent_dim)
    
    def forward(self, x):
        bs, audio_dim, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.audioEmbedding(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class OutputProcess(nn.Module):
    def __init__(self, vec_len, latent_dim):
        super().__init__()
        self.vec_len = vec_len
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.vec_len)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)
        output = output.reshape(nframes, bs, self.vec_len)
        output = output.permute(1, 2, 0)
        return output
