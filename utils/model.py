import torch
from torch import nn
import numpy as np

class VAE(nn.Module):
    def __init__(self, config):
        """
        Method: __init__(self,config)
        - Purpose: constructor for initializing VAE.
        - Attributes:
            - self.seq_size: number of time steps.                    (INPUT)
            - self.batch_size: batch size.                            (INPUT)
            - self.input_size: number of features in the input.       (LSTM)
            - self.lstm_size: number of features in the hidden state. (LSTM)
            - self.num_layers: number of recurrent layers.            (LSTM)
            - self.directions: number of LSTM directions.             (LSTM)
            - self.latent_size: latent space size.                    (VAE)
            - self.attention_size: latent space size.                 (VSAM)
            - self.sample_rep: number of sampling from latent space.  (INFERENCE)
            - self.batch_norm: 1d batch normalization.
            - self.softmax: softmax activation function.
            - self.SoftP: softplus activation function.
        """
        super(VAE, self).__init__()
        self.config = config
        self.seq_size = config['seq_size']
        self.lstm_size = config['lstm_size']
        self.latent_size = config['latent_size']
        self.attention_size = config['attention_size']
        self.input_size = config['input_size']
        self.batch_size = config['batch_size']
        self.num_layers = config['num_lyears']
        self.directions = config['directions']
        self.sample_reps = config['sample_reps']
        self.training = config['train']
        self.batch_norm = nn.BatchNorm1d(self.input_size)
        self.softmax = nn.Softmax(dim=0)
        self.SoftP=nn.Softplus()

        # Bi-LSTM 인코더
        self.encoder1 = nn.LSTM(self.input_size, self.lstm_size, self.num_layers, bidirectional = True)
        # 인코딩 값으로 VAE 레이어에 전달할 평균, 로그 분산값 생성하는 레이어.
        self.encoder_mu = nn.Linear(self.lstm_size* self.directions, self.latent_size) 
        self.encoder_logvar = nn.Linear(self.lstm_size *self.directions, self.latent_size)
        # 어텐션 레이어에서 디코더로 넘어가기 전 c_det 값들로부터 c_t값 평균과 로그 분산값 생성하는 레이어.
        self.encoder_att_mu = nn.Linear( self.lstm_size * self.directions ,  self.attention_size )
        self.encoder_att_logvar = nn.Linear(self.lstm_size * self.directions ,  self.attention_size )
        # VAE를 통과한 z값들과 VSAM을 통과한 c값들을 디코더에 입력값으로 넣어주기 위한 레이어.
        self.decoder1 = nn.LSTM(self.latent_size + self.attention_size, self.lstm_size, self.num_layers, bidirectional = True)
        # 디코더로부터 아웃풋 값을 생성하는 레이어.
        self.reconstruction_mu = nn.Linear( self.lstm_size * self.directions, self.input_size)
        self.reconstruction_logvar = nn.Linear( self.lstm_size * self.directions, self.input_size)

    def encode(self, x):
        """
        Method: encode(self, x)
        - Parameters:
            - x: batched input sequence
        - Purpose:
            - returns encoded mu and logvar for variational layer, output for VSAM layer.
        """
        # output_encoder: (seq_size, batch_size, lstm_size * 2)
        # h_n: (directions, batch_size, lstm_size), 마지막 시점의 역/순방향 은닉층 값들을 저장.
        output_encoder, (h_n,c_n) = self.encoder1(x)
        # 배치별 역/순방향 은닉층 값들 붙여주기. (batch_size,lstm_size×2)
        h1=torch.cat((h_n[0,:,:], h_n[1,:,:]), 1)
        # Tanh 활성화 함수 적용 후에 미리 구성해둔 encoder_mu, encoder_logvar 통과, 로그 분산 값은 Softplus 활성화 함수 적용.
        # (batch_size, latent_size), (batch_size, latent_size), (seq_size, batch_size, lstm_size * 2)
        return self.encoder_mu(torch.tan(h1)), self.SoftP(self.encoder_logvar(torch.tan(h1))), output_encoder 

    def reparameterize(self, mu, logvar):
        """
        Method: reparameterize(self, mu, logvar)
        - Parameters:
            - mu: average
            - logvar: log variance
        - Purpose:
            - reparameterization trick for back propagation.
            - allows for the generation of latent space values.
        """
        # 인코딩된 값들을 사용해서 z, c와 같은 latent space 변수들을 리턴.
        std = torch.exp(logvar*0.5)
        eps = torch.randn_like(std)
        return mu+eps*std
        
    def attention(self, out_encoder):
        """
        Method: attention(self, out_encoder)
        - Parameters:
            - out_encoder: output from 'encode' method, containing hidden states from bi-LSTM.
        - Purpose:
            - returns deterministic context vectors.
        """
        for batch_idx in range(self.batch_size):
            if batch_idx == 0:
		        # 행렬곱으로 유사도 매트릭스 추출, 정규화.
                s_tensor = torch.div(torch.matmul(out_encoder[:,batch_idx,:], (out_encoder[:,batch_idx,:]).t()) , np.sqrt(self.lstm_size * self.directions))
                # 소프트맥스 활성화 함수. 아웃풋 텐서의 형태는 (seq_size, seq_size)
                a_tensor = self.softmax(s_tensor)
                # 더 이상 연산에 필요없는 유사도 매트릭스는 제거.
                del s_tensor
                # (seq_size, 1, seq_size) * (seq_size, lstm_size*2, 1) 브로드캐스팅 후 2차원 기준 합산.
                c_det = (a_tensor.unsqueeze(1) * out_encoder[:,batch_idx,:].unsqueeze(2)).sum(dim=2)
                # (seq_size, 1, lstm_size*2) 형태로 첫번째 배치에 대한 c_det 값들을 계산. 
                c_det = c_det.unsqueeze(1)
            else:
		        # 나머지 배치에 대해서 똑같은 방식으로 연산을 진행하고, 1차원 기준으로 텐서를 이어붙인다.
                s_score = ((torch.mm(out_encoder[:,batch_idx,:], (out_encoder[:,batch_idx,:]).t())) / np.sqrt(self.lstm_size * self.directions))
                a_score = self.softmax(s_score)
                del s_score
                a_tensor = torch.cat((a_tensor, a_score), 1)
                c_det_batch = (a_tensor.unsqueeze(1) * out_encoder[:,batch_idx,:].unsqueeze(2)).sum(dim=2)
                c_det_batch = c_det_batch.unsqueeze(1)
                # (seq_size, batch_size, lstm_size*2) 
                c_det = torch.cat((c_det, c_det_batch), 1)
        return c_det

    def decode(self, z, c):
        """
        Method: decode(self, z, c)
        - Parameters:
            - z: latent space representation produced by the variational layer.
            - c: context vectors produced by the VSAM layer.
        - Purpose:
            - decodes the concatenated z, c vectors to produce the reconstruction params.
        """
        # z    : (batch_size, latent_size)
        # c    : (seq_size, batch_size, attention_size)
        # zrep : (seq_size, batch_size, latent_size)
        zrep = z.unsqueeze(0).repeat(self.seq_size, 1, 1)
        # zandc: (seq_size, batch_size, latent_size+attention_size)
        zandc = torch.cat((zrep,c), 2)
        # output_decoder: (seq_size, batch_size, lstm_size*2)
        # state_decoder : ((2, batch_size, lstm_size), (2, batch_size, lstm_size))
        output_decoder, state_decoder = self.decoder1(zandc)
        return self.reconstruction_mu(torch.tan(output_decoder)),self.SoftP(self.reconstruction_logvar(torch.tan(output_decoder))),  state_decoder
        
    def forward(self, x):  
        """
        Method: forward(self, x)
        - Parameters:
            - x: batched input sequence (seq_size, batch_size, input_size)
        - Purpose:
            - Implements the full forward pass of the VAE model, including encoding, attention mechanism, and decoding for reconstruction.
            - Handles both training and inference modes.
        """
        # Encode input sequence and extract mu, logvar, and Bi-LSTM output.
        # mu, logvar: latent space parameters from VAE
        # output_encoder: hidden states of the Bi-LSTM (for attention)
        mu, logvar, output_encoder = self.encode(x)
        # Apply attention mechanism to Bi-LSTM output
        # c_det: deterministic context vectors from attention
        c_det = self.attention(output_encoder)
        # Pass the context vectors (c_det) through the attention VAE layer
        # to generate latent space variables (c_t_mu, c_t_logvar) for each timestep
        for t in range(self.seq_size):
            if t == 0:
                # For the first timestep, initialize c_t_mu, c_t_logvar and reparameterize to get c_final
                c_0 = c_det[t, :, :].unsqueeze(0)  # (1, batch_size, lstm_size * 2)
                c_t_mu = self.encoder_att_mu(c_0)  # (1, batch_size, attention_size)
                c_t_mus = c_t_mu
                c_t_logvar = self.SoftP(self.encoder_att_logvar(c_0))  # (1, batch_size, attention_size)
                c_t_logvars = c_t_logvar
                c_final = self.reparameterize(c_t_mu, c_t_logvar)  # Reparameterized latent vector
                del c_0
            else:
                # For subsequent timesteps, concatenate the latent variables
                c_current = c_det[t, :, :].unsqueeze(0)  # (1, batch_size, lstm_size * 2)
                c_t_mu = self.encoder_att_mu(c_current)  # (1, batch_size, attention_size)
                c_t_logvar = self.SoftP(self.encoder_att_logvar(c_current))  # (1, batch_size, attention_size)
                c_final = torch.cat((c_final, self.reparameterize(c_t_mu, c_t_logvar)), 0)  # Concatenate over time
                # Store all mus and logvars for monitoring purposes
                c_t_mus = torch.cat((c_t_mu, c_t_mus), 0)  # Concatenate all mus
                c_t_logvars = torch.cat((c_t_logvar, c_t_logvars), 0)  # Concatenate all logvars

        # If training, sample z and reconstruct the output
        if self.training:
            # Sample latent variable z using the reparameterization trick
            z = self.reparameterize(mu, logvar)  # (batch_size, latent_size)
            # Decode z and c_final to get the reconstructed output
            output_mu, output_logvar, state_decoder = self.decode(z, c_final)
            # Use the mean output as the final reconstruction
            output = output_mu
            # Return all relevant variables for loss computation and analysis
            return output, mu, logvar, output_encoder, state_decoder, c_det, z, c_final, c_t_mus, c_t_logvars, output_mu
        # If not training (inference mode), sample L times from z and generate multiple reconstructions
        else:
            for l in range(self.sample_reps):
                # Sample latent variable z for each iteration
                z = self.reparameterize(mu, logvar)
                if l == 0:
                    # Decode the first sampled z and store the output
                    output_mu, output_logvar, state_decoder = self.decode(z, c_final)
                    output_all = output_mu.unsqueeze(0)  # (1, seq_size, batch_size, input_size)
                    output = output_mu  # Store the first output
                else:
                    # Decode and concatenate subsequent outputs
                    output_mu, output_logvar, state_decoder = self.decode(z, c_final)
                    output_all = torch.cat((output_all, output_mu.unsqueeze(0)), 0)  # (L, seq_size, batch_size, input_size)
            # Return the final output and all the sampled reconstructions
            return output, mu, logvar, output_encoder, state_decoder, c_det, z, c_final, c_t_mus, c_t_logvars, output_mu, output_all