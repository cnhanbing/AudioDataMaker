import torch
import torchaudio

from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

class AudioSeparator:
    def __init__(self, window_size=30, stride=28):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        self.model=HDEMUCS_HIGH_MUSDB_PLUS.get_model().to(self.device)
        self.target_sr = 44100
        self.window_size = window_size * self.target_sr  
        self.stride = stride * self.target_sr 
    
    def separate_audio(self, input_file):
        # 加载混合音频文件
        mixture, sample_rate = torchaudio.load(input_file)
        mixture = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)(mixture)
        mixture = mixture.to(self.device)

        # 分段处理
        output_segments = []
        for i in range(0, mixture.shape[1] - self.window_size + 1, self.stride):
            segment = mixture[:, i:i+self.window_size].unsqueeze(0).to(self.device)  # 添加批次维度
            with torch.no_grad():
                estimated_sources = self.model(segment)
                estimated_sources = estimated_sources.squeeze(0)
            output_segments.append(estimated_sources.cpu())

        # 使用重叠加法处理分段的输出
        if len(output_segments) > 0:
            # result = torch.cat(output_segments, dim=2)
            total_length = mixture.shape[1]
            result = self.overlap_add(output_segments, total_length)

            vocals=result[3,:,:]
            # print(vocals.shape)
            # 保存分离的音频源
            # torchaudio.save(output_file, vocals, self.target_sr)
            # print('*' * 100)
            return vocals,self.target_sr#,output_file
        else:
            print("No segments processed.")

    def overlap_add(self, segments, total_length):
        # 创建一个全零的张量作为输出
        output = torch.zeros(segments[0].shape[0], segments[0].shape[1], total_length)        
        # 逐个段添加到输出中
        start = 0
        for segment in segments:
            segment_length = segment.size(-1)
            end = start + segment_length
            
            # 将当前段加入输出中
            output[:, :, start:end] += segment
            
            # 更新下一个段的起始位置
            start += self.stride
        
        # 返回重叠添加后的结果
        return output  