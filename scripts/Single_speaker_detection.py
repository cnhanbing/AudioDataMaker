from pyannote.audio import Pipeline
import torch
import torchaudio
from collections import defaultdict



def is_single_speaker(audio_file_path, model_name, use_auth_token, use_cuda=True):
    try:
        # 实例化Speaker Diarization流水线
        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=use_auth_token)

        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # 将流水线推断到GPU上（如果需要的话）
        if use_cuda:
            pipeline.to(torch.device("cuda"))

        # 运行流水线来执行说话者分离
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate})

        # 统计不同的说话者数量
        speaker_count = defaultdict(int)

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_count[speaker] += 1

        # 判断是否只有一个说话者
        return len(speaker_count) == 1, waveform.shape[-1]/sample_rate

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False



