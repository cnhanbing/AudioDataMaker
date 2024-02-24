
import os
import shutil
import subprocess
import sys

import uuid
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torchaudio
from torchaudio.transforms import Resample

from utils import (VideoIDExtractor, Down_Audio, convert_audio, get_channel_id,
                   get_channel_videos, copy_most_common_speaker_wav)
from scripts.HDemucs_plus import AudioSeparator
from scripts.Audio_Slicer import Slicer
from scripts.whisper import Whisper_ASR
from scripts.skeaker_V import (speaker_verification, Speaker_Verification)
from scripts.Single_speaker_detection import is_single_speaker


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args: DictConfig) -> None:

    # 根据配置文件建立临时目录
    DIR=args.dir
    for _, value in DIR.items():
        os.makedirs(value, exist_ok=True)

    # 定义数据集索引的保存位置
    ASR_results=Path(DIR.results)/'datamate.txt'
    error_log_file=Path(DIR.results)/'error_log.txt'    

    # 通过YouTube网址获取视频ID
    Pick_id=VideoIDExtractor()
    video_ids=Pick_id.process_file_list(args.base.videoUrl)
 
    # 下载YouTube音频
    down_file=[]
    for video_id in video_ids:
        audio_path=Down_Audio(video_id=video_id,output_folder=DIR.youtube)
        down_file.append(audio_path)

    # 使用HDemucs去除背景音乐及一定的噪音
    vocals=[]
    inter_results=AudioSeparator()
    for file in down_file:
        waveform,sr=inter_results.separate_audio(file)
        vocals.append((waveform,sr))

    # 静音裁切音频 引用https://github.com/openvpi/audio-slicer
    for waveform in vocals:
        vocal,sr=waveform
        slicer = Slicer(
            sr=sr,
            threshold=args.slicer.threshold,
            min_length=args.slicer.min_length,
            min_interval=args.slicer.min_interval,
            hop_size=args.slicer.hop_size,
            max_sil_kept=args.slicer.max_sil_kept
        )
        chunks = slicer.slice(vocal)
        
        for i, chunk in enumerate(chunks):
            # 采样率转换，下面的判断是否为单人说话需要16K的音频
            if sr != 16000:
                resampler = Resample(sr, 16000)
                resampled_chunk = resampler(chunk)
                chunk = resampled_chunk.squeeze(0)
            if chunk.shape[0] != 1 :
                chunk = chunk.mean(dim=0, keepdim=True)
            # 构建输出文件路径
            output_file = os.path.join(DIR.segment, f'{uuid.uuid4()}.wav')
            # 将静音段保存为 WAV 文件
            torchaudio.save(output_file, chunk, 16000)

    # 使用pyannote/speaker-diarization-3.1判断是否是单人说话
    audio_files=list(Path(DIR.segment).glob('**/*.wav'))
    for i,audio_file_path in enumerate(audio_files):
        result =is_single_speaker(audio_file_path, 
                                  args.single_speaker.model_name, 
                                  args.single_speaker.HF_Access_Tokens, 
                                  args.single_speaker.use_cuda)
        if result[0]:
            shutil.copy2(audio_file_path, DIR.solo_voice)
            print(f'已经将{audio_file_path}复制到{DIR.solo_voice}')

    # 分类器，将说话人说得最多的音频复制到指定目录
    speaker_verification(DIR.solo_voice,Path(DIR.root)/'Speaker_class.txt')
    copy_most_common_speaker_wav(record_file=Path(DIR.root)/'Speaker_class.txt',
                                 wav_directory=DIR.solo_voice,
                                 destination_directory=DIR.Speech_V)
     
    #修改名字
    files=os.listdir(DIR.Speech_V)
    renamed_files = []
    for file_counter, filename in enumerate(files):
        if filename.endswith(".wav"):
            new_filename = f"{args.sperker_info.ID}-{file_counter:04}.wav"
            new_file_path = Path(DIR.Speech_V).joinpath(new_filename)
            old_file_path = Path(DIR.Speech_V).joinpath(filename)
            old_file_path.rename(new_file_path)
            renamed_files.append(new_file_path)
        
    #将语音转文字
    Whisper_ASR(input_directory=DIR.Speech_V,
                output_file=ASR_results,
                error_log_file=error_log_file,
                id=args.sperker_info.ID,
                idx=args.sperker_info.index)
    

    audio_files=list(Path(DIR.Speech_V).glob('**/*.wav'))
    for i,audio_file_path in enumerate(audio_files): 
        convert_audio(input_file=audio_file_path,output_file=Path(DIR.results)/audio_file_path.name,target_sr=22050,target_channels=1)

    # 将英文转音素
    python_path = sys.executable
    filelists = str(ASR_results)
    text_index = 2
    command = f"{python_path} preprocess.py --text_index {text_index} --filelists {filelists}"
    subprocess.run(command, shell=True)
    print(f'****数据集保存已经保存在{DIR.results}目录下,可直接复制到Vits项目中的DUMMY2目录下****')

    #清空临时目录
    shutil.rmtree(DIR.root)

if __name__ == '__main__':
    main()