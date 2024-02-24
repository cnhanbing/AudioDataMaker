import os
import re
import torch
import torchaudio
import torchaudio.models
import yt_dlp
import os
import shutil
import requests
from collections import Counter

######################################################################################################################################################
#返回video_ids
class VideoIDExtractor:
    def __init__(self):
        pass

    def extract_video_id(self, youtube_url):
        # 匹配 YouTube 视频链接中的视频 ID
        pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)(?:&\S+)?'
        match = re.match(pattern, youtube_url)
        if match:
            return match.group(1)
        else:
            return None

    def process_file_list(self, input_file, output_file=None):
        video_ids = []
        # 打开输入文件
        with open(input_file, 'r') as f_input:
            # 逐行读取输入文件中的 YouTube 视频链接
            for line in f_input:
                youtube_url = line.strip()
                # 提取视频ID
                video_id = self.extract_video_id(youtube_url)
                # 如果视频ID存在，将其写入输出文件或列表
                if video_id:
                    if output_file:
                        # 写入输出文件
                        with open(output_file, 'a') as f_output:
                            f_output.write(video_id + '\n')
                    video_ids.append(video_id)

        if output_file:
            return None
        else:
            return video_ids
        
######################################################################################################################################################

def Down_Audio(video_id, output_folder):

    # 设置下载音频的参数
    audio_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_folder}/{video_id}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # 设置音频格式为MP3
        }],
    }
    # 构建视频 URL
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    # 使用yt-dlp下载音频
    with yt_dlp.YoutubeDL(audio_opts) as audio_ydl:
        audio_info = audio_ydl.extract_info(video_url, download=True)
        audio_file_path = audio_ydl.prepare_filename(audio_info)
        # 将文件路径中的扩展名改为.mp3
        audio_file_path = os.path.splitext(audio_file_path)[0] + '.mp3'

    return audio_file_path

######################################################################################################################################################
 
def convert_audio(input_file, output_file, target_sr=22050, target_channels=2):
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(input_file)

    # 如果目标通道数不等于输入通道数，则进行通道重采样
    if target_channels != waveform.shape[0]:
        # 如果目标通道数为1且输入通道数为2的情况，将两个通道合并为一个
        if target_channels == 1 and waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 如果目标通道数为2且输入通道数为1的情况，将输入通道重采样为2
        elif target_channels == 2 and waveform.shape[0] == 1:
            waveform = torch.cat([waveform, waveform], dim=0)

    # 转换采样率
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)

    # 保存为wav文件
    torchaudio.save(output_file, waveform, sample_rate=target_sr)

    print(f'音频转换已经完成，转换后的文件采样率：{target_sr},有{target_channels}个通道，请查看{output_file}')

    return waveform
######################################################################################################################################################

def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text

######################################################################################################################################################
#通过频道下的一个视频链接获取频道号
def get_channel_id(video_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'items' in data and len(data['items']) > 0:
        channel_id = data['items'][0]['snippet']['channelId']
        return channel_id
    else:
        return None
######################################################################################################################################################   
#通过频道号获取频道下所有视频的ID
def get_channel_videos(api_key, channel_id):
    url = 'https://www.googleapis.com/youtube/v3/search'
    
    # 搜索参数
    params = {
        'part': 'snippet',
        'channelId': channel_id,
        'key': api_key,
        'maxResults': 50,
    }

    # 发送 GET 请求
    response = requests.get(url, params=params)
    
    # 处理响应
    if response.status_code == 200:
        data = response.json()
        videos = [item['id']['videoId'] for item in data.get('items', []) if 'videoId' in item.get('id', {})]
        return videos
    else:
        print(f"Request failed with status code {response.status_code}")
        return []
######################################################################################################################################################   
def copy_most_common_speaker_wav(record_file, wav_directory, destination_directory):
    # 读取记录文件，统计每个说话人的行数
    speaker_counts = Counter()    
    with open(record_file, 'r') as file:
        for line in file:
            speaker = line.split('|')[0]
            speaker_counts[speaker] += 1
            
    # 找到说话人说的行数最多的说话人
    most_common_speaker = speaker_counts.most_common(1)[0][0]
    
    # 找到该说话人对应的所有 WAV 文件
    files_to_copy=[]
    with open(record_file, 'r') as file:
        for line in file:
            speaker, wavfile = line.strip().split('|')
            if speaker == most_common_speaker:           
                files_to_copy.append(wavfile)

    # 复制 WAV 文件到目标目录
    for filename in files_to_copy:
        source_filepath = os.path.join(wav_directory, filename)
        destination_filepath = os.path.join(destination_directory, filename)
        shutil.copy(source_filepath, destination_filepath)
  