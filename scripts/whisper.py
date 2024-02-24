import os
import glob
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def Whisper_ASR(input_directory, output_file, error_log_file,id,idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # 加载音频文件数据集
    def load_audio_dataset(directory):
        audio_files = glob.glob(os.path.join(directory, '*.wav'))
        return [{"audio_file": audio_file} for audio_file in audio_files]

    audio_dataset = load_audio_dataset(input_directory)

    # 遍历音频文件并进行识别
    with open(output_file, 'a') as result_file, open(error_log_file, 'a') as error_log:
        for i, data in enumerate(audio_dataset):
            audio_file = data["audio_file"]
            try:
                # 使用 Whisper 模型进行识别
                result = pipe(audio_file)
                print(
                    f"{i+1:05}/{len(audio_dataset)}:\t{audio_file}:\t{result['text']}")
                
                basefilename=os.path.basename(audio_file)

                # 将识别结果写入输出文件
                result_file.write(f"DUMMY2/{id}/{basefilename}|{idx}|{result['text']}\n")
            except Exception as e:
                # 记录出错的文件到错误日志
                error_log.write(f"Error processing file: {audio_file}\n")
                error_log.write(f"Error message: {str(e)}\n")

    print("识别完成，结果已写入到输出文件。")
