import os
import shutil
import nemo.collections.asr as nemo_asr

def speaker_verification(source_dir, result_file):
    # 实例化说话者验证模型
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

    # 获取源目录下所有文件列表
    files = sorted(os.listdir(source_dir))
    if not files:
        print("Source directory is empty.")
        return

    # 如果结果文件已存在，则清空内容
    with open(result_file, "w") as f:
        f.write("")

    # 用于存放已经确认的说话者及其对应的编号
    speaker_dict = {}

    # 遍历源目录中的所有文件
    for filename in files:
        if filename.endswith(".wav"):
            audio_path = os.path.join(source_dir, filename)
            speaker_id = None
            # 逐个与之前处理过的文件进行比对
            for prev_speaker_id, prev_filename in speaker_dict.items():
                reference_audio_path = os.path.join(source_dir, prev_filename)
                # 使用说话者验证模型验证说话者
                is_same_speaker = speaker_model.verify_speakers(reference_audio_path, audio_path)
                if is_same_speaker:
                    speaker_id = prev_speaker_id
                    break
            if speaker_id is None:
                # 如果与之前的文件都不同，则创建新的编号
                speaker_id = f"{len(speaker_dict) + 1:03d}"
                speaker_dict[speaker_id] = filename
            # 将当前文件及其对应的说话者编号写入结果文件
            with open(result_file, "a") as f:
                f.write(f"speaker{speaker_id}|{filename}\n")
    #说话人验证

def Speaker_Verification(source_dir, destination_dir, reference_audio_path):
    # 实例化说话者验证模型
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(source_dir, filename)
            # 使用说话者验证模型验证说话者
            is_same_speaker = speaker_model.verify_speakers(reference_audio_path, audio_path)
            if is_same_speaker:
                # 如果是同一说话者，则复制文件到目标目录
                shutil.copy(audio_path, os.path.join(destination_dir, filename))




