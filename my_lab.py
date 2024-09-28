from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()
from autoprocess import open_slice,open_asr,open1abc,open1Bb
from audio_emotion_analyse import get_all_emotion,draw_emotion,save_list
import librosa
import os
import soundfile as sf
import numpy as np
import torch
import traceback

def slice_wav(train_file_name):
    slice_inp_path = "resources/train/" + train_file_name
    slice_opt_root = "resources/slice/" + train_file_name
    threshold = -34
    min_length = 4000
    min_interval = 300
    hop_size = 10
    max_sil_kept = 500
    _max = 0.9
    alpha = 0.25
    n_process = 4
    slice_generator = open_slice(slice_inp_path, slice_opt_root, threshold, min_length, min_interval, hop_size,max_sil_kept, _max, alpha, n_process)
    for message, visible_update_1, visible_update_2 ,visible_update_3,visible_update_4,visible_update_5in in slice_generator:
        print(message)
    print(">>>>>>切割结束\n")

def asr_slice(train_file_name):
    asr_inp_dir = "resources/slice/" + train_file_name
    asr_opt_dir = "resources/asr/" + train_file_name
    asr_model = "达摩 ASR (中文)"
    asr_size = "large"
    asr_lang = "zh"
    asr_precision = "float32"

    asr_generator = open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision)
    for message, visible_update_1, visible_update_2, visible_update_3, visible_update_4, visible_update_5 in asr_generator:
        print(message)
    print(">>>>>>asr结束\n")

def get_bert_semantic(train_file_name,gpu):
    inp_text = "resources/asr/" + train_file_name + "/" + train_file_name + ".list"
    inp_wav_dir = "resources/slice/" + train_file_name
    exp_name = "exp_" + train_file_name
    gpu_numbers1a = gpu
    gpu_numbers1Ba = gpu
    gpu_numbers1c = gpu
    bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    cnhubert_base_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    pretrained_s2G = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"

    open1abc_generator = open1abc(inp_text, inp_wav_dir, exp_name, gpu_numbers1a, gpu_numbers1Ba, gpu_numbers1c,
                                  bert_pretrained_dir, cnhubert_base_dir, pretrained_s2G)
    for message, visible_update_1, visible_update_2 in open1abc_generator:
        print(message)

    print(">>>>>>数据预处理一件三连结束\n")

def slice_for_emotion(audio_name,window,hop):  #窗口切片音频
    # 加载音频文件
    audio, sr = librosa.load(audio_name)

    # 设置切片时长（以秒为单位）
    window_length = window * sr  # 窗口长度
    hop_length = hop * sr  # 移动步幅

    # 计算补全长度（半个窗口长度）
    pad_length = window_length // 2

    # 在音频的两端进行补全
    audio = np.pad(audio, (pad_length, pad_length), mode='constant')  #将两边补全，这样不论窗口大小输出向量数都相同

    # 获得文件去除后缀的名字
    filename = audio_name
    new_filename = os.path.basename(filename)
    new_filename = os.path.splitext(new_filename)[0]

    # 创建保存切片的文件夹
    path = f"resources/emo_slice/{new_filename}/{window}"
    if not os.path.exists(path):
        os.makedirs(path)

    for start in range(0, len(audio) - window_length + 1, hop_length):
        slice_audio = audio[start:start + window_length]
        sf.write(f"resources/emo_slice/{new_filename}/{window}/slice_{start // hop_length}.wav", slice_audio, sr)

    print(f">>>>文件({audio_name})切片成功,保存路径({path})")
    return path


def weighted_sum(vectors, weights):
    # 确保权值是一个一维数组
    weights = np.array(weights)

    # 计算加权求和
    result = np.tensordot(weights, vectors, axes=(0, 0))
    result=np.round(result,decimals=2)
    return result

def get_average_emo(filename):
    emo=[]
    for window in range(4, 20, 4):
        emo_slice_path = slice_for_emotion(f"resources/train/{filename}", window, 1)
        labels, audio_emotion = get_all_emotion(emo_slice_path)  # 获取每个切片的情感
        save_list(labels, f"resources/emotion_data/{filename}/labels_{window}.txt")
        save_list(audio_emotion, f"resources/emotion_data/{filename}/audio_emotion_{window}.txt")
        draw_emotion(audio_emotion, labels, filename,window)  # 绘制情感图-
        emo.append(audio_emotion)
    weights = [0.35, 0.3, 0.2, 0.15]
    result = weighted_sum(emo, weights)
    save_list(result, f"resources/emotion_data/{filename}/audio_emotion_average.txt")
    draw_emotion(result, labels, filename, "average")  # 绘制情感图
    return result

def read_pt(path):
    file=torch.load(path)
    print(file)
    print(file.shape)
    return file

def get_audio_durations(folder_path):
    durations = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):  # 可以根据需要添加其他音频格式
            file_path = os.path.join(folder_path, filename)
            duration = librosa.get_duration(filename=file_path)  # 获取音频持续时间
            durations[filename] = duration
    return durations


def read_intervals_from_txt(file_path):  #读取每个slice的音频范围，返回一个元组(start,end)，以便寻找对应的感情区域
    intervals = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除空白字符和换行符
            line = line.strip()
            if line:  # 确保行不是空的
                # 分割字符串并转换为浮点数
                start, end = map(float, line.split('/'))
                intervals.append((start, end))  # 将元组添加到列表中

    return intervals

def name2go(wav_name, lines,semantic):
    lines.append("%s\t%s" % (wav_name, semantic))



def interval_to_emo(intervals, average_emo, filename):
    # config = {}
    # gpu_numbers = "0"
    # gpu_names = gpu_numbers.split("-")
    # all_parts = len(gpu_names)
    # for i_part in range(all_parts):
    #     config.update(
    #         {
    #             "i_part": str(i_part),
    #             "all_parts": str(all_parts),
    #         }
    #     )
    # os.environ.update(config)
    # semantic_path = f"logs/exp_{filename}/emo_semantic.tsv"

    all_indices = []
    number_of_vectors = len(average_emo)

    # 计算每个向量对应的时间长度（假设向量均匀分布在音频上）
    duration_per_vector = 1

    # 为每个向量分配时间戳（这里使用向量时间区间的中心点）
    vector_times = [i * duration_per_vector for i in range(number_of_vectors)]

    # 为每个区间找到对应的向量索引
    interval_vectors = []

    for interval in intervals:
        start_time, end_time = interval
        # 找到所有时间戳在当前区间内的向量索引
        indices = [i for i, t in enumerate(vector_times) if t >= start_time and t < end_time]
        interval_vectors.append(indices)

    # 输出结果
    for idx, indices in enumerate(interval_vectors):
        corresponding_vectors = [average_emo[i] for i in indices]
        all_indices.append(corresponding_vectors)  # 转换为tensor并保存

    # semantic = " ".join([str(i) for i in all_indices])
    # with open(inp_text, "r", encoding="utf8") as f:
    #     lines = f.read().strip("\n").split("\n")
    #
    # lines1 = []
    # for line in lines[int(i_part):: int(all_parts)]:
    #         # print(line)
    #     try:
    #         # wav_name,text=line.split("\t")
    #         wav_name, spk_name, language, text = line.split("|")
    #         wav_name = os.path.basename(wav_name)
    #         # name2go(name,lines1)
    #         name2go(wav_name, lines1,semantic)
    #     except:
    #         print(line, traceback.format_exc())
    # with open(semantic_path, "w", encoding="utf8") as f:
    #     f.write("\n".join(lines1))

    # 保存为文本文件
    save_list(all_indices, f"logs/exp_{filename}/all_vector.txt")

    return all_indices

def train_valle(train_file_name,gpu):
    batch_size1Bb = 4
    total_epoch1Bb = 15
    exp_name = "exp_" + train_file_name
    if_dpo = False
    if_save_latest1Bb = True
    if_save_every_weights1Bb = True
    save_every_epoch1Bb = 5
    gpu_numbers1Bb = gpu
    pretrained_s1 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    open1Bb_generator = open1Bb(batch_size1Bb, total_epoch1Bb, exp_name, if_dpo, if_save_latest1Bb,if_save_every_weights1Bb, save_every_epoch1Bb, gpu_numbers1Bb, pretrained_s1)
    for message, visible_update_1, visible_update_2 in open1Bb_generator:
        print(message)
    print("Step5:GPT训练结束\n")

if __name__=="__main__":
    train_filename="shoulinrui.m4a"
    inp_text = "resources/asr/" + train_filename + "/" + train_filename + ".list"
    gpu = "0"  # 多个要-，单个就打数字
    slice_wav(train_filename)  #切割音频（原GPT切割）
    asr_slice(train_filename)  #切割音频asr识别
    get_bert_semantic(train_filename,gpu)  #获得文本自特征、音频自特征和sovits预训练模型反向推出来的semantic
    average_emo=get_average_emo(train_filename)  #获得平均情感向量 (93,9) 93和步幅、音频长度有关，步幅为1s则就这里就表示93s每一秒一个取样，9为9种类型情感

    slice_log_path = "resources/slice/shoulinrui.m4a/slice_log.txt"
    intervals = read_intervals_from_txt(slice_log_path)
    all_vector=interval_to_emo(intervals,average_emo,train_filename)  #得到每段切分音频对应的情感向量[第n段音频，时间段内包含的所有情感向量，9种情感]

