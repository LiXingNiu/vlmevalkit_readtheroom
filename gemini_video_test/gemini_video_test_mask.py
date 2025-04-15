import os
import os.path as osp
import argparse
import pickle
import pandas as pd
import time
import logging
import re
from google.api_core import retry
import google.generativeai as genai
from google.generativeai import GenerativeModel

# 配置日志和API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key="xxxx")
GENERATION_CONFIG = {"temperature": 0.0}

TOM_PROMPT = """
Please carefully watch the following video and answer the question based on its content. Remember, you are very perceptive and excellent at reading between the lines, especially regarding the psychological states of the people in the video, including their Events, Beliefs, Intents, Emotions, and Desires.

Consider the following steps in your thinking process (you do not need to write these down):

Observe Events: Summarize the relevant events and actions happening in the video. If names are explicitly mentioned, you can refer to them by name; otherwise, use descriptive phrases.
Analyze Beliefs: Reflect on what the characters believe or think, including their understanding of events or their assumptions about others' mental states.
Speculate Intents: Consider the intentions of the characters—what they aim to do or achieve.
Perceive Emotions: Observe the emotions of the characters, expressed through facial expressions, tone, or behavior (e.g., happy, angry, sad, nervous, excited, disappointed).
Understand Desires: Think about what the characters like or dislike—their desires. 
    """

TMPL_SUB = """
These are masked blank frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question. If subtitles are provided, use them to determine the answer. If they are missing or empty, rely on the question and the options, drawing upon common sense to arrive at the best possible answer. \
Based on your understanding, respond with only the letter (A, B, C, D, or E) of the correct option.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-path", required=True)
    parser.add_argument("--output-xlsx", required=True)
    parser.add_argument("--use-subtitles", action="store_true")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--model", default="models/gemini-1.5-pro-latest",
                        help="Gemini model name")
    parser.add_argument("--ToM", action="store_true",
                        help="Add Theory of Mind prompt")
    parser.add_argument("--mask-vision", action="store_true",
                        help="Mask video frames and use subtitles only")
    return parser.parse_args()


def build_output_filename(args):
    """构建动态输出文件名"""
    base_name = osp.splitext(args.output_xlsx)[0]

    model_part = args.model.replace("/", "_").replace("models_", "")
    base_name += f"_{model_part}"

    if args.use_subtitles:
        base_name += "_sub"

    if args.ToM:
        base_name += "_ToM"

    if args.mask_vision:
        base_name += "_mask"

    return base_name + ".xlsx"


def init_output_file(output_path):
    """初始化输出文件"""
    if not osp.exists(output_path):
        pd.DataFrame(columns=[
            'index', 'video', 'video_path', 'question',
            'candidates', 'QID', 'answer', 'prediction', 'score'
        ]).to_excel(output_path, index=False)


def get_invalid_qids(output_path):
    """获取无效QID集合"""
    if not osp.exists(output_path):
        return set()

    df = pd.read_excel(output_path)
    invalid = set()
    for _, row in df.iterrows():
        pred = str(row['prediction']).strip().upper()
        if not re.fullmatch(r'^[A-E]$', pred) or pd.isna(row['prediction']):
            invalid.add(str(row['QID']))
    return invalid


def clean_invalid_records(output_path, invalid_qids):
    """清理无效记录"""
    if not osp.exists(output_path) or not invalid_qids:
        return

    df = pd.read_excel(output_path)
    cleaned_df = df[~df['QID'].astype(str).isin(invalid_qids)]
    cleaned_df.to_excel(output_path, index=False)


def get_pending_qids(df, output_path):
    """获取待处理QID集合"""
    if not osp.exists(output_path):
        return set(df['QID'].astype(str).tolist())

    processed_df = pd.read_excel(output_path)
    valid_qids = set(processed_df['QID'].astype(str).tolist())
    all_qids = set(df['QID'].astype(str).tolist())
    return all_qids - valid_qids


def append_result_to_excel(result, output_path):
    """追加单行结果到Excel"""
    df = pd.DataFrame([result])
    with pd.ExcelWriter(
            output_path,
            mode='a',
            engine='openpyxl',
            if_sheet_exists='overlay'
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            header=False,
            startrow=writer.sheets['Sheet1'].max_row
        )


def upload_and_wait(video_path):
    """上传视频并等待处理完成"""
    video_file = genai.upload_file(video_path)
    logger.info(f"Uploaded video: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        logger.info("Waiting for video processing...")
        time.sleep(5)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    return video_file


@retry.Retry(timeout=300)
def generate_response(video_file, prompt, model_name):
    """生成带重试的响应（有视频）"""
    try:
        model = GenerativeModel(model_name)
        response = model.generate_content(
            [prompt, video_file],
            generation_config=GENERATION_CONFIG,
            request_options={"timeout": 300}
        )
        return response.text
    finally:
        genai.delete_file(video_file.name)
        logger.info(f"Deleted video file: {video_file.uri}")


@retry.Retry(timeout=300)
def generate_text_response(prompt, model_name):
    """生成带重试的响应（纯文本）"""
    try:
        model = GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG,
            request_options={"timeout": 300}
        )
        return response.text
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        raise


def process_row(row, args):
    """处理单行数据"""
    # 处理字幕
    subtitles = ""
    if args.use_subtitles and row['subtitle_path']:
        subtitle_path = osp.join(args.data_root, row['subtitle_path'])
        if osp.exists(subtitle_path):
            with open(subtitle_path, 'rb') as f:
                subs = pickle.load(f)
                subtitles = '\n'.join(
                    f"{seg['text']}({seg['start']}s->{seg['end']}s)".replace('\\N', ' ')
                    for seg in subs["segments"] if seg['text'].strip()
                )

    # 构建prompt
    if args.mask_vision:
        prompt_tmpl = TMPL_SUB
    else:
        prompt_tmpl = FRAMES_TMPL_SUB if args.use_subtitles else FRAMES_TMPL_NOSUB

    base_prompt = prompt_tmpl.format(subtitles) + f"\nQuestion: {row['question']}\nOptions:\n" + "\n".join(
        eval(row['candidates']))

    # 添加ToM提示
    if args.ToM:
        full_prompt = TOM_PROMPT + "\n" + base_prompt
    else:
        full_prompt = base_prompt

    # 处理视频或纯文本
    if args.mask_vision:
        logger.info("Using masked vision mode (video omitted)")
        response = generate_text_response(full_prompt, args.model)
    else:
        video_path = osp.join(args.data_root, "video", f"{row['video']}.mp4")
        if not osp.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_file = upload_and_wait(video_path)
        response = generate_response(video_file, full_prompt, args.model)

    # 解析答案
    prediction = next((c for c in response if c in ['A', 'B', 'C', 'D', 'E']), '')

    # 计算分数
    answer = str(row['answer']).strip().upper()
    score = 1 if prediction.strip().upper() == answer else 0

    return {
        "index": row.name,
        "video": row['video'],
        "video_path": "MASKED" if args.mask_vision else osp.join("video", f"{row['video']}.mp4"),
        "question": row['question'],
        "candidates": row['candidates'],
        "QID": row['QID'],
        "answer": row['answer'],
        "prediction": prediction,
        "score": score
    }


def main():
    args = parse_args()
    final_output_path = build_output_filename(args)
    logger.info(f"Final output path: {final_output_path}")

    # 清理历史无效记录
    invalid_qids = get_invalid_qids(final_output_path)
    clean_invalid_records(final_output_path, invalid_qids)

    # 初始化输出文件
    if not osp.exists(final_output_path):
        pd.DataFrame(columns=[
            'index', 'video', 'video_path', 'question',
            'candidates', 'QID', 'answer', 'prediction', 'score'
        ]).to_excel(final_output_path, index=False)

    # 获取待处理QID
    df = pd.read_csv(args.tsv_path, sep='\t')
    pending_qids = get_pending_qids(df, final_output_path)
    logger.info(f"Total pending QIDs: {len(pending_qids)}")

    for _, row in df.iterrows():
        qid = str(row['QID'])
        if qid not in pending_qids:
            logger.info(f"Skipping valid QID: {qid}")
            continue

        try:
            result = process_row(row, args)
            logger.info(f"Processed QID {qid} - Score: {result['score']}")
        except Exception as e:
            logger.error(f"Failed QID {qid}: {str(e)}")
            result = {
                "index": row.name,
                "video": row['video'],
                "video_path": "MASKED" if args.mask_vision else osp.join("video", f"{row['video']}.mp4"),
                "question": row['question'],
                "candidates": row['candidates'],
                "QID": qid,
                "answer": row['answer'],
                "prediction": f"Error: {str(e)}",
                "score": 0
            }

        try:
            append_result_to_excel(result, final_output_path)
        except Exception as e:
            logger.error(f"Failed to save QID {qid}: {str(e)}")

    logger.info(f"Processing completed. Results saved to {final_output_path}")


FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Based on your understanding, respond with only the letter (A, B, C, D, or E) of the correct option.
"""

FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the video. \
Based on your understanding, respond with only the letter (A, B, C, D, or E) of the correct option.
"""

if __name__ == "__main__":
    main()