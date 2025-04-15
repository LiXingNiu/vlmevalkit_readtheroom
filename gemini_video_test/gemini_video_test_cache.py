import os
import os.path as osp
import argparse
import pickle
import pandas as pd
import time
import logging
import re
import datetime
from google.api_core import retry
import google.generativeai as genai
from google.generativeai import GenerativeModel, caching

# 配置日志和API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key="xxx")
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
    return base_name + ".xlsx"


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
        time.sleep(2)  # 等待2秒后查询一次状态
        video_file = genai.get_file(video_file.name)
    if video_file.state.name != "ACTIVE":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    return video_file


@retry.Retry(timeout=300)
def generate_response(video_file, prompt, model_name, delete_after=True):
    """
    根据视频文件生成回复，若视频较大则采用缓存优化；
    delete_after 参数为 True 时，在生成回复后删除上传的视频文件，
    若为 False，则不删除（已上传的视频会被复用，待统一清理）。
    """
    try:
        # 通过计算 token 数量判断是否使用缓存
        token_result = GenerativeModel(model_name).count_tokens([video_file])
        video_token_count = token_result.total_tokens
        min_cache_token_num = 32768  # 缓存阈值
        use_cache = video_token_count >= min_cache_token_num

        if use_cache:
            cache = caching.CachedContent.create(
                model=model_name,
                display_name=video_file.name,
                contents=[video_file],
                ttl=datetime.timedelta(minutes=5)
            )
            model = genai.GenerativeModel.from_cached_content(cached_content=cache)
            response = model.generate_content(
                [prompt],
                generation_config=GENERATION_CONFIG,
                request_options={"timeout": 300}
            )
            cache.delete()
            logger.info("Cache is deleted.")
        else:
            model = GenerativeModel(model_name=model_name, generation_config=GENERATION_CONFIG)
            response = model.generate_content(
                [prompt, video_file],
                generation_config=GENERATION_CONFIG,
                request_options={"timeout": 300}
            )
        return response.text
    finally:
        if delete_after:
            genai.delete_file(video_file.name)
            logger.info(f"Deleted video file: {video_file.uri}")


def process_row(row, args, uploaded_videos):
    """
    处理单行数据，同时判断是否复用已经上传的视频文件。
    如果检查到上传的视频状态不为 ACTIVE，则给出提示，并重新上传视频。
    """
    video_id = row['video']
    video_path = osp.join(args.data_root, "video", f"{video_id}.mp4")
    if not osp.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 检查之前是否上传过同一个视频
    if video_id in uploaded_videos:
        video_file_candidate = uploaded_videos[video_id]
        current_file = genai.get_file(video_file_candidate.name)
        if current_file.state.name != "ACTIVE":
            logger.warning(
                f"Previously uploaded video for video id '{video_id}' is not active "
                f"(current state: {current_file.state.name}). Re-uploading the video."
            )
            video_file = upload_and_wait(video_path)
            uploaded_videos[video_id] = video_file
        else:
            video_file = current_file
            logger.info(f"Reusing uploaded video for video id: {video_id}")
    else:
        video_file = upload_and_wait(video_path)
        uploaded_videos[video_id] = video_file

    # 处理字幕（如果需要）
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

    prompt_tmpl = FRAMES_TMPL_SUB if args.use_subtitles else FRAMES_TMPL_NOSUB
    base_prompt = f"{prompt_tmpl.format(subtitles)}\nQuestion: {row['question']}\nOptions:\n" + "\n".join(
        eval(row['candidates'])
    )
    if args.ToM:
        full_prompt = TOM_PROMPT + "\n" + base_prompt
    else:
        full_prompt = base_prompt

    # 使用已上传视频生成回复，此处设置 delete_after=False
    response_text = generate_response(video_file, full_prompt, args.model, delete_after=False)

    prediction = next((c for c in response_text if c in ['A', 'B', 'C', 'D', 'E']), '')
    answer = str(row['answer']).strip().upper()
    score = 1 if prediction.strip().upper() == answer else 0

    return {
        "index": row.name,
        "video": video_id,
        "video_path": osp.join("video", f"{video_id}.mp4"),
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

    # 初始化输出文件（如果不存在）
    if not osp.exists(final_output_path):
        pd.DataFrame(columns=[
            'index', 'video', 'video_path', 'question',
            'candidates', 'QID', 'answer', 'prediction', 'score'
        ]).to_excel(final_output_path, index=False)

    # 获取待处理QID
    df = pd.read_csv(args.tsv_path, sep='\t')
    pending_qids = get_pending_qids(df, final_output_path)
    logger.info(f"Total pending QIDs: {len(pending_qids)}")

    # 用于复用上传视频的缓存字典，key 为 video id，value 为视频文件对象
    uploaded_videos = {}

    for _, row in df.iterrows():
        qid = str(row['QID'])
        if qid not in pending_qids:
            logger.info(f"Skipping valid QID: {qid}")
            continue

        try:
            result = process_row(row, args, uploaded_videos)
            logger.info(f"Processed QID {qid} - Score: {result['score']}")
        except Exception as e:
            logger.error(f"Failed QID {qid}: {str(e)}")
            result = {
                "index": row.name,
                "video": row['video'],
                "video_path": osp.join("video", f"{row['video']}.mp4"),
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

    # 处理完成后统一清理所有复用过的上传视频
    for video_id, video_file in uploaded_videos.items():
        try:
            genai.delete_file(video_file.name)
            logger.info(f"Deleted reused video file: {video_file.uri} for video id: {video_id}")
        except Exception as e:
            logger.error(f"Failed to delete video file for video id {video_id}: {str(e)}")

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