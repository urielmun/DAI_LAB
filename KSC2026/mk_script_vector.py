# make_video_scripts.py
"""
허깅페이스 모델을 활용해서 로컬에 대형 모델을 로드하고, 입력 데이터셋의 각 장면 정보를 읽어 
텍스트-투-비디오(Text-to-Video)용 프롬프트를 자동으로 생성하는 파이프라인
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-14B"


def build_video_script_llm(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    def generate(messages, max_new_tokens=280):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

        return tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    return generate


def build_video_script_prompt(row):
    scene_text = row.get("scene_text", "")
    characters = row.get("characters", [])
    mood = row.get("mood", [])
    important_nodes = row.get("important_nodes", [])

    system_prompt = """
You are a film director writing prompts for a silent text-to-video model.

You must write in English only.
Never use Chinese, Korean, Japanese, or any non-English language.
Do not include dialogue, quoted speech, subtitles, captions, readable text, or voice-over.
Use the source dialogue only to infer facial expression, body language, emotional tension, and staging.

Return valid JSON only.
The JSON must contain exactly one key: "prompt".
The value must be a single English paragraph.
"""

    user_prompt = f"""
Create a silent cinematic video prompt.

Scene:
{scene_text}

Characters:
{", ".join(characters)}

Return JSON only:
{{"prompt":"..."}}

Rules:
- English only.
- No dialogue or readable text.
- Use a consistent dark Victorian detective film style in every scene.
- Muted colors, cold lighting, realistic cinematography.
- Write exactly 3 sentences describing action, camera, and final composition.
"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def extract_json_object(text):
    """
    모델이 앞뒤에 불필요한 텍스트를 붙였을 때 JSON만 추출.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {
            "prompt": text
        }

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "prompt": text
        }


def normalize_video_prompt(obj):
    if not isinstance(obj, dict):
        prompt = str(obj)
    else:
        prompt = obj.get("prompt", "")

    if isinstance(prompt, dict):
        prompt = prompt.get("prompt", "")

    if isinstance(prompt, list):
        prompt = " ".join(str(x) for x in prompt)

    prompt = str(prompt).strip()

    # prompt 안에 또 JSON 문자열이 들어간 경우 복구
    if prompt.startswith("{") and '"prompt"' in prompt:
        try:
            nested = json.loads(prompt)
            prompt = nested.get("prompt", prompt)
        except Exception:
            pass

    # 과도한 길이 제한
    words = prompt.split()
    if len(words) > 120:
        prompt = " ".join(words[:120])

    return {
        "prompt": prompt
    }


def generate_video_scripts_jsonl(
    input_jsonl_path,
    output_jsonl_path,
    model_name=MODEL_NAME,
    max_new_tokens=280,
):
    llm = build_video_script_llm(model_name)

    count = 0

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl_path, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            messages = build_video_script_prompt(row)

            raw_output = llm(
                messages,
                max_new_tokens=max_new_tokens,
            )

            parsed = extract_json_object(raw_output)
            video_script = normalize_video_prompt(parsed)

            output_row = {
                **row,
                "video_script": video_script,
                "video_script_raw_output": raw_output,
            }

            fout.write(
                json.dumps(output_row, ensure_ascii=False) + "\n"
            )

            count += 1

            print(
                f"[{line_idx}] "
                f"scene_id={row.get('scene_id')} "
                f"prompt_len={len(video_script['prompt'])}"
            )

    print(f"saved {count} rows -> {output_jsonl_path}")
    

generate_video_scripts_jsonl(
    input_jsonl_path="/home/urie111/rag/data/SignOfFour_t2v_input.jsonl",
    output_jsonl_path="/home/urie111/rag/data/SignOfFour_video_scripts.jsonl",
    model_name="Qwen/Qwen3-14B",
    max_new_tokens=180,
) 