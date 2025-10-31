import re, json, torch, csv, os
import gradio as gr
import numpy as np
from typing import List, Dict, Any
from transformers import RobertaTokenizerFast, AutoModelForTokenClassification
import sys
sys.path.append("C:/Users/dldms/GEC/roberta_gector_k5000_noCE_1900K/")
from gector_utils import GectorVocab

# ==============================================
# 0. 환경 설정
# ==============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "C:/Users/dldms/GEC/roberta_gector_k5000_noCE_1900K/"

print("🚀 Loading model and tokenizer...")
# ✅ roberta-base fast tokenizer 기반
infer_tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base",
    add_prefix_space=True
)
infer_tokenizer._tokenizer.model = infer_tokenizer.backend_tokenizer.model.__class__.from_file(
    f"{OUTPUT_DIR}/vocab.json",
    f"{OUTPUT_DIR}/merges.txt"
)

infer_model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR).to(device).eval()

with open(f"{OUTPUT_DIR}/LABEL2ID.json", "r", encoding="utf-8") as f:
    LABEL2ID = json.load(f)
with open(f"{OUTPUT_DIR}/ID2LABEL.json", "r", encoding="utf-8") as f:
    ID2LABEL = {int(k): v for k, v in json.load(f).items()}
voc = GectorVocab.from_files(OUTPUT_DIR, verb_file=f"{OUTPUT_DIR}/verb-form-vocab.txt")
KEEP_ID = LABEL2ID.get("$KEEP", 0)


# ==============================================
# 1. 추론 함수
# ==============================================
@torch.inference_mode()
def predict_actions_for_tokens(tokens: List[str]) -> List[str]:
    enc = infer_tokenizer(
        [tokens],
        is_split_into_words=True,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True
    ).to(device)

    outputs = infer_model(**enc)
    preds = outputs.logits.argmax(-1).squeeze(0).detach().cpu().numpy()
    word_ids = enc.word_ids(batch_index=0)

    actions_by_word = []
    seen_words = set()
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in seen_words:
            seen_words.add(wid)
            lab_id = int(preds[idx])
            act = ID2LABEL.get(lab_id, "$KEEP")
            actions_by_word.append(act)

    if len(actions_by_word) < len(tokens):
        actions_by_word += ["$KEEP"] * (len(tokens) - len(actions_by_word))
    elif len(actions_by_word) > len(tokens):
        actions_by_word = actions_by_word[:len(tokens)]

    return actions_by_word


def apply_and_detokenize(tokens: List[str], actions: List[str]) -> Dict[str, Any]:
    edited_tokens = voc.apply_actions(tokens, actions)
    text = " ".join(edited_tokens)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[\{“\"'])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]\}”\"'])", r"\1", text)
    return {"edited_tokens": edited_tokens, "edited_text": text}


# ==============================================
# 2. 하이라이트 + 비교 표시
# ==============================================
def compare_inference(sentence: str):
    if not sentence.strip():
        return "", "", "", "", None

    tokens = sentence.strip().split()
    actions = predict_actions_for_tokens(tokens)
    result = apply_and_detokenize(tokens, actions)
    edited_text = result["edited_text"]

    # === 색상 정의 ===
    color_wrong = "#ffb3b3"   # 빨강 (오류)
    color_fixed = "#b3ffb3"   # 초록 (교정된 단어)

    input_highlight = ""
    output_highlight = ""

    # === 입력 문장 (오류만 빨강 표시) ===
    for tok, act in zip(tokens, actions):
        if act == "$KEEP":
            input_highlight += f"{tok} "
        else:
            input_highlight += f"<span style='background-color:{color_wrong}; padding:2px 4px; border-radius:3px;'>{tok}</span> "

    # === 교정 문장 (교정된 단어만 초록 표시) ===
    for tok, act in zip(result["edited_tokens"], actions):
        if act == "$KEEP":
            output_highlight += f"{tok} "
        else:
            output_highlight += f"<span style='background-color:{color_fixed}; padding:2px 4px; border-radius:3px;'>{tok}</span> "

    # === 로그 저장 ===
    log_path = "correction_log.csv"
    log_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["Input", "Corrected", "Actions"])
        writer.writerow([sentence, edited_text, " ".join(actions)])

    return sentence, edited_text, input_highlight.strip(), output_highlight.strip(), log_path


# ==============================================
# 3. Gradio UI 구성
# ==============================================
with gr.Blocks(title="오타/문법 교정 데모") as demo:
    gr.Markdown("## ✨ 오타/문법 교정 데모\n문장을 입력하고 **교정하기**를 누르면, 수정된 문장과 차이를 하이라이트로 표시합니다.")
    gr.Markdown(
        "🟥 **빨강:** 오류로 판단된 단어  🟩 **초록:** 교정된 단어"
    )

    with gr.Row():
        input_box = gr.Textbox(label="입력 문장", placeholder="문장을 입력하세요...", lines=3)
        output_box = gr.Textbox(label="(복사용) 교정된 문장", interactive=False, lines=3)

    with gr.Row():
        input_highlight = gr.HTML(label="입력 문장 (오류 표시)")
        output_highlight = gr.HTML(label="교정된 문장 (교정 표시)")

    with gr.Row():
        run_btn = gr.Button("교정하기", variant="primary")
        clear_btn = gr.Button("초기화")
        download_btn = gr.File(label="결과 CSV 다운로드", interactive=False)

    run_btn.click(
        fn=compare_inference,
        inputs=input_box,
        outputs=[input_box, output_box, input_highlight, output_highlight, download_btn]
    )
    clear_btn.click(lambda: ("", "", "", "", None), None, [input_box, output_box, input_highlight, output_highlight, download_btn])

demo.launch()
