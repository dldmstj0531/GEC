# app.py
import re
import gradio as gr
from difflib import SequenceMatcher

# ── 모델 로딩/추론 자리 (원하는 코드로 교체) ─────────────────────
def load_model():
    return None
MODEL = load_model()

def model_predict(text: str) -> str:
    if not text.strip():
        return text
    rules = {
        "  +": " ",
        r"\s+([,.!?])": r"\1",
        r"\bi\b": "I",
        r"\bteh\b": "the",
        r"\brecieve\b": "receive",
        r"\boccured\b": "occurred",
    }
    out = text
    for pat, repl in rules.items():
        out = re.sub(pat, repl, out)
    return out
# ────────────────────────────────────────────────────────────────

TOKEN_PATTERN = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
def _tokenize_preserve_ws(s: str):
    return TOKEN_PATTERN.findall(s)

def highlight_differences(src: str, tgt: str) -> str:
    s_tok = _tokenize_preserve_ws(src)
    t_tok = _tokenize_preserve_ws(tgt)
    sm = SequenceMatcher(a=s_tok, b=t_tok)
    src_out, tgt_out = [], []
    join = lambda toks: "".join(toks)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            src_out.append(join(s_tok[i1:i2]))
            tgt_out.append(join(t_tok[j1:j2]))
        elif op == "replace":
            src_out.append(f"<mark class='del'>{join(s_tok[i1:i2])}</mark>")
            tgt_out.append(f"<mark class='ins'>{join(t_tok[j1:j2])}</mark>")
        elif op == "delete":
            src_out.append(f"<mark class='del'>{join(s_tok[i1:i2])}</mark>")
        elif op == "insert":
            tgt_out.append(f"<mark class='ins'>{join(t_tok[j1:j2])}</mark>")

    return f"""
    <div class="diff-grid">
      <div class="diff-box">
        <div class="diff-title">입력 문장</div>
        <div class="diff-text">{''.join(src_out)}</div>
      </div>
      <div class="diff-box">
        <div class="diff-title">교정된 문장</div>
        <div class="diff-text">{''.join(tgt_out)}</div>
      </div>
    </div>
    """

def correct_and_diff(user_text: str):
    corrected = model_predict(user_text or "")
    diff_html = highlight_differences(user_text or "", corrected or "")
    return corrected, diff_html

custom_css = """
.diff-grid { display: grid; gap: 12px; grid-template-columns: 1fr 1fr; }
.diff-box { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; background: white; }
.diff-title { font-weight: 600; margin-bottom: 8px; color: #111827; }
.diff-text { line-height: 1.9; font-size: 16px; color: #111827; word-break: break-word; white-space: pre-wrap; }
mark.ins { background: #d1fae5; padding: 2px 3px; border-radius: 6px; }
mark.del { background: #fee2e2; padding: 2px 3px; border-radius: 6px; text-decoration: line-through; }
"""

with gr.Blocks(css=custom_css, title="오타/문법 교정 데모") as demo:
    gr.Markdown("# ✨ 오타/문법 교정 데모\n문장을 입력하고 **교정하기**를 누르면, 복사용 교정문과 **차이 하이라이트**가 표시됩니다.")
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(label="입력 문장", placeholder="예) i recieve teh package yesterday ,  it was very good  .", lines=4)
            with gr.Row():
                btn = gr.Button("교정하기", variant="primary")
                clear = gr.Button("초기화")
        with gr.Column(scale=3):
            out_tgt = gr.Textbox(label="(복사용) 교정된 문장", lines=3)
            diff_html = gr.HTML(label="차이 하이라이트")

    btn.click(fn=correct_and_diff, inputs=inp, outputs=[out_tgt, diff_html])
    clear.click(lambda: ("", ""), None, [out_tgt, diff_html]).then(lambda: "", None, inp)

if __name__ == "__main__":
    demo.launch()

