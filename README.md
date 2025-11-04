# ✏️ 오타 수정 by “오탁수정”

## 🎬 Demo

[🔗 데모 영상 보기](https://github.com/user-attachments/assets/f7994a93-9fe0-4fba-9020-d531c2f827f6)

---

## 🧠 GEC: Grammar and Spelling Error Correction

**영어 문장의 문법성 오류 및 오타를 자동으로 교정하는 GEC (Grammatical Error Correction)** 프로젝트입니다.  
**RoBERTa 기반 GEC 모델을 파인튜닝**하여, 실제 학습된 모델이 교정한 결과를 **Gradio 웹 인터페이스로 시각화**합니다.

---

## 🚀 주요 기능

### ✅ 학습된 GEC 모델 직접 연동

- `roberta_gector_k5000_noCE_1900K` 폴더 내 학습된 모델(`model.safetensors`)을 불러와 실제 추론 수행

### ✅ 문법 및 오타 자동 교정

- 문장 내 오류(시제, 단수/복수, 철자, 어순 등)를 인식해 자연스럽게 교정

### ✅ 시각적 하이라이트

| 상태 | 색상   | 의미      |
| ---- | ------ | --------- |
| 🟥   | 빨간색 | 오류 단어 |
| 🟩   | 초록색 | 교정 단어 |

### ✅ 로그 저장

- 모든 교정 결과(입력문장, 교정문, 예측 태그)를 `correction_log.csv`로 자동 저장

### ✅ 다운로드 기능

- 웹 UI에서 교정 결과를 CSV 파일로 즉시 다운로드 가능

---

## ⚙️ 실행 방법

### 1️⃣ 패키지 설치

아래 명령어 한 줄로 실행 환경을 자동 세팅할 수 있습니다.

```bash
pip install -r requirements.txt
```

requirements.txt에는 torch, transformers, gradio, numpy 등
실행에 필요한 주요 라이브러리와 버전이 포함되어 있습니다.

### 2️⃣ 앱 실행

```bash
python app3.py
```

실행 후 브라우저에서 자동으로 열립니다: http://127.0.0.1:7860

### 3️⃣ 사용 방법

1. 문장을 입력
2. [교정하기] 버튼 클릭
3. 아래에 빨강/초록 하이라이트로 수정 차이 표시
4. 교정 결과를 CSV로 다운로드 가능

---

## 📘 모델 정보

| 항목                   | 내용                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| **Base Model**         | RoBERTa                                                                                  |
| **Fine-tuned Task**    | Grammar & Spelling Error Correction                                                      |
| **Training Dataset**   | 약 1.9M 문장 (synthetic + BEA 기반)                                                      |
| **Training Objective** | Token-level edit-tag classification (`KEEP`, `REPLACE`, `DELETE`, `ADD`, `TRANSFORM` 등) |

---

## 📚 Reference & Dataset

- [C4_200M Synthetic GEC Dataset](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction)
- [BEA-2019 Shared Task (W&I + LOCNESS)](https://www.cl.cam.ac.uk/research/nl/bea2019st/)

---

## 🧑‍💻 Author

| 이름                     | 역할                            | 연락처                                                   |
| ------------------------ | ------------------------------- | -------------------------------------------------------- |
| **이서율 (SeoYool Lee)** | AI / NLP Research & Development | 📧 [dldmstj0531@gmail.com](mailto:dldmstj0531@gmail.com) |
| **오정탁 (Jungtak Oh)**  | AI / NLP Research & Development | 📧 [jungtak99@gmail.com](mailto:jungtak99@gmail.com)     |
| **김아람 (Ahram Kim)**   | AI / NLP Research & Development | 📧 [ahram0223@naver.com](mailto:ahram0223@naver.com)     |
| **황호성 (Hosung Hwang)**| AI / NLP Research & Development | 📧 [hhs6228@gmail.com](mailto:hhs6228@gmail.com)         |
| **박재영 (Jaeyoung Park)**| AI / NLP Research & Development | 📧 [kingjea0624@gmail.com](mailto:kingjea0624@gmail.com) |

> Team "오탁수정" (LikeLion Team 2)
