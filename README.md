🧠 GEC: Grammar and Spelling Error Correction Demo

영어 문장의 문법성 오류 및 오타를 자동으로 교정하는 GEC(Grammatical Error Correction) 프로젝트입니다.
RoBERTa 기반 GEC 모델을 파인튜닝하여, 실제로 학습된 모델이 교정한 결과를 Gradio 웹 인터페이스로 시각화합니다.

🚀 주요 기능

✅ 학습된 GEC 모델 직접 연동

roberta_gector_k5000_noCE_1900K 폴더 내 학습된 모델(model.safetensors)을 불러와 실제 추론 수행

✅ 문법 및 오타 자동 교정

문장 내 오류(시제, 단수/복수, 철자, 어순 등)를 인식해 자연스럽게 교정

✅ 시각적 하이라이트

🟥 오류 단어 → 빨간색

🟩 교정 단어 → 초록색

✅ 로그 저장

모든 교정 결과(입력문장, 교정문, 예측 태그)를 correction_log.csv로 자동 저장

✅ 다운로드 기능

웹 UI에서 교정 결과를 CSV 파일로 바로 다운로드 가능

🧩 프로젝트 구조
GEC/
│
├── app3.py                             # Gradio 데모 실행 스크립트
├── requirements.txt                    # 실행 환경 재현용 패키지 목록
│
├── roberta_gector_k5000_noCE_1900K/    # 학습된 GEC 모델 및 토크나이저
│   ├── model.safetensors
│   ├── vocab.json
│   ├── merges.txt
│   ├── LABEL2ID.json
│   ├── ID2LABEL.json
│   ├── gector_utils.py
│   └── verb-form-vocab.txt
│
├── correction_log.csv                  # (실행 후 자동 생성) 교정 로그
└── README.md                           # 프로젝트 설명 파일

⚙️ 실행 방법
1️⃣ 패키지 설치

아래 명령어 한 줄이면, 동일한 환경을 자동으로 세팅할 수 있습니다.

pip install -r requirements.txt


🔹 requirements.txt에는 torch, transformers, gradio, numpy 등
실행에 필요한 주요 라이브러리와 버전이 포함되어 있습니다.

2️⃣ 앱 실행
python app3.py


실행 후 브라우저에서 자동으로 열립니다:

http://127.0.0.1:7860

3️⃣ 사용 방법

문장을 입력

[교정하기] 버튼 클릭

아래에 빨강/초록 하이라이트로 수정 차이 표시

교정 결과를 CSV로 다운로드 가능

📘 모델 정보

Base Model: RoBERTa

Fine-tuned Task: Grammar & Spelling Error Correction

Training Dataset: 약 1.9M 문장 (synthetic + BEA 기반)

Training Objective: Token-level edit-tag classification
(KEEP / REPLACE / DELETE / ADD / TRANSFORM 등)

✨ 예시
입력 문장	교정 결과
She go to school every days.	She goes to school every day.
I recieve teh package yesterday.	I received the package yesterday.
📄 License

이 프로젝트는 학습 및 연구 목적의 데모용으로 배포됩니다.
상업적 사용 전 반드시 라이선스를 확인하세요.

🧑‍💻 Author

이서율 (SeoYool Lee)
AI / NLP Research & Development
📧 Contact: (optional: dldmstj0531@gmail.com)