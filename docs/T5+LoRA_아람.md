# [아람] `T5` + `LoRA`

## 1. 모델 아키텍처 및 구성

- **1-1. 기반 아키텍처 (Base Model)** : `t5-base`
  - Google이 사전 학습한 **Encoder-Decoder Transformer** 모델
- **1-2. 파인튜닝 구조** : `LoRA`
  - `peft` 라이브러리를 사용한 **파라미터 효율적 파인튜닝 (PEFT)** 방식
  - **LoRA 설정:**
    - **r (Rank)** : 64
    - **lora_alpha** : 64 (**r**과 동일하게 설정하여 스케일링)
    - **lora_dropout**: 0.1
  - **적용 모듈 (Target Modules):** `["q","k","v","o","wi","wo"]`
    - 어텐션 레이어(Q, K, V, O)와 T5의 피드포워드 네트워크(wi, wo) 내의 모든 Linear 레이어에 LoRA 어댑터를 적용하여, 모델의 광범위한 파라미터가 태스크에 적응

---

## 2. 과제 및 학습 방식

- **핵심 과제 (Task):** **문법 오류 수정 (Grammatical Error Correction, GEC)**
  - 문법/철자/구두점 오류가 있는 문장(Noise)을 문법적으로 올바른 문장(Clean)으로 변환
- **작업 유형 (Task Type):** **Sequence-to-Sequence (Seq2Seq)**
  - 하나의 시퀀스(오류 문장)를 다른 시퀀스(교정 문장)로 매핑
- **학습 방식 (Learning Method):** **지도 학습 기반 파인튜닝 (Supervised Fine-tuning)**
  - `(noise, clean)` 문장 쌍으로 구성된 데이터셋을 사용하여 학습
- **라벨 수 (Number of Labels):**
  - 분류(Classification) 모델이 아니므로 "라벨 수"가 고정되지 않음
  - "라벨"은 생성될 타겟 문장(토큰 시퀀스) 그 자체
  - **어휘 크기(Vocabulary Size)**가 중요하며, `t5-base` 토크나이저의 어휘 크기(약 32,128개)가 생성 가능한 토큰의 집합이 된다.

---

## 3. 데이터 처리 및 구성

- **학습 데이터셋:** `C4 (synthetic)` + `BEA-19 (human-annotated)`
  - **C4 (190K):** `60,000`개 샘플링
  - **BEA-19:** `20,000`개 \* 2 샘플링
  - **데이터 혼합 전략:** BEA-19 데이터를 **3배 오버샘플링 (Oversampling)** 하여 C4 데이터(60k)와 BEA-19 데이터(20k \* 3 = 60k)의 비율을 약 1:1로 맞춤.
  - **최종 학습 데이터:** 약 12만 개의 (C4 + BEA\*3) 혼합 및 셔플된 데이터 (`c4_bea_combined_train.csv`)
- **데이터 클리닝 (Cleaning):**
  - **결측치 제거:** `dropna`
  - **No-op 제거:** 원본(noise)과 수정본(clean)이 동일한 데이터 제거
  - **길이 필터링:** 3단어 미만 또는 128단어 초과 문장 제거
- **토큰화 (Tokenization):**
  - **Prefix:** `grammar correction:` (T5 모델에 작업을 지시하는 프롬프트)
  - **최대 토큰 수 (Max Length):**
    - `MAX_INPUT_LENGTH`: **128**
    - `MAX_TARGET_LENGTH`: **128**

---

## 4. 학습 및 평가 방식

- **학습 구성 (Training Configuration)**
  - **입력 (Input):** `grammar correction: [NOISE_SENTENCE]` (최대 128 토큰)
  - **출력 (Output):** `[CLEAN_SENTENCE]` (최대 128 토큰)
  - **Epochs:** 4
  - **Optimizer:** `adafactor`
  - **Learning Rate (LR):** `1e-4`
  - **LR 스케줄러:** `cosine`
  - **배치 크기 (Batch Size):**
    - `TRAIN_BS` (GPU당): 4
    - `GRAD_ACCUM` (누적): 8
    - **유효 배치 크기 (Effective Batch Size):** 4 \* 8 = **32**
  - **저장 (Saving):**
    - `save_strategy = "epoch"`: 매 에포크마다 모델 저장
    - `load_best_model_at_end = True`: 학습 종료 시 Validation `loss`가 가장 낮았던 체크포인트를 최종 모델로 로드합니다. (로그상 4번째 에포크 모델)
- **평가 지표 (Evaluation Metrics)**
  1. **학습 중 평가 (Validation Metric):** **SacreBLEU**
     • 학습 중에는 매 에포크마다 Validation set에 대한 BLEU 점수를 계산. (최종 `41.10`)
     • BLEU는 GEC의 표준 지표는 아니지만, 생성된 문장이 타겟 문장과 얼마나 유사한지 빠르게 확인하는 용도.
  2. **최종 평가 (Official Metric):** **ERRANT (M2 Scorer)**
     • 원본 파일(`orig`)을 모델로 예측하여 `submission.txt`를 생성

<br />

| Epoch | Training Loss | Validation Loss | Bleu      |
| ----- | ------------- | --------------- | --------- |
| 1     | 1.946600      | 1.871731        | 40.331337 |
| 2     | 1.920700      | 1.842324        | 40.879510 |
| 3     | 1.912700      | 1.835822        | 41.026049 |
| 4     | 1.891200      | 1.834465        | 41.102345 |

<br />

```
===============================
--- 임의 문장 테스트 시작 (시간 측정) ---
===============================


===== correct_sentence =====
Original:  He go to school every day.
Corrected: He goes to school every day.
Time taken: 0.3260 seconds


===== correct_sentence =====
Original:  She has two child.
Corrected: She has two children.
Time taken: 0.2922 seconds


===== correct_sentence =====
Original:  She is teacher.
Corrected: She is a teacher.
Time taken: 0.3335 seconds


===== correct_sentence =====
Original:  He arrived to the airport on time.
Corrected: He arrived at the airport on time.
Time taken: 0.3618 seconds


===== correct_sentence =====
Original:  I every day go to school.
Corrected: I go to school every day.
Time taken: 0.3735 seconds


===== correct_sentence =====
Original:  He told that he was tired.
Corrected: He told him that he was tired.
Time taken: 0.3943 seconds


===== correct_sentence =====
Original:  This is very importent information.
Corrected: This is very important information.
Time taken: 0.3731 seconds

--- 테스트 완료 ---
Total time for 7 sentences: 2.4544 seconds
Average time per sentence: 0.3506 seconds
```

---

## 5. 추론 로직 (Inference Logic)

- **생성 방식 (Generation):** **빔 서치 (Beam Search)**
  - 8개의 가능한 후보(beam)를 탐색하여 최적의 문장을 생성합니다.
- **핵심 파라미터:**
  - `NUM_BEAMS`: **8**
  - `LENGTH_PENALTY`: **0.7** (1.0보다 작으므로, 약간 더 짧은 문장을 생성하도록 유도)
  - `NO_REPEAT_NGRAM`: **3** (동일한 3-gram이 반복되는 것을 방지)
  - `REPETITION_PENALTY`: **1.07** (전반적인 단어 반복을 약하게 억제)
- **후처리 (Post-processing):**
  - **`post_detok`:** T5가 토큰화 과정에서 분리한 구두점(`.` , `?` 등)이나 하이픈(), 따옴표(`"`) 주변의 공백을 정규식을 사용해 깔끔하게 정리. (예: `easy - going` -> `easy-going`)

---

## 6. 한계점 및 관찰 (Limitations & Observations)

1. 단순/명확한 오류는 매우 잘 교정
   - 임의 테스트 결과, 수일치(`go` $\rightarrow$ `goes`), 복수형(`child` $\rightarrow$ `children`), 관사(`a teacher`), 철자(`important`) 등 짧은 문장 내의 명시적인 오류는 성공적으로 교정.
   - 추론 속도도 문장당 평균 0.35초.
2. 과잉 교정 발생
   - 공식 테스트셋에서 문법적으로 올바른 문장(`apply for a job`)을 오히려 틀린 문장(`apply to a job`)으로 수정하는 오류가 발생.
   - 모델이 학습 데이터의 특정 패턴에 과적합되어, 맥락을 무시하고 수정을 감행한 것으로 보임.
3. 길고 복잡한 문맥 파악에 취약
   - 모델이 단순한 규칙 수정에는 강점을 보였지만, 문장이 길어지거나 여러 절이 엮여 전후 문맥을 전체적으로 이해해야 하는 복잡한 교정에는 한계를 보임.
   - 예를 들어, 5번 샘플에서 `...for 20 years ago` 부분은 문맥상 어색함에도 불구하고 수정하지 못함.

### **================= 끝 =================**
