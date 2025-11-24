#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-0.6B 댓글 감성 분류 - Getting Started
한 파일로 학습부터 제출 파일 생성까지 모든 과정을 수행합니다.
Google Colab에서 바로 실행 가능합니다.
"""

# ============================================================================
# 1. 패키지 설치 및 임포트
# ============================================================================

import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Colab 환경 체크 및 패키지 설치
def install_packages():
    """필요한 패키지 설치"""
    try:
        import google.colab
        IN_COLAB = True
        print("🎯 Google Colab 환경 감지")
    except:
        IN_COLAB = False
        print("💻 로컬 환경에서 실행 중")
    
    packages = [
        "transformers>=4.44.0",
        "datasets>=2.16.0",
        "accelerate>=1.0.0",
        "peft>=0.11.1",
        "bitsandbytes>=0.43.1",
        "sentencepiece",
        "protobuf==3.20.3"
    ]
    
    print("\n📦 필요한 패키지 설치 중...")
    for package in packages:
        print(f"install... {package}")
        os.system(f"pip install -q {package}")
    print("✅ 패키지 설치 완료!")

# 패키지 설치 실행
install_packages()

# 필요한 라이브러리 임포트
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel

# ============================================================================
# 2. 데이터셋 준비
# ============================================================================

def prepare_dataset():
    """댓글 분류 데이터셋 다운로드 및 준비"""
    
    print("\n" + "="*60)
    print("📊 데이터셋 준비")
    print("="*60)
    
    # 학습 데이터 다운로드
    print("\n1️⃣ 학습 데이터 다운로드 중...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tykimos/tykimos.github.io/master/warehouse/dataset/tarr_train.txt",
        filename="tarr_train.txt",
    )
    
    # 테스트 데이터 다운로드
    print("2️⃣ 테스트 데이터 다운로드 중...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tykimos/tykimos.github.io/master/warehouse/dataset/tarr_sample_submit.txt",
        filename="tarr_sample_submit.txt",
    )
    
    # 데이터 로드
    df_train = pd.read_csv('tarr_train.txt', delimiter='\t')
    df_test = pd.read_csv('tarr_sample_submit.txt', delimiter='\t')
    
    print(f"\n✅ 데이터 로드 완료:")
    print(f"   - 학습 데이터: {len(df_train)}개")
    print(f"   - 테스트 데이터: {len(df_test)}개")
    
    # 데이터 샘플 출력
    print("\n📝 데이터 샘플:")
    for i in range(min(3, len(df_train))):
        print(f"   [{i+1}] {df_train.iloc[i]['comment'][:50]}...")
        print(f"       레이블: {df_train.iloc[i]['label']}")
    
    return df_train, df_test

def create_instruction_dataset(df_train):
    """Instruction 형식으로 데이터셋 변환"""
    
    print("\n3️⃣ Instruction 형식 데이터셋 생성 중...")
    
    def create_prompt(comment, label=None):
        instruction = "다음 댓글이 긍정적인지 부정적인지 분류하세요. 긍정이면 1, 부정이면 0으로 답하세요."
        
        if label is not None:
            return f"""### Instruction:
{instruction}

### Input:
{comment}

### Response:
{label}"""
        else:
            return f"""### Instruction:
{instruction}

### Input:
{comment}

### Response:"""
    
    # 학습 데이터 변환
    train_data = []
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="데이터 변환"):
        text = create_prompt(row['comment'], str(row['label']))
        train_data.append({'text': text})
    
    # 학습/검증 분할 (90:10)
    split_idx = int(len(train_data) * 0.9)
    train_dataset = Dataset.from_list(train_data[:split_idx])
    val_dataset = Dataset.from_list(train_data[split_idx:])
    
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    # 데이터셋 저장
    dataset.save_to_disk("comment_classification_dataset")
    
    print(f"✅ 데이터셋 생성 완료:")
    print(f"   - 학습: {len(dataset['train'])}개")
    print(f"   - 검증: {len(dataset['validation'])}개")
    
    return dataset

# ============================================================================
# 3. 모델 학습
# ============================================================================

def setup_model_and_tokenizer(model_name="Qwen/Qwen3-0.6B", use_4bit=True):
    """모델과 토크나이저 설정"""
    
    print(f"\n🤖 모델 로딩: {model_name}")
    
    # 4-bit 양자화 설정 (메모리 절약)
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora(model, r=8, lora_alpha=16):
    """LoRA 설정"""
    
    model = prepare_model_for_kbit_training(model)
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def train_model(dataset, output_dir="./qwen3-classification-finetuned"):
    """모델 학습"""
    
    print("\n" + "="*60)
    print("🎓 모델 학습")
    print("="*60)
    
    # 모델과 토크나이저 설정
    model, tokenizer = setup_model_and_tokenizer()
    
    # LoRA 설정
    print("\n⚙️ LoRA 설정 중...")
    model = setup_lora(model, r=8, lora_alpha=16)
    
    # 데이터 토크나이징
    print("\n📝 데이터 토크나이징...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=256
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="토크나이징"
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 학습 설정
    print("\n🔧 학습 설정...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # 빠른 학습을 위해 1 에폭
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),  # GPU 있으면 FP16 사용
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=60,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        optim="adamw_torch",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=42
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 학습 시작
    print("\n🚀 학습 시작...")
    print("   (약 5-10분 소요)")
    trainer.train()
    
    # 모델 저장
    print(f"\n💾 모델 저장: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ 학습 완료!")
    
    return output_dir

# ============================================================================
# 4. 추론 및 제출 파일 생성
# ============================================================================

def load_trained_model(model_path="./qwen3-classification-finetuned"):
    """학습된 모델 로드"""
    
    print("\n🔄 학습된 모델 로딩 중...")
    
    base_model_name = "Qwen/Qwen3-0.6B"
    
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 로드
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("✅ LoRA 어댑터 로드 완료")
    except:
        model = base_model
        print("⚠️ LoRA 어댑터를 찾을 수 없어 베이스 모델 사용")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def classify_comment(model, tokenizer, comment):
    """댓글 분류"""
    
    prompt = f"""### Instruction:
다음 댓글이 긍정적인지 부정적인지 분류하세요. 긍정이면 1, 부정이면 0으로 답하세요.

### Input:
{comment}

### Response:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Response 추출
    if "### Response:" in response:
        answer = response.split("### Response:")[-1].strip()
        for char in answer:
            if char in ['0', '1']:
                return int(char)
    
    return 1  # 기본값

def generate_submission(df_test, model_path="./qwen3-classification-finetuned"):
    """제출 파일 생성"""
    
    print("\n" + "="*60)
    print("📄 제출 파일 생성")
    print("="*60)
    
    # 모델 로드
    model, tokenizer = load_trained_model(model_path)
    model.eval()
    
    predicted_labels = []
    
    # 각 댓글 분류
    print("\n🔍 댓글 분류 진행...")
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="분류 진행"):
        comment = row['comment']
        predicted_label = classify_comment(model, tokenizer, comment)
        predicted_labels.append(predicted_label)
        
        # 처음 3개 샘플 출력
        if idx < 3:
            print(f"   [{idx+1}] {comment[:40]}...")
            print(f"       → 예측: {predicted_label}")
    
    # 예측 결과 추가
    df_test['label'] = predicted_labels
    
    # 결과 통계
    label_counts = pd.Series(predicted_labels).value_counts()
    print(f"\n📊 분류 결과:")
    print(f"   - 긍정(1): {label_counts.get(1, 0)}개")
    print(f"   - 부정(0): {label_counts.get(0, 0)}개")
    
    # 파일 저장
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'tarr_my_submit_{current_time}.txt'
    
    df_test[['id', 'comment', 'label']].to_csv(file_name, sep='\t', index=False)
    
    print(f"\n✅ 제출 파일 생성 완료: {file_name}")
    
    return file_name

# ============================================================================
# 5. 메인 실행 함수
# ============================================================================

def main():
    """전체 파이프라인 실행"""
    
    print("🎯 Qwen3-0.6B 댓글 감성 분류 시작")
    print("="*60)
    
    try:
        # 1. 데이터셋 준비
        df_train, df_test = prepare_dataset()
        
        # 2. 데이터셋 생성
        dataset = create_instruction_dataset(df_train)
        
        # 3. 모델 학습
        model_path = train_model(dataset)
        
        # 4. 제출 파일 생성
        submission_file = generate_submission(df_test, model_path)
        
        print("\n" + "="*60)
        print("🎉 모든 작업 완료!")
        print(f"📁 생성된 제출 파일: {submission_file}")
        print("="*60)
        
        # Colab에서 파일 다운로드
        try:
            from google.colab import files
            print("\n📥 제출 파일 다운로드 중...")
            files.download(submission_file)
        except:
            print(f"\n📌 제출 파일이 '{submission_file}'로 저장되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 6. 실행
# ============================================================================

if __name__ == "__main__":
    # GPU 확인
    if torch.cuda.is_available():
        print(f"🎮 GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 CPU 모드로 실행 (느릴 수 있습니다)")
    
    # 메인 함수 실행
    main()