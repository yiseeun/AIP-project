# 1. 라이브러리 임포트
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 2. 설정값
VOCAB_SIZE = 10000   # 사용할 단어 개수 (상위 10,000개)
MAX_LEN = 100        # 문장 최대 길이
NOISE_RATIO = 0.3    # 단어를 지울 확률
BATCH_SIZE = 128

print("▶ 설정 완료")
print(f"- 단어 집합 크기: {VOCAB_SIZE}")
print(f"- 문장 최대 길이: {MAX_LEN}")
print(f"- 노이즈 비율: {NOISE_RATIO * 100}%")

# 3. 데이터 로드(IMDB)
print("\n▶ 데이터 다운로드 및 로딩 중...")
(train_data, _), (test_data, _) = imdb.load_data(num_words=VOCAB_SIZE)

print(f"학습 데이터 개수: {len(train_data)}개")
print(f"테스트 데이터 개수: {len(test_data)}개")

# 4. 패딩
print("\n▶ 패딩 작업 중...")
train_data = pad_sequences(train_data, maxlen=MAX_LEN, padding='post', truncating='post')
test_data  = pad_sequences(test_data,  maxlen=MAX_LEN, padding='post', truncating='post')

print(f"전처리 완료 train_data shape: {train_data.shape}")
print(f"전처리 완료 test_data  shape: {test_data.shape}")

# 5. 노이즈 주입 함수
def add_noise(sequences, noise_ratio=0.1):
    """
    시퀀스에서 일정 비율로 단어를 0(PAD/MASK)으로 바꿔서 노이즈를 줌.
    """
    noisy_sequences = sequences.copy()
    rows, cols = noisy_sequences.shape

    for r in range(rows):
        for c in range(cols):
            if noisy_sequences[r, c] == 0:
                continue
            # noise_ratio 확률로 단어 삭제 (0으로 변경)
            if np.random.rand() < noise_ratio:
                noisy_sequences[r, c] = 0

    return noisy_sequences

print("\n▶ 노이즈 주입 중...")
train_data_noisy = add_noise(train_data, NOISE_RATIO)
test_data_noisy  = add_noise(test_data,  NOISE_RATIO)
print("노이즈 주입 완료!")

# 6. 학습용 Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_data_noisy, train_data))
train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data_noisy, test_data))
test_dataset = test_dataset.batch(BATCH_SIZE)

print("\n▶ tf.data.Dataset 생성 완료")

# 7. 숫자를 단어로 바꾸는 부분
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = '[PAD/MASK]'
reverse_word_index[1] = '[START]'
reverse_word_index[2] = '[UNK]'

def decode_review(text_seq):
    return ' '.join([reverse_word_index.get(i, '?') for i in text_seq])

print("\n" + "="*50)
print("             결과 확인 (Sample 0번)")
print("="*50)
print("\n[1] 정답 문장 (Target):")
print(decode_review(train_data[0]))

print("\n[2] 노이즈 들어간 문장 (Input):")
print(decode_review(train_data_noisy[0]))
print("\n" + "="*50)

print("\n[안내]")
print("모델 학습 시:")
print("  - 입력(x): 노이즈 데이터 (train_data_noisy 또는 train_dataset의 첫 번째)")
print("  - 정답(y): 원본 데이터 (train_data 또는 두 번째)")
print("예시) model.fit(train_data_noisy, train_data, ...)")
