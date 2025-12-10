# 1. 라이브러리 임포트
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, TimeDistributed

# 2. 설정값
VOCAB_SIZE = 10000   # 사용할 단어 개수 (상위 10,000개)
MAX_LEN = 100        # 문장 최대 길이
NOISE_RATIO = 0.3    # 단어를 지울 확률
BATCH_SIZE = 128
EMBEDDING_DIM = 128  # 추가 설정값: 임베딩 차원
GRU_UNITS = 128      # 추가 설정값: GRU 유닛 수
EPOCHS = 10          # 학습 파라미터: 에포크 수

# 3. 데이터 로드(IMDB)
(train_data, _), (test_data, _) = imdb.load_data(num_words=VOCAB_SIZE)

# 4. 패딩
train_data = pad_sequences(train_data, maxlen=MAX_LEN, padding='post', truncating='post')
test_data  = pad_sequences(test_data,  maxlen=MAX_LEN, padding='post', truncating='post')

# 5. 노이즈 주입 함수
def add_noise(sequences, noise_ratio=0.1):
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

train_data_noisy = add_noise(train_data, NOISE_RATIO)
test_data_noisy  = add_noise(test_data,  NOISE_RATIO)

# 6. 학습용 Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_data_noisy, train_data))
train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data_noisy, test_data))
test_dataset = test_dataset.batch(BATCH_SIZE)

input_layer = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(input_layer)

# 인코더
encoder_output, state_h = GRU(GRU_UNITS, return_sequences=False, return_state=True)(embedding_layer)

# 디코더
decoder_input = embedding_layer 
decoder_gru = GRU(GRU_UNITS, return_sequences=True)(decoder_input, initial_state=state_h)

# 출력 레이어
decoder_output = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_gru)

model = Model(inputs=input_layer, outputs=decoder_output)


# 옵티마이저 설정: Adam (Learning Rate = 0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 손실 함수 설정: Sparse Categorical Crossentropy (정수 인코딩, 패딩 토큰(0) 무시)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, 
    ignore_class=0 
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset
)
