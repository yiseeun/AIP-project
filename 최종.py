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

# --- 텍스트 시퀀스용 GRU 기반 Encoder/Decoder 클래스 정의 ---

class Encoder(tf.keras.layers.Layer):
    def __init__(self, gru_units, embedding_dim, vocab_size, max_len):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True)
        self.gru = GRU(gru_units, return_sequences=False, return_state=True, name='encoder_gru')

    def call(self, input_features):
        embedded = self.embedding(input_features)
        # GRU의 최종 상태(state_h)를 인코딩된 잠재 벡터로 사용
        output_sequence, state_h = self.gru(embedded)
        return state_h, embedded

class Decoder(tf.keras.layers.Layer):
    def __init__(self, gru_units, vocab_size):
        super(Decoder, self).__init__()
        self.gru = GRU(gru_units, return_sequences=True, name='decoder_gru')
        self.output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))

    def call(self, decoder_input_and_state):
        embedded_input, initial_state = decoder_input_and_state
        gru_output = self.gru(embedded_input, initial_state=initial_state)
        return self.output_layer(gru_output)

class Autoencoder(tf.keras.Model):
    def __init__(self, vocab_size, max_len, embedding_dim, gru_units):
        super(Autoencoder, self).__init__()
        self.loss_history = [] # 이전 코드의 self.loss를 대체
        self.encoder = Encoder(gru_units, embedding_dim, vocab_size, max_len)
        self.decoder = Decoder(gru_units, vocab_size)

    def call(self, input_features):
        state_h, embedded_input = self.encoder(input_features)
        reconstructed = self.decoder((embedded_input, state_h))
        return reconstructed

# --- 모델 초기화 및 학습 ---
model = Autoencoder(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, GRU_UNITS)

# 옵티마이저 설정
opt = Adam(learning_rate=LEARNING_RATE)

# 손실 함수 설정: Sparse Categorical Crossentropy (텍스트 시퀀스 복원에 적합)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, 
    ignore_class=0  # 패딩 토큰(0) 무시
)

model.compile(
    optimizer=opt,
    loss=loss_fn,
    metrics=['accuracy']
)

# Keras의 표준 fit 메서드 사용 (수동 train_loop 대체)
print("\n--- 모델 학습 시작 (GRU Denoising Autoencoder) ---")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset
)

print("\n--- 모델 학습 완료 ---")

# --- 시각화 (Loss 그래프) ---
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# --- 텍스트 복원 시각화 (이전 MNIST 시각화 대체) ---
# 단어 인덱스 -> 단어 매핑을 위한 딕셔너리 준비
word_index = imdb.get_word_index()
index_to_word = {v + 3: k for k, v in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"

def decode_sequence(sequence, index_to_word):
    return ' '.join([index_to_word.get(i, '?') for i in sequence if i > 3])

# 테스트 데이터 샘플 선택 및 복원 확인
sample_index = 5
input_noisy = test_data_noisy[sample_index:sample_index+1]
target_original = test_data[sample_index:sample_index+1]

predictions = model.predict(input_noisy)
predicted_sequence = np.argmax(predictions, axis=-1)[0]

print("\n--- 복원 테스트 (샘플) ---")
print(f"**잡음이 섞인 입력 (디코딩):**")
print(' '.join([index_to_word.get(i, '?') for i in input_noisy[0] if i != 0]))

print("\n**원본 타겟 (디코딩):**")
print(decode_sequence(target_original[0], index_to_word))

print("\n**복원된 예측 시퀀스 (디코딩):**")
print(decode_sequence(predicted_sequence, index_to_word))
