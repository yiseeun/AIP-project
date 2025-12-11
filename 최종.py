# 1. 라이브러리 임포트
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, TimeDistributed
import matplotlib.pyplot as plt

# 2. 설정값
VOCAB_SIZE = 10000   # 사용할 단어 개수 (상위 10,000개)
MAX_LEN = 100        # 문장 최대 길이
NOISE_RATIO = 0.3    # 단어를 지울 확률
BATCH_SIZE = 128
EMBEDDING_DIM = 128  # 추가 설정값: 임베딩 차원
D_MODEL = EMBEDDING_DIM  # CNN 채널 수
N_BLOCKS = 6             # Dilated Residual 블록 수
KERNEL_SIZE = 3
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

# 6. 학습용 Dataset 생성 (PAD=0은 손실에서 제외)
def make_ds(x_noisy, y_clean, batch_size, shuffle=False):
    sample_weight = (y_clean != 0).astype("float32")  # PAD=0 → weight=0
    ds = tf.data.Dataset.from_tensor_slices((x_noisy, y_clean, sample_weight))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x_noisy), seed=0)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_dataset = make_ds(train_data_noisy, train_data, BATCH_SIZE, shuffle=True)
test_dataset  = make_ds(test_data_noisy,  test_data,  BATCH_SIZE, shuffle=False)

# 7. 모델 구조: Residual Dilated 1D CNN (channels_last: (B,T,C))
class ResidualDilatedBlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2, use_glu=True, norm="batch"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size는 홀수를 권장합니다다."
        self.use_glu = use_glu
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm_type = norm

        # dilated Conv1D (same padding)
        self.conv = tf.keras.layers.Conv1D(
            filters=channels * (2 if use_glu else 1),
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding="same"
        )
        # 1x1 Conv to project back to channels
        self.proj = tf.keras.layers.Conv1D(filters=channels, kernel_size=1, padding="same")

        if norm == "batch":
            self.norm = tf.keras.layers.BatchNormalization()
        elif norm == "layer":
            self.norm = tf.keras.layers.LayerNormalization()
        else:
            raise ValueError("norm must be 'batch' or 'layer'")

    def call(self, x, training=False):
        # x: (B,T,C)
        h = self.conv(x)  # (B,T,2C or C)
        if self.use_glu:
            a, b = tf.split(h, num_or_size_splits=2, axis=-1)  # (B,T,C),(B,T,C)
            h = a * tf.sigmoid(b)
        h = self.proj(h)                # (B,T,C)
        h = self.dropout(h, training=training)
        h = h + x                       # residual
        h = self.norm(h, training=training)
        return h

# CNN Denoiser Model 
def build_cnn_denoiser(vocab_size, d_model=256, n_blocks=6, kernel_size=3, dropout=0.2, use_glu=True, norm="batch"):
    inp = tf.keras.Input(shape=(MAX_LEN,), dtype="int32")          # (B,T)
    emb = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)(inp)  # (B,T,C)

    # Keras Conv1D는 mask를 자동 전파하지 않음, sample_weight로 PAD를 무시
    x = emb
    dilations = [1] + [2**i for i in range(1, n_blocks)]
    for d in dilations[:n_blocks]:
        x = ResidualDilatedBlock(d_model, kernel_size, dilation=d, dropout=dropout, use_glu=use_glu, norm=norm)(x)

    # 최종 분류 (각 위치 -> vocab 점수). 
    logits = tf.keras.layers.Dense(vocab_size, activation=None)(x)  # (B,T,V)
    return tf.keras.Model(inp, logits, name="CNN_Denoiser")

# 모델 생성
model = build_cnn_denoiser(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_blocks=N_BLOCKS,
    kernel_size=KERNEL_SIZE,
    dropout=0.2,
    use_glu=True,
    norm="batch",
)
model.summary()

# 옵티마이저
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

# from_logits=True (Dense에 activation=None이므로)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])


print("\n--- 모델 학습 시작 (CNN Denoising Autoencoder) ---")
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)
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
