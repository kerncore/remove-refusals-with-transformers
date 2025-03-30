import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

# === НАСТРОЙКИ ===
MODEL_PATH = "/workspace/remove-refusals-with-transformer/llama-3.3-ft"
REFUSAL_VECTOR_PATH = MODEL_PATH.replace("/", "_") + "_refusal_dir.pt"

# === ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("[*] Загружаем модель...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# === УСТАНОВКА PADDING TOKEN ===
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# === ЗАГРУЗКА ВЕКТОРА ОТКАЗА ===
refusal_dir = torch.load(REFUSAL_VECTOR_PATH).to(model.device)

# === АБЛЯЦИЯ — УДАЛЕНИЕ "вредного" НАПРАВЛЕНИЯ ===
def direction_ablation_hook(activation: torch.Tensor, direction: torch.Tensor, alpha=1.0) -> torch.Tensor:
    while direction.ndim < activation.ndim:
        direction = direction.unsqueeze(0)
    proj = torch.sum(activation * direction, dim=-1, keepdim=True) * direction
    return activation - alpha * proj

# === ДОБАВОЧНЫЙ СЛОЙ ДЛЯ МОДЕЛИ ===
class AblationDecoderLayer(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha

    def forward(self, hidden_states, *args, **kwargs):
        ablated = direction_ablation_hook(hidden_states, refusal_dir, alpha=self.alpha)
        return (ablated,) + ((kwargs.get("past_key_value"),) if kwargs.get("use_cache") else ())

# === ВСТАВКА ОДНОГО СЛОЯ В ЦЕНТР МОДЕЛИ ===
print("[*] Внедряем refusal_dir в модель...")
try:
    layers = model.model.layers
except AttributeError:
    layers = model.model.decoder.layers  # fallback (другие архитектуры)

layer_idx = int(len(layers) * 0.6)  # как при генерации refusal_dir
layers.insert(layer_idx, AblationDecoderLayer(alpha=0.7))

# === ЧАТ-ЦИКЛ ===
print(f"\n🚀 Чат с моделью {MODEL_PATH} с вектором отказа активирован!\n")
conversation = []
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

while True:
    user_input = input("🧑: ")
    if user_input.strip() == "":
        continue

    conversation.append({"role": "user", "content": user_input})

    toks = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = (toks != tokenizer.pad_token_id).long()

    output = model.generate(
        toks,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    decoded = tokenizer.batch_decode(output[0][toks.shape[-1]:], skip_special_tokens=True)
    assistant_reply = "".join(decoded).strip()
    conversation.append({"role": "assistant", "content": assistant_reply})
    print()
