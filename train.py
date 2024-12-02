from model import CFM, UNetT, DiT, Trainer
from model.utils import get_tokenizer
from model.dataset import load_dataset


# -------------------------- Dataset Settings --------------------------- #

target_sample_rate = 16000
n_mel_channels = 128
hop_length = 160

tokenizer = "char"  # 'pinyin', 'char', or 'custom'
# tokenizer_path = 'WenetSpeech4TTS_Premium_pinyin/vocab.txt'  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
# tokenizer_path = 'aihub_dialects_heavy/vocab.txt'
tokenizer_path = "data/aihub_dialects.txt" # DEPRECATED
# dataset_name = "Emilia_ZH_EN"
# dataset_name = "WenetSpeech4TTS_Premium"
dataset_name = 'aihub_dialects_heavy'

# -------------------------- Training Settings -------------------------- #

# exp_name = "F5TTS_Base_wenet"  # F5TTS_Base | E2TTS_Base
exp_name = "F5TTS_Base_aihub_heavy"

learning_rate = 7.5e-5

batch_size_per_gpu = 9600  # 8 GPUs, 8 * 19200 = 153600
batch_size_type = "frame"  # "frame" or "sample"
# max_samples = 16  # max sequences per batch if use frame-wise batch_size. we set 32 for small models, 64 for base models
max_samples = 6
grad_accumulation_steps = 1  # note: updates = steps / grad_accumulation_steps
max_grad_norm = 1.0

epochs = 100  # use linear decay, thus epochs control the slope
num_warmup_updates = 20000  # warmup steps
save_per_updates = 50000  # save checkpoint per steps
last_per_steps = 5000  # save last checkpoint per steps

# model params
if exp_name == "F5TTS_Base_aihub_heavy":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
elif exp_name == "E2TTS_Base":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)


# ----------------------------------------------------------------------- #


def main():
    if tokenizer == "custom":
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = dataset_name
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    mel_spec_kwargs = dict(
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        model,
        epochs,
        learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        checkpoint_path=f"ckpts/{exp_name}",
        batch_size=batch_size_per_gpu,
        batch_size_type=batch_size_type,
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        wandb_project="AIHUB-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=last_per_steps,
    )

    train_dataset = load_dataset(dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
