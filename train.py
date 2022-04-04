from argparse import ArgumentParser
from datetime import datetime
import hashlib
import os
from random import randint

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import youtokentome as yttm

from transformer_model import TransformerMT
from data_utils import MTData

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3


def compute_grad_norm():
    total_norm = 0
    parameters = [
        p for p in model.parameters()
        if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_path')
    parser.add_argument(
        '--device', default='0', help='GPU device id.'
    )
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--max_grad_norm', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--logdir', type=str, help='Path to directory '
        'that will contain logs.'
    )
    parser.add_argument('--savedir', type=str, help='Path to directory '
        'that will contain the trained_model.'
    )
    parser.add_argument(
        '--train_source', type=str, help='Path to .txt file '
        'containing source sentences for training.'
    )
    parser.add_argument(
        '--train_target', type=str, help='Path to .txt file '
        'containing target sentences for training.'
    )
    parser.add_argument(
        '--valid_source', type=str, help='Path to .txt file '
        'containing source sentences for validation.'
    )
    parser.add_argument(
        '--valid_target', type=str, help='Path to .txt file '
        'containing source sentences for validation.'
    )
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    tokenizer = yttm.BPE(args.tokenizer_path)

    max_seq_length = args.max_seq_length
    device = torch.device(f'cuda:{args.device}')

    model = TransformerMT(
        vocab_size=tokenizer.vocab_size(), max_seq_length=max_seq_length,
        num_encoder_layers=args.n_layers, num_decoder_layers=args.n_layers
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model.to(device)

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = args.lr

    optimizer = optim.AdamW(
        params=model.parameters(),
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98), eps=1e-9, lr=lr
    )
    loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    hash_f = hashlib.sha256()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    hash_f.update(bytes(current_time, encoding='ascii'))
    model_hash = str(hash_f.hexdigest())[:16]

    logdir = os.path.join(args.logdir, model_hash)
    savedir = os.path.join(args.savedir, model_hash + '.pt')
    writer = SummaryWriter(log_dir=logdir, flush_secs=60, max_queue=2)
    writer.add_text(tag='Hyperparameters', text_string=str(args.__dict__))
    writer.add_text(tag='Optimizer', text_string=optimizer.__repr__())

    def sentences_to_tokens(batch: list):
        source_sentences, target_sentences = list(zip(*batch))
        source_tokens = tokenizer.encode(
            source_sentences, output_type=yttm.OutputType.ID,
            bos=True, eos=True
        )
        target_tokens = tokenizer.encode(
            target_sentences, output_type=yttm.OutputType.ID,
            bos=True, eos=True
        )

        padded_source = nn.utils.rnn.pad_sequence(
            [torch.tensor(_) for _ in source_tokens],
            padding_value=PAD_IDX
        )
        padded_target = nn.utils.rnn.pad_sequence(
            [torch.tensor(_) for _ in target_tokens],
            padding_value=PAD_IDX
        )
        ps = padded_source[:max_seq_length]
        pt = padded_target[:max_seq_length]
        return ps, pt


    train_data = MTData(
        source_path=args.train_source, target_path=args.train_target,
        lowercase=False
    )

    val_data = MTData(
        source_path=args.valid_source, target_path=args.valid_target,
        lowercase=False
    )

    num_workers = args.num_workers

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False, sampler=None,
        batch_sampler=None, num_workers=num_workers, pin_memory=True,
        collate_fn=sentences_to_tokens
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, sampler=None,
        batch_sampler=None, num_workers=num_workers, pin_memory=True,
        collate_fn=sentences_to_tokens
    )

    def prepare_data(source: torch.Tensor, target: torch.Tensor) -> dict:
        ss, _ = source.size()
        ts, _ = target.size()
        source_padding_mask = (source == PAD_IDX).T
        target_padding_mask = (target == PAD_IDX).T
        src_mask = torch.zeros(ss, ss)
        tgt_mask = torch.triu(
            torch.ones(ts, ts), diagonal=1
        ) == 1
        kwargs = {
            'src': source, 'tgt': target,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'src_key_padding_mask': source_padding_mask,
            'tgt_key_padding_mask': target_padding_mask,
        }
        return kwargs


    def train_iteration(epoch_number: int) -> float:
        model.train()

        loss_over_period = 0
        step = len(train_dataloader) * epoch_number

        for batch in tqdm(
            train_dataloader, desc=f'Training, epoch #{epoch_number + 1}'
        ):

            optimizer.zero_grad()
            source, target = batch

            input_target = target[:-1]
            kwargs = prepare_data(source=source, target=input_target)

            for k, v in kwargs.items():
                kwargs[k] = v.to(device)

            scores = model(**kwargs)
 
            _, _, vocab_size = scores.size()
            scores = scores.view(-1, vocab_size)

            true_tokens = target[1:].to(device).view(-1)
            loss_value = loss(scores, true_tokens)

            loss_over_period += loss_value.item()

            loss_value.backward()

            grad_norm = compute_grad_norm()

            if grad_norm > args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=args.max_grad_norm
                )

            if randint(0, 100) > 90:
                writer.add_scalar(
                    tag=f'Grad norm', scalar_value=grad_norm,
                    global_step=step
                )

            optimizer.step()
            step += 1

        loss_over_period /= len(train_dataloader)

        writer.add_scalar(
            tag='Loss/train', scalar_value=loss_over_period,
            global_step=epoch_number
        )
        return loss_over_period

    def validation_iteration(epoch_number: int) -> float:
        model.eval()
        loss_over_period = 0

        for batch in tqdm(
            val_dataloader, desc=f'Validating, epoch #{epoch_number + 1}'
        ):

            source, target = batch

            input_target = target[:-1]
            kwargs = prepare_data(source=source, target=input_target)

            for k, v in kwargs.items():
                kwargs[k] = v.to(device)

            with torch.no_grad():
                scores = model(**kwargs)
 
            _, _, vocab_size = scores.size()
            scores = scores.view(-1, vocab_size)

            true_tokens = target[1:].to(device).view(-1)
            loss_value = loss(scores, true_tokens)

            loss_over_period += loss_value.item()

        loss_over_period /= len(val_dataloader)
        writer.add_scalar(
            tag='Loss/validation', scalar_value=loss_over_period,
            global_step=epoch_number
        )
        return loss_over_period

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        train_iteration(epoch_number=epoch)
        validation_loss = validation_iteration(epoch_number=epoch)
    
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), savedir)