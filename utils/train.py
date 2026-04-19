import sys
from tqdm import tqdm
import torch

def VAE_train(model, train_loader, test_loader, criterion, optimiser, device, epochs=50,
              patience=3, anneal_epochs=10, save_every: int = 10, save_path: str = 'checkpoint.pt', tqd_bar=True):
    """we keep the training loop shared so both architectures get the same optimisation logic"""
    losses = {'train_tloss': [], 'train_recon_loss':[], 'train_kl_loss':[],
              'test_tloss': [], 'test_recon_loss':[], 'test_kl_loss':[] }
    orig_beta = criterion.beta
    model = model.to(device)
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    # we use AMP on CUDA because the project spends a lot of time in matrix ops and
    # there's not much value in paying full fp32 cost for this comparison
    use_amp = device.type == 'cuda'
    # we start the scaler a bit lower because the summed reconstruction loss can spike
    # early on and we don't want the scaler collapsing straight away
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=2**10)

    # we only compile when PyTorch supports it because it's basically free speed
    # for the long training runs and doesn't change the notebook logic
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    bar = tqdm(range(epochs), desc="Training", 
               unit="epoch", postfix={'avg loss': '—'},  file=sys.stderr, dynamic_ncols=True) if tqd_bar else range(epochs)
    for epoch in bar:
        model.train()
        # we ramp beta in slowly because hitting the KL term too hard from epoch 1
        # makes these sequence VAEs collapse much more easily
        prog = min(1, epoch/anneal_epochs)
        current_beta = prog * orig_beta
        criterion.beta = current_beta
        train_tloss, train_recon_loss, train_kl_loss = 0, 0, 0

        for x, _ in train_loader:  # we ignore (year, accession) metadata
            x = x.to(device).float()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                x_hat, mu, logvar = model(x)
            # we cast back to fp32 for the loss because the cross-entropy sum gets noisy
            # in half precision on long sequences
            loss, recon_loss, kl_loss = criterion(x_hat.float(), x, mu.float(), logvar.float())
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            # we unscale before clipping so the clip happens in fp32 and doesn't hide
            # exploding gradients behind the scaler
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            # we keep these on device until the end of the epoch so we're not syncing
            # back to CPU on every batch
            train_tloss += loss.detach()
            train_recon_loss += recon_loss.detach()
            train_kl_loss += kl_loss.detach()

        # we normalise by sample count instead of batch count so the curves still mean
        # the same thing if we change batch size between experiments
        n_samples = len(train_loader.dataset)
        avg_train_loss = (train_tloss / n_samples).item()
        avg_train_recon_loss = (train_recon_loss / n_samples).item()
        avg_train_kl_loss = (train_kl_loss / n_samples).item()
        losses['train_tloss'].append(avg_train_loss)
        losses['train_recon_loss'].append(avg_train_recon_loss)
        losses['train_kl_loss'].append(avg_train_kl_loss)
        if tqd_bar:
            bar.set_postfix({'avg loss':f'{avg_train_loss:.4f}'}, refresh=False)

        model.eval()
        test_tloss, test_recon_loss, test_kl_loss = 0, 0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    x_hat, mu, logvar = model(x)
                loss, recon_loss, kl_loss = criterion(x_hat.float(), x, mu.float(), logvar.float())
                test_tloss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()

        n_test = len(test_loader.dataset)
        avg_test_loss = test_tloss / n_test
        losses['test_tloss'].append(avg_test_loss)
        losses['test_recon_loss'].append(test_recon_loss / n_test)
        losses['test_kl_loss'].append(test_kl_loss / n_test)

        if epoch < anneal_epochs:
            # we ignore warmup epochs for early stopping because a tiny beta makes the
            # loss look artificially good before the real objective kicks in
            best_test_loss = float('inf')
            patience_counter = 0
        else:
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                best_model_weights = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"\nEarly stopping triggered at Epoch {epoch+1}.")
                    if best_model_weights is not None:
                        model.load_state_dict(best_model_weights)
                    break

        # we keep checkpointing available because the transformer runs are expensive enough
        # that losing a session would be painful
        if save_every and (epoch + 1) % save_every == 0:
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(), 'losses': losses},
                       save_path)
            tqdm.write(f"  Checkpoint saved at epoch {epoch+1} -> {save_path}")

    # we still restore the best weights at the end so the return value is the model
    # we actually want to analyse later, not just the final epoch
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, losses
