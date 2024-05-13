# TODO: rewrite this to sequential and maybe switch it to jax/flax?
from torch.func import vmap, grad, functional_call
import torch.optim as optim
import torch.nn.functional as F
import torch
import time
import functools
from functools import partial
import numpy as np

class MAML:
    # Trains a model for n_inner_iter using the support and returns a loss
    # using the query.
    def loss_for_task(self, net, n_inner_iter, inner_lr, x_spt, y_spt, x_qry, y_qry):
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        querysz = x_qry.size(0)

        def compute_loss(new_params, buffers, x, y):
            logits = functional_call(net, (new_params, buffers), x)
            loss = F.cross_entropy(logits, y)
            return loss

        new_params = params
        for _ in range(n_inner_iter):
            grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
            new_params = {k: new_params[k] - g * inner_lr for k, g, in grads.items()}

        qry_logits = functional_call(net, (new_params, buffers), x_qry)
        qry_loss = F.cross_entropy(qry_logits, y_qry)
        qry_acc = (qry_logits.argmax(dim=1) == y_qry).sum() / querysz

        return qry_loss, qry_acc

    def train(self, db, net, device, meta_opt, epoch, num_adaption_steps, inner_lr):
        start_time = time.time()

        batch = next(db)
        x_spt, y_spt = batch['train']
        x_qry, y_qry = batch['test']
        x_spt, y_spt = x_spt.to(device), y_spt.to(device)
        x_qry, y_qry = x_qry.to(device), y_qry.to(device)

        task_num, setsz, c_, h, w = x_spt.size()

        meta_opt.zero_grad()

        # In parallel, trains one model per task. There is a support (x, y)
        # for each task and a query (x, y) for each task.
        compute_loss_for_task = functools.partial(self.loss_for_task, net, num_adaption_steps, inner_lr)
        qry_losses, qry_accs = vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)

        # Compute the maml loss by summing together the returned losses.
        qry_losses.sum().backward()

        meta_opt.step()

        qry_losses = qry_losses.detach().sum() / task_num
        qry_accs = qry_accs.sum() / task_num
        iter_time = time.time() - start_time
        if epoch % 100 == 0:
            print(f'[Iteration {epoch}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}')
        return qry_losses, qry_accs

    def test(self, db, net, device, num_adaption_steps, inner_lr):
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        max_batches = 40
        db = iter(db) 

        qry_losses = []
        qry_accs = []

        for batch_idx in range(max_batches):
            batch = next(db)
            x_spt, y_spt = batch['train']
            x_qry, y_qry = batch['test']
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)

            task_num, setsz, c_, h, w = x_spt.size()

            # TODO: Fix gpu memory issue and run test in parallel as well.
            # This works in theory, but runs out of memory for large batches.
            # compute_loss_for_task = functools.partial(self.loss_for_task, net, num_adaption_steps)
            # qry_loss, qry_acc = vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)
            # qry_losses.append(qry_loss)
            # qry_accs.append(qry_acc)

            for i in range(task_num):
                new_params = params
                for _ in range(num_adaption_steps):
                    spt_logits = functional_call(net, (new_params, buffers), x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    grads = torch.autograd.grad(spt_loss, new_params.values())
                    new_params = {k: new_params[k] - g * inner_lr for k, g, in zip(new_params, grads)}

                # The query loss and acc induced by these parameters.
                qry_logits = functional_call(net, (new_params, buffers), x_qry[i]).detach()
                qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction='none')

                qry_losses.append(qry_loss.detach())
                qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach()) 

        qry_losses = torch.cat(qry_losses).mean().item()
        qry_accs = 100. * torch.cat(qry_accs).float().mean().item()

        return qry_losses, qry_accs