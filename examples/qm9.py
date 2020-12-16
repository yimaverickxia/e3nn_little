# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
import argparse
import datetime
import itertools
import pickle
import subprocess
import time

import torch
import wandb

from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet

from e3nn_little.nn.models.qm9 import Network


def execute(args):
    path = 'QM9'
    dataset = QM9(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # Report meV instead of eV.
    units = 1000 if args.target in [2, 3, 4, 6, 7, 8, 9, 10] else 1

    _, datasets = SchNet.from_qm9_pretrained(path, dataset, args.target)
    train_dataset, val_dataset, _test_dataset = datasets

    model = Network(
        muls=(args.mul0, args.mul1, args.mul2),
        ps=(1,) if 'shp' in args.opts else (1, -1),
        lmax=args.lmax,
        num_layers=args.num_layers,
        rad_gaussians=args.rad_gaussians,
        rad_hs=(args.rad_h,) * args.rad_layers + (args.rad_bottleneck,),
        mean=args.mean, std=args.std,
        atomref=dataset.atomref(args.target),
    )
    model = model.to(device)

    wandb.watch(model)

    loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    for step, data in enumerate(loader):
        print('profile', step, flush=True)
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(), record_shapes=True) as prof:
            data = data.to(device)
            pred = model(data.z, data.pos, data.batch)
            mse = (pred.view(-1) - data.y[:, args.target]).pow(2)
            mse.mean().backward()
        if step == 5:
            prof.export_chrome_trace(f"{datetime.datetime.now()}.json")
            break

    modules = [model.embedding, model.radial] + list(model.layers) + [model.atomref]
    lrs = [0.1, 1] + [1] * len(model.layers) + [0.1]
    param_groups = []
    for lr, module in zip(lrs, modules):
        jac = []
        for data in DataLoader(train_dataset[:20]):
            data = data.to(device)
            jac += [torch.autograd.grad(model(data.z, data.pos), module.parameters())[0].flatten()]
        jac = torch.stack(jac)
        kernel = jac @ jac.T
        print('kernel({}) = {:.2e} +- {:.2e}'.format(module, kernel.mean().item(), kernel.std().item()), flush=True)
        lr = lr / (kernel.mean() + kernel.std()).item()
        param_groups.append({
            'params': list(module.parameters()),
            'lr': lr,
        })

    lrs = torch.tensor([x['lr'] for x in param_groups])
    lrs = args.lr * lrs / lrs.max().item()

    for group, lr in zip(param_groups, lrs):
        group['lr'] = lr.item()

    optim = torch.optim.Adam(param_groups)
    print(optim, flush=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=25, factor=0.5, verbose=True)

    dynamics = []
    wall = time.perf_counter()
    wall_print = time.perf_counter()

    for epoch in itertools.count():

        errs = []
        loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        for step, data in enumerate(loader):
            data = data.to(device)

            pred = model(data.z, data.pos, data.batch)
            optim.zero_grad()
            (pred.view(-1) - data.y[:, args.target]).pow(2).mean().backward()
            optim.step()

            err = pred.view(-1) - data.y[:, args.target]
            errs += [err.cpu().detach()]

            if time.perf_counter() - wall_print > 15:
                wall_print = time.perf_counter()
                w = time.perf_counter() - wall
                e = epoch + (step + 1) / len(loader)
                print((
                    f'[{e:.1f}] ['
                    f'wall={w / 3600:.2f}h '
                    f'wall/epoch={w / e:.0f}s '
                    f'wall/step={1e3 * w / e / len(loader):.0f}ms '
                    f'step={step}/{len(loader)} '
                    f'mae={units * torch.cat(errs)[-200:].abs().mean():.5f} '
                    f'lr={min(x["lr"] for x in optim.param_groups):.1e}-{max(x["lr"] for x in optim.param_groups):.1e}]'
                ), flush=True)

        train_err = torch.cat(errs)

        errs = []
        loader = DataLoader(val_dataset, batch_size=256)
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.z, data.pos, data.batch)

            err = pred.view(-1) - data.y[:, args.target]
            errs += [err.cpu().detach()]
        val_err = torch.cat(errs)

        lrs = [
            x['lr']
            for x in optim.param_groups
        ]
        dynamics += [{
            'epoch': epoch,
            'wall': time.perf_counter() - wall,
            'train': {
                'mae': {
                    'mean': units * train_err.abs().mean().item(),
                    'std': units * train_err.abs().std().item(),
                },
                'mse': {
                    'mean': units * train_err.pow(2).mean().item(),
                    'std': units * train_err.pow(2).std().item(),
                }
            },
            'val': {
                'mae': {
                    'mean': units * val_err.abs().mean().item(),
                    'std': units * val_err.abs().std().item(),
                },
                'mse': {
                    'mean': units * val_err.pow(2).mean().item(),
                    'std': units * val_err.pow(2).std().item(),
                }
            },
            'lrs': lrs,
        }]
        wandb.log(dynamics[-1])

        print(f'[{epoch}] Target: {args.target:02d}, MAE TRAIN: {units * train_err.abs().mean():.5f} ± {units * train_err.abs().std():.5f}, MAE VAL: {units * val_err.abs().mean():.5f} ± {units * val_err.abs().std():.5f}', flush=True)

        scheduler.step(val_err.pow(2).mean())

        yield {
            'args': args,
            'dynamics': dynamics,
            'state': {k: v.cpu() for k, v in model.state_dict().items()},
        }


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mul0", type=int, default=128)
    parser.add_argument("--mul1", type=int, default=12)
    parser.add_argument("--mul2", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--rad_gaussians", type=int, default=50)
    parser.add_argument("--rad_h", type=int, default=256)
    parser.add_argument("--rad_bottleneck", type=int, default=256)
    parser.add_argument("--rad_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr1", type=float, default=1.0)
    parser.add_argument("--w_exp", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=200)
    parser.add_argument("--opts", type=str, default="x")
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--mean", type=float, default=0)
    parser.add_argument("--std", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    wandb.login()
    wandb.init(project="qm9", config=args.__dict__)

    with open(args.output, 'wb') as handle:
        pickle.dump(args, handle)

    for data in execute(args):
        data['git'] = git
        with open(args.output, 'wb') as handle:
            pickle.dump(args, handle)
            pickle.dump(data, handle)


if __name__ == "__main__":
    main()
