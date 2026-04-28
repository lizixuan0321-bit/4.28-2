import os
import glob
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by, set_seed
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.msfin import MSFIN


def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()
    loss_meter = Meter()
    acc_meter = Meter()
    label = torch.arange(args.way).repeat(args.query).cuda()
    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            model.module.mode = 'encoder'
            data1, data2 = model(data)
            data_shot1, data_query1 = data1[:k], data1[k:]
            data_shot2, data_query2 = data2[:k], data2[k:]
            model.module.mode = 'local_feat1'
            local_shot1 = model(data_shot1)
            local_query1 = model(data_query1)
            model.module.mode = 'local_feat2'
            local_shot2 = model(data_shot2)
            local_query2 = model(data_query2)
            model.module.mode = 'msfn1'
            glo_shot1 = model(data_shot1)
            glo_query1 = model(data_query1)
            sele_local_shot1 = model(local_shot1)
            sele_local_query1 = model(local_query1)
            _, c, n = sele_local_shot1.shape
            sele_local_shot1 = sele_local_shot1.reshape(args.way, -1, c, n).permute(0,2,1,3).reshape(args.way, c, -1)
            sele_local_query1 = sele_local_query1.reshape(args.way*args.query, -1, c, n).permute(0,2,1,3).reshape(args.way*args.query, c, -1)
            model.module.mode = 'msfn2'
            glo_shot2 = model(data_shot2)
            glo_query2 = model(data_query2)
            sele_local_shot2 = model(local_shot2)
            sele_local_query2 = model(local_query2)
            _, c, n = sele_local_shot2.shape
            sele_local_shot2 = sele_local_shot2.reshape(args.way, -1, c, n).permute(0,2,1,3).reshape(args.way, c, -1)
            sele_local_query2 = sele_local_query2.reshape(args.way*args.query, -1, c, n).permute(0,2,1,3).reshape(args.way*args.query, c, -1)
            all_shot = torch.cat((glo_shot1, glo_shot2, sele_local_shot1, sele_local_shot2), dim=-1)
            all_qry = torch.cat((glo_query1, glo_query2, sele_local_query1, sele_local_query2), dim=-1)
            model.module.mode = 'mfin'
            logits = model((all_shot, all_qry))
            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)
            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')
    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def _build_test_loader(args, seed=None):
    """Build a fresh test loader. Setting seed makes the episode sampling reproducible."""
    if seed is not None:
        set_seed(seed)
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    return DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)


def _build_val_loader(args, n_episodes=1200, seed=12345):
    """Build a fresh, large, seeded validation loader for robust checkpoint re-ranking."""
    set_seed(seed)
    Dataset = dataset_builder(args)
    valset = Dataset('val', args)
    sampler = CategoriesSampler(valset.label, n_episodes, args.way, args.shot + args.query)
    loader = DataLoader(valset, batch_sampler=sampler, num_workers=8, pin_memory=True)
    return [b for b in loader]  # materialize -> identical episodes per checkpoint


def _select_best_checkpoint(model, args):
    """
    Re-evaluate every saved checkpoint candidate on a *large* fixed validation set,
    then pick the one with highest val accuracy.  This corrects the high-variance
    selection of `max_acc.pth` that the original 200-episode online val would make.
    """
    candidates = []

    # 1) Always include the train-time max_acc checkpoint (legacy)
    ma = os.path.join(args.save_path, 'max_acc.pth')
    if os.path.isfile(ma):
        candidates.append(('max_acc', ma))

    # 2) Include top-K checkpoints saved during training (primary source)
    topk_paths = sorted(glob.glob(os.path.join(args.save_path, 'topk_epoch*.pth')))
    for p in topk_paths:
        candidates.append((os.path.basename(p), p))

    # 3) Fallback: only include periodic [0-9]*.pth when top-K tracking is
    #    unavailable (e.g. resumed training, older runs without top-K saving,
    #    or accidental file deletion).  When top-K exists it already captures
    #    the best epochs by val accuracy, so re-ranking the rarely-competitive
    #    early-epoch periodic checkpoints just wastes time.
    if not topk_paths:
        for p in sorted(glob.glob(os.path.join(args.save_path, '[0-9]*.pth'))):
            candidates.append((os.path.basename(p), p))

    # de-dup paths
    seen, uniq = set(), []
    for n, p in candidates:
        if p in seen:
            continue
        seen.add(p)
        uniq.append((n, p))

    if not uniq:
        return None, None  # nothing to load

    if len(uniq) == 1:
        return uniq[0][1], None

    print(f'[ log ] re-ranking {len(uniq)} candidate checkpoints on a large fixed val set ...')
    val_loader = _build_val_loader(args)

    best_path, best_acc = None, -1.0
    rankings = []
    for name, path in uniq:
        load_model(model, path)
        _, val_acc, _ = evaluate(name, model, val_loader, args, set='val')
        rankings.append((float(val_acc), name, path))
        if val_acc > best_acc:
            best_acc = float(val_acc)
            best_path = path
    rankings.sort(key=lambda x: -x[0])
    print('[ log ] checkpoint re-ranking results:')
    for v, n, _ in rankings[:8]:
        print(f'    val_acc={v:.3f}  {n}')
    return best_path, best_acc


def test_main(model, args):

    # 1) Pick the best checkpoint via large-val re-ranking (more reliable than 200-ep online val).
    best_path, best_val = _select_best_checkpoint(model, args)
    if best_path is None:
        best_path = os.path.join(args.save_path, 'max_acc.pth')
    print(f'[ log ] using checkpoint: {best_path}' + (f'  (val_acc={best_val:.3f})' if best_val is not None else ''))
    model = load_model(model, best_path)

    # 2) Fix the test sampler seed for reproducibility (avoids the user seeing different
    #    numbers between runs of the same checkpoint).
    test_loader = _build_test_loader(args, seed=getattr(args, 'test_seed', 2024))

    # 3) Evaluate
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = MSFIN(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
