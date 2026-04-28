import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run, get_resume_file,load_model
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.msfin import MSFIN
from test import test_main, evaluate



def train(epoch, model, loader, optimizer, args=None):
    model.train()
    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']
    label = torch.arange(args.way).repeat(args.query).cuda() 
    label = label.type(torch.LongTensor)
    label = label.cuda()

    loss_meter = Meter()
    acc_meter = Meter()
    k = args.way * args.shot

    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):
        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        model.module.mode = 'encoder'

        data1,data2 = model(data)
        _,data_aux = model(data_aux)
        data_shot1, data_query1 = data1[:k], data1[k:]
        data_shot2, data_query2 = data2[:k], data2[k:]

        if args.shot > 1:
            data_shot1 = data_shot1.view(args.shot, args.way, *data_shot1.shape[1:])
            data_shot2 = data_shot2.view(args.shot, args.way, *data_shot2.shape[1:])
            data_shot1 = data_shot1.mean(dim=0)
            data_shot2 = data_shot2.mean(dim=0)

        model.module.mode = 'local_feat1'
        local_shot1 = model(data_shot1)
        local_query1 = model(data_query1)                   #([20, 320, 10, 10])


        model.module.mode = 'local_feat2'
        local_shot2 = model(data_shot2)
        local_query2 = model(data_query2)                  # [20, 640, 5, 5])

        model.module.mode = 'msfn1'
        glo_shot1= model(data_shot1)
        glo_query1 = model(data_query1)
        sele_local_shot1 = model(local_shot1)
        sele_local_query1 = model(local_query1)
        sele_local_shot1, sele_local_query1 = local_shape(sele_local_shot1, sele_local_query1)

        model.module.mode = 'msfn2'
        glo_shot2= model(data_shot2)
        glo_query2 = model(data_query2)
        sele_local_shot2 = model(local_shot2)
        sele_local_query2 = model(local_query2)
        sele_local_shot2, sele_local_query2 = local_shape(sele_local_shot2, sele_local_query2)


        model.module.mode = 'mfin'

        all_shot = torch.cat((glo_shot1,glo_shot2,sele_local_shot1,sele_local_shot2),dim = -1)
        all_qry = torch.cat((glo_query1,glo_query2,sele_local_query1,sele_local_query2),dim = -1)

        logits, absolute_logits= model((all_shot, all_qry))
        epi_loss = F.cross_entropy(logits, label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss
        loss = args.lamb * epi_loss + loss_aux

        acc = compute_accuracy(logits, label)
        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        #detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()


    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()

def local_shape(sele_local_shot,sele_local_query):
    _, c, n = sele_local_shot.shape
    sele_local_shot = sele_local_shot.reshape(args.way, -1, c, n)
    sele_local_shot = sele_local_shot.permute(0, 2, 1, 3).reshape(args.way, c, -1)

    sele_local_query = sele_local_query.reshape(args.way * args.query, -1, c, n)
    sele_local_query = sele_local_query.permute(0, 2, 1, 3).reshape(args.way * args.query, c, -1)

    return sele_local_shot,sele_local_query


def train_main(args):

    start_epoch =args.start_epoch
    stop_epoch = args.max_epoch
    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    Dataset = dataset_builder(args)
    trainset = Dataset('train', args)

    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    model = MSFIN(args).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print('Number  of parameter: % .2fM' % (total / 1e6))
    model = nn.DataParallel(model, device_ids=args.device_ids)

    if args.resume:
        resume_file = get_resume_file(args.save_path)
        print("resume_file:",resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['params'])

    if not args.no_wandb:
        wandb.watch(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # Track top-K checkpoints by val accuracy, so the very best (not the very first lucky one) is used.
    top_k = 3
    top_ckpts = []  # list of (val_acc, epoch, path)

    for epoch in range(start_epoch, args.max_epoch + 1):
        start_time = time.time()
        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)

        lr_scheduler.step()
        val_loss, val_acc, _ = evaluate(epoch, model ,val_loader, args, set='val')

        if not args.no_wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)

        if val_acc > max_acc :
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            outfile = os.path.join(args.save_path, 'max_acc.pth')
            torch.save({'epoch':epoch, 'params':model.state_dict()}, outfile)

        # Always-on top-K tracker (independent of args.save_freq) to retain the best candidates.
        if len(top_ckpts) < top_k or val_acc > min(c[0] for c in top_ckpts):
            ckpt_path = os.path.join(args.save_path, f'topk_epoch{epoch}.pth')
            torch.save({'epoch': epoch, 'params': model.state_dict(), 'val_acc': float(val_acc)}, ckpt_path)
            top_ckpts.append((float(val_acc), epoch, ckpt_path))
            top_ckpts.sort(key=lambda x: -x[0])
            # Drop older worse checkpoints
            while len(top_ckpts) > top_k:
                _, _, drop_path = top_ckpts.pop()
                if os.path.exists(drop_path):
                    try:
                        os.remove(drop_path)
                    except OSError:
                        pass

        if (epoch % args.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(args.save_path, '{:d}.pth'.format(epoch))
            torch.save({'epoch':epoch, 'params':model.state_dict()}, outfile)

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

    # Persist the list of top-K checkpoints for later use by test stage.
    args.top_ckpts = top_ckpts
    print('[ log ] Top-K checkpoints by val_acc:')
    for v, e, p in top_ckpts:
        print(f'    epoch {e:>3d}  val_acc={v:.3f}  {p}')

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    set_seed(args.seed)
    model = train_main(args)
    test_acc, test_ci = test_main(model, args)
    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})
