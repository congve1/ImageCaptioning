import datetime
import logging
import time

import torch
from torch import nn
from torch import distributed as dist

from image_captioning.utils.metric_logger import MetricLogger
from image_captioning.modeling.utils import clip_gradients
from image_captioning.layers.cross_entropy_loss import LanguageModelCriterion
from image_captioning.layers.rl_loss import RewardCriterion
from image_captioning.utils.rewards import get_self_critical_reward
from image_captioning.utils.comm import get_world_size, is_main_process, get_rank, synchronize

from .inference import inference


def reduce_loss(loss):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            loss /= world_size
    return loss


def do_train(
        model,
        train_data_loader,
        val_data_loader,
        optimizer,
        scheduler,
        checkpointer,
        vocab,
        arguments
):
    device = arguments.pop('device')
    checkpoint_period = arguments.pop('checkpoint_period')
    log_period = arguments.pop('log_period')
    val_period = arguments.pop('val_period')
    beam_size = arguments.pop('beam_size')
    scst_iter = arguments.pop('scst_iter')
    seq_per_img = arguments.pop('seq_per_img')
    grad_clip = arguments.pop('grad_clip')
    metric_logger_name = arguments.pop('metric_logger_name')
    logger = logging.getLogger('image_captioning.trainer')
    logger.info("Start training")
    if is_main_process():
        meters = MetricLogger(
            delimiter='  ', log_period=log_period, name=metric_logger_name
        )
    distributed = get_world_size() > 1
    max_iter = len(train_data_loader)
    start_iter = arguments['iteration']
    model.train()
    start_training_time = time.time()
    criterion = LanguageModelCriterion()
    rl_criterion = RewardCriterion()
    best_cider_score = arguments['best_cider_score']
    end = time.time()
    for iteration, data in enumerate(train_data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments['iteration'] = iteration

        scheduler.step()

        att_features = data['att_features'].to(device)
        fc_features = data['fc_features'].to(device)
        cap_lens = data['cap_lens'].to(device)
        captions = data['captions'].to(device)
        if scst_iter == -1 or iteration <= scst_iter:
            outputs, *_ = model(fc_features, att_features, captions)
            loss = criterion(outputs, captions[:, 1:], cap_lens+1)
        else:
            sample_model = model.module if distributed else model
            sample_seqs,\
            sample_seq_log_probs,\
            *_ = sample_model.sample(
                fc_features, att_features
            )
            with torch.no_grad():
                greed_seqs,\
                greed_seqs_log_probs,\
                *_ = sample_model.greedy_search(
                    fc_features, att_features
                )
                rewards = get_self_critical_reward(
                    sample_seqs, greed_seqs, data['all_captions'],
                    seq_per_img, vocab
                ).to(device)
            loss = rl_criterion(
                sample_seq_log_probs, rewards, sample_seqs, vocab
            )
        loss_reduced = reduce_loss(loss)
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(optimizer, grad_clip)
        optimizer.step()

        batch_time = time.time() - end
        if is_main_process():
            meters.update(loss=loss_reduced)
            meters.update(batch_time=batch_time, data=data_time)
            eta_seconds = meters.batch_time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            current_lr = optimizer.param_groups[0]['lr']
            if iteration % log_period == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "rank:{rank}",
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        rank=get_rank(),
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=current_lr,
                    )
                )
            meters.add_scalar('loss', loss.item(), iteration)
            meters.add_scalar('lr', current_lr, iteration)
        # save model and do evaluation
        if iteration % checkpoint_period == 0:
            arguments['save_last_checkpoint'] = True
            checkpointer.save('model_{:07d}'.format(iteration), **arguments)
        # validate and save model(now do not validate model during training)
        if iteration % val_period == 0:
            val_model = model.module if distributed else model
            val_criterion = LanguageModelCriterion()
            val_loss, predictions, scores = inference(
                val_model, val_criterion, val_data_loader,
                vocab, beam_size, device
            )
            if is_main_process():
                logger.info("validation loss:{:.4f}".format(val_loss))
                meters.add_scalar('val_loss', val_loss, iteration)
                for metric, score in scores.items():
                    logger.info("{}: {:.5f}".format(metric, score))
                    meters.add_scalar("metric/"+metric, score, iteration)
                if scores['CIDEr'] > best_cider_score:
                    best_cider_score = scores['CIDEr']
                    arguments['best_cider_score'] = best_cider_score
                    logger.info("best cider score: {:.4f}".format(best_cider_score))
                    checkpointer.save(
                        'model_best', save_last_checkpoint=False, best_cider_score=best_cider_score
                    )
            model.train()
            synchronize()

        if iteration == max_iter:
            checkpointer.save('model_final', **arguments)
        end = time.time()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iter
        )
    )
