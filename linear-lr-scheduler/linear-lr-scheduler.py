def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    #Phase 1: Warmup phase (step < warmup_steps)
    if step < warmup_steps:
        return float(step * initial_lr / warmup_steps)

    #Phase 2: Post-training phase (step >= total_steps)
    elif step >= total_steps:
        return float(final_lr)

    #Phase 2: Linear decay phase (warmup_steps <= step < total_steps)
    else:
        decay_steps = total_steps - warmup_steps
        current_delay_step = total_steps - step
        return float(final_lr + (initial_lr - final_lr) * current_delay_step / decay_steps)