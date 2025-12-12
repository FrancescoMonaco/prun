import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import logging

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


def evaluate_model(
    model_name,
    model,
    tokenizer,
    task_list=[],
    use_accelerate=False,
    add_special_tokens=False,
):
    """
    Evaluate a given model on specified tasks using the lm_eval framework.
    Args:
        model_name (str): Name of the model.
        model (torch.nn.Module): The model to evaluate.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        task_list (list): List of task names to evaluate on.
        use_accelerate (bool): Whether to use HuggingFace Accelerate for distributed evaluation.
        add_special_tokens (bool): Whether to add special tokens to the tokenizer.
    Returns:
        dict: A dictionary with evaluation results.
    """
    log.info(f"Evaluating model {model_name} on tasks: {task_list}")

    # Ensure model is on the correct device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Check if model is already on device (if it's not sharded)
    try:
        if model.device.type != device:
            log.info(f"Moving model to {device}...")
            model = model.to(device)
    except AttributeError:
        # Model might be sharded or not have .device
        pass

    log.info(f"Model device: {device}")

    # Ensure tokenizer padding side is correct for decoder-only models
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    # Wrap the model and tokenizer
    hflm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)

    results = simple_evaluate(
        model=hflm_model,
        tasks=task_list,
        batch_size=8,
        device=device,
    )
    return results
