import torch
from torch.utils.data import DataLoader
from lm_eval import tasks, evaluator
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
        dict: A dictionary with evaluation