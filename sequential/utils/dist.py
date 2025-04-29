import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_rank_and_world_size():
    usage = '''
    Usage:
        rank, world_size = get_rank_and_world_size()
        Get rank and world size from environment variables in Three modes:
            1. Single XPU mode
                `WORLD_SIZE` is not set in environment variables
            2. Multi XPU mode with RANK in environment variables
                `RANK` and `WORLD_SIZE` are set in environment variables
            3. Multi XPU mode with LOCAL_RANK and NODE_RANK in environment variables
                `LOCAL_RANK` and `NODE_RANK` and `WORLD_SIZE` are set in environment variables
        Return:
            rank: int
            world_size: int
    '''

    # Single XPU mode
    if os.environ.get("WORLD_SIZE") is None:
        logger.warning("WORLD_SIZE is not set")
        return None, None

    # Multi XPU mode with RANK in environment variables
    if os.environ.get("RANK"):
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        return rank, world_size

    # Multi XPU mode with LOCAL_RANK and NODE_RANK in environment variables
    if os.environ.get("LOCAL_RANK") and os.environ.get("NODE_RANK") \
        and os.environ.get("NUM_NODES"):
        num_node = int(os.environ.get("NUM_NODES"))
        node_rank = int(os.environ.get("NODE_RANK"))
        local_rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))

        rank = node_rank * num_node + local_rank
        return rank, world_size

    raise ValueError("Failed to get rank and world size.\n" + usage)