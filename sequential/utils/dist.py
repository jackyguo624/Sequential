import os
import logging

logger = logging.getLogger(__name__)
logger.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

def get_rank_and_world_size():
    if os.environ.get("WORLD_SIZE") is None:
        logger.warning("WORLD_SIZE is not set")
        return None, None

    num_node = int(os.environ.get("NODE_RANK", 1))
    rank = int(os.environ.get("NODE_RANK")) * num_node + \
        int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    return rank, world_size
