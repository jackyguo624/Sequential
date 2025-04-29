import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)

def get_rank_and_world_size():
    if os.environ.get("WORLD_SIZE") is None:
        logger.warning("WORLD_SIZE is not set")
        return None, None
    if os.environ.get("RANK"):
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        return rank, world_size

    assert os.environ.get("LOCAL_RANK") and os.environ.get("NODE_RANK") \
        and os.environ.get("NUM_NODES")

    num_node = int(os.environ.get("NUM_NODES"))
    node_rank = int(os.environ.get("NODE_RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))

    rank = node_rank * num_node + local_rank
    return rank, world_size
