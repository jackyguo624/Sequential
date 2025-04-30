import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)

def get_dist_env():
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")
    node_rank = os.environ.get("NODE_RANK")
    num_nodes = os.environ.get("NUM_NODES")
    return f"RANK: {rank}, WORLD_SIZE: {world_size}, LOCAL_RANK: {local_rank}, NODE_RANK: {node_rank}, NUM_NODES: {num_nodes}"

def get_rank_and_world_size():
    usage = '''
    Usage:
        Get rank and world_size from environment variables in three modes:
            1. Single XPU mode
                `RANK` is not set in environment variables
            2. Multi XPU mode with RANK in environment variables
                `RANK` and `WORLD_SIZE` are set in environment variables
            3. Multi XPU mode with LOCAL_RANK and NODE_RANK in environment variables
                `LOCAL_RANK` and `NODE_RANK` and `WORLD_SIZE` are set in environment variables
        Return:
            rank: int
            world_size: int
    '''

    # Single XPU mode
    if os.environ.get("RANK") is None:
        logger.warning("RANK is not set")
        return None, None

    # Multi XPU mode with RANK in environment variables
    if os.environ.get("RANK") and os.environ.get("WORLD_SIZE"):
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

    raise ValueError("Failed to get rank and world size.\n" + usage + "\n" 
                     + get_dist_env())