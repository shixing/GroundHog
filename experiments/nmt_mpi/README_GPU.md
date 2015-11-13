To run the code on multi-gpu with mpi:
1. you need to use cuda7.5 (IMPORTANT)
2. CNMeM disabled (this is default setting)
3. Have different base_compiledir for different process.

NOTE: to observe speedup, you need to feed GPU enough minibatches (more than 40). The first 40 minibatches are usually 2x or 3x slow than later batches.

