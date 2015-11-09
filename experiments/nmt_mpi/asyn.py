
import logging
import time
import train_mpi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format="%(asctime)s: "+str('0')+" %(name)s: %(levelname)s: %(message)s")


train_mpi.compile('1')

