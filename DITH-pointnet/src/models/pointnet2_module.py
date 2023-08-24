from src.models.pointnet_module import PointNetModule
from src.models.components.pointnet2 import PointNet2SemSegMsg


class PointNet2Module(PointNetModule):    
    def __init__(
        self,
        data_dir: str = '',
        num_class: int = 2,
        lr: float = 0.001,
    ):
        super().__init__()
        
        self.model = PointNet2SemSegMsg(hparams=self.hparams)