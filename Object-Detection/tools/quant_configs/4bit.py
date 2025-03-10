
class Config:
    def __init__(self):
        # calibration settings
        self.optim_size = 256
        self.calib_size = 1
        self.optim_batch_size = 1
        self.calib_batch_size = 1
        self.w_bit = 4
        self.a_bit = 4
        self.calib_metric = 'mse'
        self.matmul_head_channel_wise = True
        self.eq_n = 128
        self.search_round = 3
        # reconstruction settings
        self.reconstruct_mlp = True
        self.optim_metric = 'hessian_perturb'
        self.optim_mode = 'qdrop'
        self.pct = 0.999
        self.drop_prob = 0.5
        self.quant_ratio = 0.5
        self.keep_gpu = False
