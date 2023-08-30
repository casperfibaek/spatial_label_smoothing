import os

if __name__ == "__main__":
    loss_method          = ['half', 'max', None]
    loss_strenght        = [1, 1.01]
    loss_kernel_radius   = [1, 2, 3]
    loss_kernel_circular = [True, False]
    loss_kernel_sigma    = [0.5, 1.0, 2.0]



    for lm in loss_method:
        for ls in loss_strenght:
            for lkr in loss_kernel_radius:
                for lkc in loss_kernel_circular:
                    for lks in loss_kernel_sigma:
                        os.system(f'python train.py --loss_method {lm} --loss_strenght {ls} --loss_kernel_radius {lkr} --loss_kernel_circular {lkc} --loss_kernel_sigma {lks}')


