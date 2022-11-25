from utils.dataset import HRTDataset
from utils.resp_lasso import resp_lasso
from tqdm import tqdm
import matlab
from matlab.engine import start_matlab
from hrt import HRT
import numpy as np


def benchmark_lasso(dataset, dmat, total=10000, num=49):
    bar = tqdm(total=total)
    mse_ = np.zeros((total,), dtype=np.float64)
    cos_ = np.zeros((total,), dtype=np.float64)
    avg_ = np.zeros((total,), dtype=np.float64)
    cnt = 0
    for data in dataset:
        xx_, yy_ = data
        sp = yy_['output_s']
        measure = yy_['output_m']
        t_raw = xx_['input_2']
        for t_, m_, s_ in zip(t_raw.numpy(), measure.numpy(), sp.numpy()):
            resp = resp_lasso(t_[:num, :], m_[:num], dmat)

            if cnt >= total:
                return mse_, cos_, avg_
            mse_[cnt] = np.mean(np.square(resp - s_))
            cos_[cnt] = np.dot(resp, s_) / (np.sqrt(np.dot(resp, resp)) + 1e-6) / (np.sqrt(np.dot(s_, s_)) + 1e-6)
            avg_[cnt] = np.mean(m_)
            cnt += 1
            bar.set_description('mse: {:.5f}, cos: {:.5f}'.format(np.mean(mse_[:cnt]), np.mean(cos_[:cnt])))
            bar.update()

    mse_ = np.mean(mse_)
    cos_ = np.mean(cos_)
    avg_ = np.mean(avg_)
    print('SNR: {:.4f}'.format(10 * np.log(avg_ / noise_std)))
    print('MSE: {:.4f}'.format(mse_))
    print('Cos: {:.4f}'.format(cos_))
    print('PSNR: {:.4f}'.format(10. * np.log10(1. / mse_)))
    print('SAM: {:.4f}'.format(np.arccos(cos_)))

    return mse_, cos_, avg_


def benchmark_dl(dataset, model, total=10000):
    bar = tqdm(total=total)
    mse_ = np.zeros((total,), dtype=np.float64)
    cos_ = np.zeros((total,), dtype=np.float64)
    avg_ = np.zeros((total,), dtype=np.float64)
    cnt = 0
    for data in dataset:
        xx_, yy_ = data
        sp = yy_['output_s']
        measure = yy_['output_m']
        resp = model(xx_)[0]
        for s_, r_, m_ in zip(sp.numpy(), resp.numpy(), measure.numpy()):

            if cnt >= total:
                return mse_, cos_, avg_
            mse_[cnt] = np.mean(np.square(r_ - s_))
            cos_[cnt] = np.dot(r_, s_) / (np.sqrt(np.dot(r_, r_)) + 1e-6) / (np.sqrt(np.dot(s_, s_)) + 1e-6)
            avg_[cnt] = np.mean(m_)
            cnt += 1
            bar.set_description('mse: {:.5f}, cos: {:.5f}'.format(np.mean(mse_[:cnt]), np.mean(cos_[:cnt])))
            bar.update()

    mse_ = np.mean(mse_)
    cos_ = np.mean(cos_)
    avg_ = np.mean(avg_)
    print('SNR: {:.4f}'.format(10 * np.log(avg_ / noise_std)))
    print('MSE: {:.4f}'.format(mse_))
    print('Cos: {:.4f}'.format(cos_))
    print('PSNR: {:.4f}'.format(10. * np.log10(1. / mse_)))
    print('SAM: {:.4f}'.format(np.arccos(cos_)))

    return mse_, cos_, avg_

def benchmark_tval3(dataset, total=10000, num=49):
    bar = tqdm(total=total)
    mse_ = np.zeros((total,), dtype=np.float64)
    cos_ = np.zeros((total,), dtype=np.float64)
    avg_ = np.zeros((total,), dtype=np.float64)
    cnt = 0

    mlab = start_matlab()
    mlab.addpath('./TVAL3', './TVAL3/Utilities', './TVAL3/Solver')

    for data in dataset:
        xx_, yy_ = data
        sp = yy_['output_s']
        measure = yy_['output_m']
        t_raw = xx_['input_2']
        for t_, m_, s_ in zip(t_raw.numpy(), measure.numpy(), sp.numpy()):

            mat_t_ = matlab.double(t_[:num, :].tolist())
            mat_m_ = matlab.double(m_[:num].reshape(-1, 1).tolist())
            resp = mlab.tval3_solve(mat_t_, mat_m_)
            resp = np.array(resp).astype(np.float32).squeeze()

            if cnt >= total:
                return mse_, cos_, avg_
            mse_[cnt] = np.mean(np.square(resp - s_))
            cos_[cnt] = np.dot(resp, s_) / (np.sqrt(np.dot(resp, resp)) + 1e-6) / (np.sqrt(np.dot(s_, s_)) + 1e-6)
            avg_[cnt] = np.mean(m_)
            cnt += 1
            bar.set_description('mse: {:.5f}, cos: {:.5f}'.format(np.mean(mse_[:cnt]), np.mean(cos_[:cnt])))
            bar.update()

    mlab.quit()
    mse_ = np.mean(mse_)
    cos_ = np.mean(cos_)
    avg_ = np.mean(avg_)
    print('SNR: {:.4f}'.format(10 * np.log(avg_ / noise_std)))
    print('MSE: {:.4f}'.format(mse_))
    print('Cos: {:.4f}'.format(cos_))
    print('PSNR: {:.4f}'.format(10. * np.log10(1. / mse_)))
    print('SAM: {:.4f}'.format(np.arccos(cos_)))

    return mse_, cos_, avg_

def benchmark_twist(dataset, total=10000, num=49):
    bar = tqdm(total=total)
    mse_ = np.zeros((total,), dtype=np.float64)
    cos_ = np.zeros((total,), dtype=np.float64)
    avg_ = np.zeros((total,), dtype=np.float64)
    cnt = 0

    mlab = start_matlab()
    mlab.addpath('./TwIST')

    for data in dataset:
        xx_, yy_ = data
        sp = yy_['output_s']
        measure = yy_['output_m']
        t_raw = xx_['input_2']
        for t_, m_, s_ in zip(t_raw.numpy(), measure.numpy(), sp.numpy()):

            mat_t_ = matlab.double(t_[:num, :].tolist())
            mat_m_ = matlab.double(m_[:num].reshape(-1, 1).tolist())
            resp = mlab.twist_solve_tv(mat_t_, mat_m_)
            resp = np.array(resp).astype(np.float32).squeeze()

            if cnt >= total:
                return mse_, cos_, avg_
            mse_[cnt] = np.mean(np.square(resp - s_))
            cos_[cnt] = np.dot(resp, s_) / (np.sqrt(np.dot(resp, resp)) + 1e-6) / (np.sqrt(np.dot(s_, s_)) + 1e-6)
            avg_[cnt] = np.mean(m_)
            cnt += 1
            bar.set_description('mse: {:.5f}, cos: {:.5f}'.format(np.mean(mse_[:cnt]), np.mean(cos_[:cnt])))
            bar.update()

    mlab.quit()
    mse_ = np.mean(mse_)
    cos_ = np.mean(cos_)
    avg_ = np.mean(avg_)
    print('SNR: {:.4f}'.format(10 * np.log(avg_ / noise_std)))
    print('MSE: {:.4f}'.format(mse_))
    print('Cos: {:.4f}'.format(cos_))
    print('PSNR: {:.4f}'.format(10. * np.log10(1. / mse_)))
    print('SAM: {:.4f}'.format(np.arccos(cos_)))
    return mse_, cos_, avg_


if __name__ == '__main__':
    noise_std = 0.01
    spnum = 49

    test_dataset = HRTDataset(
        spmat_path='./srcs/specs_icvl_d301_test_14k.tfrecords',
        tmat_path='./srcs/tmat_testing_10k_d301.tfrecords',
        batch_size=64,
        masks_path='srcs/masks_n49_full_100k.tfrecords',
        spnum=spnum,
        min_spnum=None,
        noise_stddev=noise_std,
        sp_amp_range=[0.2, 1.0],
        dict_outputs=True,
        output_measure=True,
        shuffle=False,
    )
    samples = test_dataset.generate_dataset()
    network = HRT(
        input_t_shape=(spnum, 301),
    ).model
    network.load_weights('./checkpoints/checkpoint-hrt-49fixed.h5')

    # benchmark_dl(samples, network)
    benchmark_twist(samples)
    mse, cos, avg = benchmark_twist(samples)


