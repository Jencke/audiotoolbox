import audiotoolbox as audio
import numpy as np


# fc = 500
# # bw = 24.7 * (4.37 * (fc / 1000.0) + 1)
# bw = 24.7 + fc / 9.265
# n_erb = (1000.0 / (24.7 * 4.37)) * np.log(4.37 * fc / 1000 + 1)
# n_erb += 1
# fc = (1 / 0.00437) * (np.exp((n_erb * 24.7 * 4.37) / 1000) - 1)
# bw2 = 24.7 + fc / 9.265
# print(bw, n_erb, freq)

# fc_min = 500 - bw/2
# fc_max = 500 + bw/2

# fc2_min = fc - bw2/2


# fc_khz = fc / 1000
# bw = 6.23 * (fc_khz)**2 + 93.39 * (fc_khz) + 28.52
# n_erb = 11.17 * (np.log((fc_khz + 0.312) / (fc_khz + 14.675))
#                  - np.log(0.312 / 14.675))
# freq = (14675 * (1 - np.exp(0.0895255 * n_erb))
#         / (np.exp(0.0895255 * n_erb) - 47.0353))
# print(bw, n_erb, freq)
