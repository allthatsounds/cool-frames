function [sig, p] = make_test_params()
%MAKE_TEST_PARAMS  Shared fixtures for the LTFAT filterbank unit test suite.
%
%   [SIG, P] = MAKE_TEST_PARAMS() returns:
%     SIG  - struct of test signals (fields described below)
%     P    - struct of scalar parameters (fs, Ls, tol, abs_tol)
%
%   Signals
%   -------
%     noise_mono    [Ls x 1]  White Gaussian noise, single channel
%     noise_stereo  [Ls x 2]  White Gaussian noise, two channels
%     sine_440      [Ls x 1]  Pure 440 Hz sinusoid
%     sine_1k       [Ls x 1]  Pure 1000 Hz sinusoid
%     impulse       [Ls x 1]  Unit impulse at sample 1
%     multi_tone    [Ls x 1]  440 Hz + 1 kHz + noise mixture
%
%   Parameters
%   ----------
%     fs       8000 Hz  (low sample rate keeps filter banks small/fast)
%     Ls       1024     signal length in samples
%     tol      1e-8     relative error tolerance for reconstruction tests
%     abs_tol  1e-12    absolute tolerance for exact-arithmetic checks

rng(42);   % fixed seed — all derived quantities are deterministic

p.fs      = 8000;
p.Ls      = 1024;
p.tol     = 1e-8;
p.abs_tol = 1e-12;

t = (0 : p.Ls - 1)' / p.fs;

sig.noise_mono   = randn(p.Ls, 1);
sig.noise_stereo = randn(p.Ls, 2);
sig.sine_440     = sin(2*pi * 440  * t);
sig.sine_1k      = sin(2*pi * 1000 * t);
sig.impulse      = [1; zeros(p.Ls - 1, 1)];
sig.multi_tone   = sig.sine_440 + 0.5*sig.sine_1k + 0.1*sig.noise_mono;

end
