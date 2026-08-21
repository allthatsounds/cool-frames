function battery = make_signal_battery(varargin)
% MAKE_SIGNAL_BATTERY  Generate the shared test signal battery used by all layers.
%
%   battery = make_signal_battery()
%   battery = make_signal_battery('save', true)
%
%   Generates all test signals with a fixed seed (rng(42)) for reproducibility.
%   The returned struct is also what is saved to signal_battery.mat.
%
%   Python equivalent: scipy.io.loadmat('signal_battery.mat', squeeze_me=True)
%
%   OUTPUTS
%     battery.signals   struct of named signal vectors / matrices
%     battery.params    struct of shared parameters (fs, Ls, tol, ...)
%
%   OPTIONAL PARAMETERS
%     'save'  (logical, default false)  save battery to signals/signal_battery.mat
%
%   See also: save_reference, compare_to_reference

    p = inputParser();
    addParameter(p, 'save', false);
    parse(p, varargin{:});

    % ── Fixed parameters ───────────────────────────────────────────────────────
    fs  = 8000;
    Ls  = 1024;
    k0  = floor(Ls / 8);   % on-bin sinusoid bin index
    n   = (0 : Ls-1)';

    % ── Fixed seed ────────────────────────────────────────────────────────────
    rng(42);

    % ── 1. Minimal sanity signals ──────────────────────────────────────────────
    imp      = zeros(Ls, 1);  imp(1)                  = 1;
    imp_shift= zeros(Ls, 1);  imp_shift(floor(Ls/4)+1)= 1;

    battery.signals.zero            = zeros(Ls, 1);
    battery.signals.impulse         = imp;
    battery.signals.impulse_shifted = imp_shift;
    battery.signals.dc              = ones(Ls, 1);

    % ── 2. Spectral edge signals ───────────────────────────────────────────────
    battery.signals.tone_onbin   = exp(2*pi*1i * k0 * n / Ls);
    battery.signals.tone_offbin  = exp(2*pi*1i * (k0 + 0.5) * n / Ls);
    battery.signals.tone_nyquist = (-1).^n;

    % ── 3. Noise ───────────────────────────────────────────────────────────────
    battery.signals.noise_real    = randn(Ls, 1);
    battery.signals.noise_complex = randn(Ls, 1) + 1i * randn(Ls, 1);
    battery.signals.noise_stereo  = randn(Ls, 2);   % two-channel (L x W)

    % ── 4. Chirp ───────────────────────────────────────────────────────────────
    t = n / fs;
    battery.signals.chirp = chirp(t, 0, t(end), fs/2);

    % ── 5. Pathological lengths (separate seeds to avoid correlation) ──────────
    rng(43);
    battery.signals.prime_length = randn(997, 1);   % N = prime

    battery.signals.short        = randn(16, 1);    % shorter than most kernels

    rng(44);
    battery.signals.long         = randn(8192, 1);  % stress memory / stride

    % ── Parameters ────────────────────────────────────────────────────────────
    battery.params.fs       = fs;
    battery.params.Ls       = Ls;
    battery.params.k0       = k0;
    battery.params.tol      = 1e-8;
    battery.params.abs_tol  = 1e-12;
    battery.params.seed     = 42;

    % ── Optional save ─────────────────────────────────────────────────────────
    if p.Results.save
        savedir  = fullfile(fileparts(mfilename('fullpath')), 'signals');
        savepath = fullfile(savedir, 'signal_battery.mat');
        if ~exist(savedir, 'dir')
            mkdir(savedir);
        end
        save(savepath, '-struct', 'battery', '-v7');
        fprintf('Signal battery saved: %s\n', savepath);
    end
end
