function export_reference_data(outDir, maxStep)
%EXPORT_REFERENCE_DATA  Generate Python-readable reference .mat files for the
%   filterbank phase-retrieval and reassignment Python port.
%
%   export_reference_data()
%   export_reference_data(outDir)
%   export_reference_data(outDir, maxStep)
%
%   Runs the MATLAB pipeline once with a fixed RNG seed and writes one .mat
%   file per algorithmic group into <outDir> (default:
%   tests/reference_data/ relative to this script).
%
%   maxStep (optional, default Inf): stop after this many pipeline steps.
%   Use maxStep=5 to export only params through neighbors (no mex-heavy
%   phase-retrieval steps that may crash on some platforms).
%
%   LOADING IN PYTHON
%   -----------------
%       import scipy.io, numpy as np
%       d = scipy.io.loadmat('params.mat', squeeze_me=True)
%
%   Cell arrays of unequal-length subbands are saved as flat (Nsum,)
%   concatenated arrays.  Use
%       np.split(flat, np.cumsum(N)[:-1])
%   to recover the per-channel list of arrays.
%
%   INDEX CONVENTIONS
%   -----------------
%   NEIGH  is saved in the convention that the C mex functions expect and
%   that the Python port uses:
%     - shape  (Nsum, 6)  [Python/C row-of-neighbours ordering]
%     - dtype  int32
%     - 0-based flat indices into the coefficient vector
%     - -1 means "no neighbour in this direction"
%   This is the result of  (comp_filterbankneighbors output) - 1  transposed.
%   filterbankconstphase.m applies the same -1 before calling the mex
%   (see line "NEIGH = NEIGH-1").
%
%   posInfo  is saved as (Nsum, 2):
%     - column 0: channel index, 0-based
%     - column 1: time position in samples  (= frame_index * a_rat[m])
%
%   HOW TO RUN
%   ----------
%   From MATLAB, cd into the filterbank/ project root, then:
%
%       export_reference_data()
%
%   All mex binaries must already be compiled.  If not:
%
%       run('mex_fb/mexinit.m')
%
%   OUTPUT FILES
%   ------------
%   params.mat             signal, filterbank geometry
%   filters.mat            full-length transfer functions (L x M complex)
%   filterbank_coeff.mat   c / ch / cd flat complex arrays + N vector
%   phasegrad.mat          tgrad, fgrad, cs from complex coefficients
%   neighbors.mat          NEIGH (Nsum x 6, 0-based), posInfo (Nsum x 2)
%   phasegrad_frommag.mat  tgrad, fgrad from magnitude (phase-mag relation)
%   heapint.mat            filterbankheapint outputs (timeinv + relgrad)
%   constphase.mat         filterbankconstphase full-pipeline output
%   reassign.mat           filterbankreassign output
%   unif_heapint.mat       ufilterbankheapint reference (uniform-hop case)

% ── 0.  Paths ─────────────────────────────────────────────────────────────
thisDir = fileparts(mfilename('fullpath'));   % tests/
fbRoot  = fullfile(thisDir, '..');            % filterbank/

addpath(fullfile(thisDir, 'shared'));
setup_filterbank_paths();
addpath(fullfile(fbRoot, 'mex_fb'));

if nargin < 1 || isempty(outDir)
    outDir = fullfile(thisDir, 'reference_data');
end
if nargin < 2 || isempty(maxStep)
    maxStep = Inf;
end
if ~exist(outDir, 'dir'), mkdir(outDir); end
fprintf('Writing reference data to:\n  %s\n\n', outDir);

% ── 1.  Fixed parameters and test signal ─────────────────────────────────
fs = 8000;
Ls = 1024;
t  = (0 : Ls - 1)' / fs;
% Deterministic signal — no RNG, so MATLAB and Python produce identical inputs.
f  = sin(2*pi*440*t) + 0.5*sin(2*pi*1000*t) + 0.3*sin(2*pi*2500*t);

% ── 2.  ERB filterbank ────────────────────────────────────────────────────
fprintf('[1/9] Building ERB filterbank ...\n');
[g, a_mat, ~, L, info] = audfilters(fs, Ls);
% a_mat : M x 2 rational hop factors  [numerator, denominator]
% info.fc  : normalised center frequencies (1 = Nyquist)
% info.tfr : function handle, call info.tfr(L) to get TFR vector at length L

M     = numel(g);
fc_n  = info.fc;
tfr_v = info.tfr(L);   % M x 1 time-frequency ratios at system length L

% Handle both formats: a_mat can be M x 2 (rational) or M x 1 (integer hops)
if size(a_mat, 2) < 2
    a_mat = [a_mat(:,1), ones(size(a_mat,1), 1)];  % promote to M x 2 rational
end
a_num = int32(a_mat(:,1));
a_den = int32(a_mat(:,2));
a_rat = double(a_num) ./ double(a_den);  % M x 1 hop ratios

% Subband lengths
N = int32(ceil(L ./ a_rat));   % M x 1
Nsum = sum(double(N));

% ── 3.  params.mat ────────────────────────────────────────────────────────
fprintf('[1/9] Saving params.mat ...\n');
p.fs    = int32(fs);
p.Ls    = int32(Ls);
p.L     = int32(L);
p.M     = int32(M);
p.N     = N;           % M x 1 int32: subband lengths
p.Nsum  = int32(Nsum);
p.a_num = a_num;       % M x 1 int32: hop numerators
p.a_den = a_den;       % M x 1 int32: hop denominators
p.a_rat = a_rat;       % M x 1 double: a_num/a_den
p.fc_n  = fc_n;        % M x 1 double: normalised center freqs (1 = Nyquist)
p.tfr   = tfr_v;       % M x 1 double: time-frequency ratios at L
p.f     = f;           % Ls x 1 double: test signal
save(fullfile(outDir, 'params.mat'), '-struct', 'p', '-v7');

% ── 4.  filters.mat ───────────────────────────────────────────────────────
% Full-length L-point DFT of every filter.  Python can use these directly
% with an IFFT to recover the impulse response, or multiply by the signal
% DFT for analysis.  G_cols(:, m) is the transfer function of filter m.
fprintf('[2/9] Saving filters.mat ...\n');
G_cols = zeros(L, M);
for m = 1 : M
    G_cols(:, m) = comp_transferfunction(g{m}, L);
end
fl.G_cols = G_cols;   % L x M complex
save(fullfile(outDir, 'filters.mat'), '-struct', 'fl', '-v7');

% ── 5.  Filterbank analysis (c, ch, cd) ───────────────────────────────────
% Prepare the three versions of the filters needed for phase gradients:
%   g_prep : analysis filters (after precomputation for length L)
%   gh     : frequency-weighted version  (for group delay)
%   gd     : time-weighted version       (for instantaneous frequency)
fprintf('[3/9] Computing filterbank coefficients ...\n');
f_pad = postpad(f, L);
[g_prep, asan] = filterbankwin(g, a_mat, L, 'normal');
[gh, gd, g_prep] = comp_phasegradfilters(g_prep, asan, L);

c  = comp_filterbank(f_pad, g_prep, asan);
ch = comp_filterbank(f_pad, gh,     asan);
cd = comp_filterbank(f_pad, gd,     asan);

% Flatten: c{m} is N(m) x 1 complex; vertcat gives Nsum x 1
c_flat  = vertcat(c{:});
ch_flat = vertcat(ch{:});
cd_flat = vertcat(cd{:});

co.c_flat  = c_flat;   % Nsum x 1 complex128
co.ch_flat = ch_flat;  % Nsum x 1 complex128
co.cd_flat = cd_flat;  % Nsum x 1 complex128
save(fullfile(outDir, 'filterbank_coeff.mat'), '-struct', 'co', '-v7');

% ── 6.  phasegrad.mat ────────────────────────────────────────────────────
% Phase gradient from complex coefficients.
% cs{m} = |c{m}|^2  (spectrogram, NOT magnitude).
% tgrad: normalised instantaneous frequency (dimensionless, relative to bin)
% fgrad: negative group delay relative to frame position (in samples)
fprintf('[4/9] Computing phase gradient from complex coefficients ...\n');
minlvl = eps;
[tgrad_c, fgrad_c, cs_c] = comp_filterbankphasegrad(c, ch, cd, L, minlvl);

pg.tgrad_flat = vertcat(tgrad_c{:});   % Nsum x 1 double
pg.fgrad_flat = vertcat(fgrad_c{:});   % Nsum x 1 double
pg.cs_flat    = vertcat(cs_c{:});      % Nsum x 1 double  (spectrogram = |c|^2)
save(fullfile(outDir, 'phasegrad.mat'), '-struct', 'pg', '-v7');

% ── 7.  neighbors.mat ────────────────────────────────────────────────────
% Neighbour graph for the non-uniform filterbank heap integration.
%
% comp_filterbankneighbors returns NEIGH as (6, Nsum) with MATLAB 1-based
% indices and 0 meaning absent.  filterbankconstphase.m subtracts 1 before
% calling the mex (line "NEIGH = NEIGH-1"), converting to C/Python 0-based
% (-1 = absent).  We apply the same conversion here so the saved arrays are
% immediately usable in Python without further adjustment.
fprintf('[5/9] Computing neighbour graph ...\n');
do_real = 1;   % positive-frequency (real-signal) filterbank
[NEIGH_raw, posInfo_raw] = comp_filterbankneighbors(a_rat, M, double(N), do_real);
% NEIGH_raw  : 6 x Nsum  MATLAB 1-based, 0 = absent
% posInfo_raw: 2 x Nsum  col 0 = channel (0-based), col 1 = time in samples

% Convert to Python convention
NEIGH_c   = int32(NEIGH_raw - 1);   % 6 x Nsum, 0-based, -1 = absent
NEIGH_py  = NEIGH_c.';              % Nsum x 6 for Python row-of-neighbours layout
posInfo_py = posInfo_raw.';          % Nsum x 2 (already 0-based channel indices)

nb.NEIGH   = NEIGH_py;     % Nsum x 6 int32, 0-based indices, -1 = absent
nb.posInfo = posInfo_py;   % Nsum x 2 double
save(fullfile(outDir, 'neighbors.mat'), '-struct', 'nb', '-v7');

if maxStep <= 5
    fprintf('\nStopping after step 5 (maxStep=%d).\n', maxStep);
    fprintf('Files in %s:\n', outDir);
    d = dir(fullfile(outDir, '*.mat'));
    for k = 1 : numel(d)
        fprintf('  %-36s  %5.0f kB\n', d(k).name, d(k).bytes / 1024);
    end
    return;
end

% ── 8.  phasegrad_frommag.mat ────────────────────────────────────────────
% Phase gradient from magnitude alone (the phase-magnitude relation).
% This mirrors the non-uniform path in filterbankconstphase.m lines 265-271.
%
% Inputs to comp_filterbankphasegradfrommag (matching filterbankconstphase):
%   abss  : scaled magnitude  abs(s{m}) / sqrt(a_rat(m))  (natural scaling)
%   tfr   : sqrt of TFR vector  (filterbankconstphase line 251: tfr = sqrt(tfr))
%   NEIGH : 0-based, passed as the raw (6 x Nsum) matrix to match the mex
%           (the mex reads column-major; C code sees 6-element rows per coeff)
%   posInfo: raw 2 x Nsum (mex reads column-major)
%
fprintf('[6/9] Computing phase gradient from magnitude ...\n');
scal         = 1 ./ sqrt(a_rat);            % M x 1 natural scaling
s_cell       = cellfun(@abs, c, 'UniformOutput', false);  % magnitude cells
s_scaled     = cellfun(@(sEl, sc) sEl * sc, s_cell, num2cell(scal), ...
                        'UniformOutput', false);
abss         = vertcat(s_scaled{:});         % Nsum x 1  scaled magnitude
sqtfr        = sqrt(tfr_v);                  % M x 1  (sqrt, matching line 251)
gderivweight = 0.5;
do_gabor     = 1;   % gabor (default) vs wavelet phase-magnitude relation

% comp_filterbankphasegradfrommag expects NEIGH in the same format that
% filterbankconstphase.m passes after "NEIGH = NEIGH-1", i.e. the
% 0-based (6 x Nsum) matrix in MATLAB column-major layout.
% posInfo is passed as-is (2 x Nsum).
[tgm_flat, fgm_flat, logs_flat] = comp_filterbankphasegradfrommag( ...
    abss, double(N), a_rat, M, sqtfr, fc_n, ...
    NEIGH_c, posInfo_raw, gderivweight, do_gabor);

pgm.tgrad_mag_flat = tgm_flat;    % Nsum x 1 double
pgm.fgrad_mag_flat = fgm_flat;    % Nsum x 1 double
pgm.logs_flat      = logs_flat;   % Nsum x 1 double  (log of scaled magnitude)
pgm.abss_flat      = abss;        % Nsum x 1 double  (scaled magnitude input)
pgm.sqtfr          = sqtfr;       % M x 1 double
pgm.scal           = scal;        % M x 1 double
save(fullfile(outDir, 'phasegrad_frommag.mat'), '-struct', 'pgm', '-v7');

% ── 9.  heapint.mat ──────────────────────────────────────────────────────
% Heap integration: reconstructs phase from magnitude + gradients.
%
% comp_filterbankheapint signature (see mex source):
%   phase = comp_filterbankheapint(s, tgrad, fgrad, neigh, posInfo,
%                                  cfreq, a, M, N, chanStart, tol, phasetype)
%   s, tgrad, fgrad : flat (Nsum, 1)
%   neigh           : 6 x Nsum  0-based, -1 = absent  (same as NEIGH_c above)
%   posInfo         : 2 x Nsum
%   chanStart       : M x 1  0-based start index of each channel (not used
%                     internally by the current mex, included for API parity)
%   phasetype 1 = TIMEINV  (gradients adjusted for time-invariant phase)
%   phasetype 2 = RELGRAD  (gradients used as-is)
fprintf('[7/9] Running heap integration ...\n');
tol       = 1e-6;
s_flat    = vertcat(cs_c{:});      % Nsum x 1  spectrogram (|c|^2)
tg_flat   = vertcat(tgrad_c{:});   % Nsum x 1  normalised IF from §6
fg_flat   = vertcat(fgrad_c{:});   % Nsum x 1  group delay from §6
chanStart = double([0; cumsum(double(N(1:end-1)))]);  % M x 1  0-based

phase_timeinv = comp_filterbankheapint( ...
    s_flat, tg_flat, fg_flat, NEIGH_c, posInfo_raw, fc_n, ...
    a_rat, M, double(N), chanStart, tol, 1);

phase_relgrad = comp_filterbankheapint( ...
    s_flat, tg_flat, fg_flat, NEIGH_c, posInfo_raw, fc_n, ...
    a_rat, M, double(N), chanStart, tol, 2);

hi.s_flat             = s_flat;           % Nsum x 1  spectrogram input
hi.tgrad_flat         = tg_flat;          % Nsum x 1  gradient input
hi.fgrad_flat         = fg_flat;          % Nsum x 1  gradient input
hi.phase_timeinv_flat = phase_timeinv;    % Nsum x 1  PHASETYPE_TIMEINV
hi.phase_relgrad_flat = phase_relgrad;    % Nsum x 1  PHASETYPE_RELGRAD
hi.tol                = tol;
save(fullfile(outDir, 'heapint.mat'), '-struct', 'hi', '-v7');

% ── 10. constphase.mat ───────────────────────────────────────────────────
% Full filterbankconstphase pipeline (default settings: natural scaling,
% gabor mode, real filterbank, tol=[1e-1, 1e-10]).
fprintf('[8/9] Running filterbankconstphase ...\n');
s_cell_abs = cellfun(@abs, c, 'UniformOutput', false);
[c_cp, newphase_cp, usedmask_cp] = filterbankconstphase( ...
    s_cell_abs, a_mat, fc_n, tfr_v);

cp.c_cp_flat      = vertcat(c_cp{:});        % Nsum x 1 complex
cp.newphase_flat  = vertcat(newphase_cp{:});  % Nsum x 1 double (radians)
cp.usedmask_flat  = double(vertcat(usedmask_cp{:}));  % Nsum x 1  0/1
save(fullfile(outDir, 'constphase.mat'), '-struct', 'cp', '-v7');

% ── 11. reassign.mat ─────────────────────────────────────────────────────
% Spectral reassignment: maps each coefficient to its estimated true TF pos.
% Input cs_c is a cell of spectrograms (|c{m}|^2); output sr is same shape.
fprintf('[9/9] Running filterbankreassign ...\n');
[sr_cell, repos_cell] = filterbankreassign(cs_c, tgrad_c, fgrad_c, a_mat, fc_n);

% repos_cell has one entry per flat coefficient.  Each entry is a vector of
% source indices (1-based MATLAB) identifying which input bins contributed.
% Convert to 0-based for Python.
Nsum_int = int32(Nsum);
repos_lengths = zeros(Nsum, 1, 'int32');
repos_flat    = {};   % variable-length; save as a struct array below
for ii = 1 : Nsum
    el = repos_cell{ii};
    repos_lengths(ii) = int32(numel(el));
    repos_flat{ii} = int32(el - 1);   % 0-based indices
end

ra.sr_flat       = vertcat(sr_cell{:});   % Nsum x 1 double (reassigned spec)
ra.repos_lengths = repos_lengths;          % Nsum x 1 int32  #sources per bin
% repos_flat is ragged (variable-length per entry); save by flattening:
repos_concatenated = int32(vertcat(repos_flat{:}) - 0);   % already 0-based above
ra.repos_concat  = repos_concatenated;  % sum(repos_lengths) x 1 int32
% Python: use np.split(repos_concat, np.cumsum(repos_lengths)[:-1]) to recover
save(fullfile(outDir, 'reassign.mat'), '-struct', 'ra', '-v7');

% ── 12. unif_heapint.mat ─────────────────────────────────────────────────
% Uniform filterbank case (a = 1 for all channels → N_unif = L per channel).
% Uses the same ERB filters but with hop = 1, exercising ufilterbankheapint.
%
% comp_ufilterbankheapint signature (see mex source):
%   phase = comp_ufilterbankheapint(s,tgrad,fgrad,cfreq,a,do_real,tol,phasetype)
%   s, tgrad, fgrad : (N_unif, M)  matrices  (column per channel)
fprintf('[+1] Building uniform filterbank reference ...\n');
a_unif     = 1;
a_mat_unif = [ones(M, 1), ones(M, 1)];   % M x 2, all ones
[g_unif, asan_unif]     = filterbankwin(g, a_mat_unif, L, 'normal');
[gh_unif, gd_unif, g_unif] = comp_phasegradfilters(g_unif, asan_unif, L);

c_unif  = comp_filterbank(f_pad, g_unif,  asan_unif);
ch_unif = comp_filterbank(f_pad, gh_unif, asan_unif);
cd_unif = comp_filterbank(f_pad, gd_unif, asan_unif);
% Each c_unif{m} has length L (since a=1 → N_unif = L)

% Stack into (L x M) matrices for the uniform mex convention
c_u_mat  = cell2mat(c_unif.');    % L x M  complex  (each cell is L x 1)
ch_u_mat = cell2mat(ch_unif.');
cd_u_mat = cell2mat(cd_unif.');

[tg_u_cell, fg_u_cell, cs_u_cell] = comp_filterbankphasegrad( ...
    c_unif, ch_unif, cd_unif, L, eps);

s_u_mat  = cell2mat(cs_u_cell.');   % L x M  double (spectrogram)
tg_u_mat = cell2mat(tg_u_cell.');   % L x M  double
fg_u_mat = cell2mat(fg_u_cell.');   % L x M  double

% Heap integration on the uniform grid
phase_u_timeinv = comp_ufilterbankheapint( ...
    s_u_mat, tg_u_mat, fg_u_mat, fc_n, a_unif, do_real, tol, 1);
phase_u_relgrad = comp_ufilterbankheapint( ...
    s_u_mat, tg_u_mat, fg_u_mat, fc_n, a_unif, do_real, tol, 2);

% Save matrices in (L x M) layout; Python reads as (L, M) arrays.
% Flat versions use column-major order: (:) in MATLAB = Fortran order.
uh.N_unif             = int32(L);
uh.M                  = int32(M);
uh.a_unif             = int32(a_unif);
uh.fc_n               = fc_n;           % M x 1
uh.c_mat              = c_u_mat;        % L x M complex
uh.s_mat              = s_u_mat;        % L x M double (spectrogram)
uh.tgrad_mat          = tg_u_mat;       % L x M double
uh.fgrad_mat          = fg_u_mat;       % L x M double
uh.phase_timeinv_mat  = phase_u_timeinv;  % L x M double
uh.phase_relgrad_mat  = phase_u_relgrad;  % L x M double
save(fullfile(outDir, 'unif_heapint.mat'), '-struct', 'uh', '-v7');

% ── Done ──────────────────────────────────────────────────────────────────
fprintf('\nDone.  Files in %s:\n', outDir);
d = dir(fullfile(outDir, '*.mat'));
for k = 1 : numel(d)
    fprintf('  %-36s  %5.0f kB\n', d(k).name, d(k).bytes / 1024);
end
fprintf('\nLoad all files in Python:\n');
fprintf('    import scipy.io, numpy as np, pathlib\n');
fprintf('    ref = {p.stem: scipy.io.loadmat(p, squeeze_me=True)\n');
fprintf('           for p in pathlib.Path(''reference_data'').glob(''*.mat'')}\n');

end  % export_reference_data
