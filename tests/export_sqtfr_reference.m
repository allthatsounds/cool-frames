function export_sqtfr_reference(outDir)
%EXPORT_SQTFR_REFERENCE  Export LTFAT's per-designer TFR convention.
%
%   export_sqtfr_reference()
%   export_sqtfr_reference(outDir)
%
%   Writes  <outDir>/sqtfr_<designer>.mat  (default outDir:
%   tests/reference_data/) for every designer LTFAT provides, containing
%   everything the Python side needs to settle one question:
%
%       what does `sqtfr` have to be, per designer, for cool-frames'
%       magnitude-path PGHI to agree with LTFAT's?
%
%   WHY A SEPARATE SCRIPT FROM export_reference_data.m
%   --------------------------------------------------
%   export_reference_data.m already exports this, but only for `audfilters`,
%   as part of a ten-step pipeline that stops on the first error.  The
%   convention is designer-specific, so it has to be swept, and a designer
%   LTFAT does not ship must skip rather than abort the run.
%
%   WHAT IT SAVES, AND WHY EACH PIECE IS NEEDED
%   -------------------------------------------
%   H, foff        the realised filters.  The Python side checks the two
%                  languages built the *same bank* before comparing anything
%                  derived from it -- otherwise designer drift shows up as a
%                  convention error and you chase the wrong thing.
%   tfr            info.tfr(L) if the designer exposes it, else NaN.  This is
%                  LTFAT's answer to the question.
%   info_fields    the names of every field of info, so a designer that
%                  carries the ratio under another name is still discoverable.
%   tgrad_*, fgrad_*  the magnitude-path gradients at TWO settings:
%                  sqtfr = sqrt(tfr) (LTFAT's own) and sqtfr = ones.  The pair
%                  lets Python solve for the ratio in closed form, because the
%                  dependence is exactly 1/gamma in tgrad and gamma in fgrad.
%   abss, scal     the SCALED magnitudes actually fed to the mex.  LTFAT
%                  applies scal = 1./sqrt(a) first; cool-frames does not, and
%                  on a non-uniform bank that shifts every cross-channel
%                  log-magnitude difference.
%
%   HOW TO RUN
%   ----------
%       >> addpath(genpath('/path/to/ltfat')); ltfatstart;
%       >> cd /path/to/cool-frames/tests
%       >> export_sqtfr_reference
%
%   Nothing here needs mex beyond what filterbankconstphase already needs.
%   Every designer is wrapped in try/catch: a failure is reported and the
%   sweep continues.

if nargin < 1 || isempty(outDir)
    thisDir = fileparts(mfilename('fullpath'));
    outDir  = fullfile(thisDir, 'reference_data');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('LTFAT: %s\n', which('ltfatstart'));
try
    fprintf('version: %s\n', ltfathelp('version'));
catch
    fprintf('version: (ltfathelp unavailable)\n');
end
fprintf('Writing to: %s\n\n', outDir);

fs = 8000;
Ls = 4096;
t  = (0 : Ls - 1)' / fs;
% Deterministic and broadband, so every channel is excited: a sweep plus
% three partials.  No RNG, so MATLAB and Python see bit-identical input.
f = 0.5 * sin(2*pi*(100*t + (3000 - 100)/(2*(Ls/fs)) * t.^2)) ...
  + sin(2*pi*440*t) + 0.5*sin(2*pi*1000*t) + 0.3*sin(2*pi*2500*t);
f = f / max(abs(f));

% name, candidate call forms.  LTFAT's argument orders differ between
% designers and between releases, so each designer lists several and the
% first one that returns a usable 5-tuple wins.  Every failed attempt is
% printed with its error, so a designer that needs a form not listed here
% tells you exactly what to add rather than just disappearing.
designers = {
    'audfilters',     { @() audfilters(fs, Ls) }
    'cqtfilters',     { @() cqtfilters(fs, 50, fs/2 - 100, 12, Ls), ...
                        @() cqtfilters(fs, Ls, 'fmin', 50, 'fmax', fs/2 - 100, 'bins', 12), ...
                        @() cqtfilters(fs, 50, fs/2 - 100, 12, Ls, 'uniform') }
    % Geometric scales, 12 per octave.  The obvious linspace(10, 0.1, 50) is
    % what LTFAT's own help suggests and it exports fine, but the resulting
    % bank is NOT a frame (measured lower bound exactly 0, 395 uncovered
    % bins), so every reconstruction number from it is meaningless.  This set
    % gives A = 1.66, kappa = 2.28.
    'waveletfilters', { @() waveletfilters(Ls, 4 * 2.^(-(0:63)/12)), ...
                        @() waveletfilters(Ls, linspace(10, 0.1, 50)), ...
                        @() waveletfilters(fs, Ls) }
    'warpedfilters',  { @() warpedfilters(@log, @exp, fs, 50, fs/2 - 100, 12, Ls), ...
                        @() warpedfilters(@(x) log(x), @(x) exp(x), fs, 50, fs/2 - 100, 12, Ls), ...
                        @() warpedfilters('log', fs, 50, fs/2 - 100, 12, Ls) }
};

for k = 1 : size(designers, 1)
    name  = designers{k, 1};
    forms = designers{k, 2};

    if isempty(which(name))
        fprintf('[skip] %-16s not in this LTFAT\n', name);
        continue
    end

    done = false;
    for j = 1 : numel(forms)
        try
            export_one(outDir, name, forms{j}, fs, Ls, f);
            done = true;
            break
        catch err
            fprintf('[try%d] %-16s %s\n', j, name, err.message);
        end
    end
    if ~done
        fprintf('[FAIL] %-16s no candidate signature worked.\n', name);
        fprintf('       Run  help %s  and add the right call to the\n', name);
        fprintf('       designers table at the top of this file.\n');
    end
end

fprintf('\nDone. Files:\n');
d = dir(fullfile(outDir, 'sqtfr_*.mat'));
for k = 1 : numel(d)
    fprintf('  %-32s %6.0f kB\n', d(k).name, d(k).bytes / 1024);
end
end


% =========================================================================
function export_one(outDir, name, builder, fs, Ls, f)

fprintf('[%s] building ...\n', name);
% Output arity varies: not every LTFAT designer returns all five, and
% "Too many output arguments" is a different failure from a wrong signature.
info = struct();
L = [];
try
    [g, a_mat, ~, L, info] = builder();
catch
    try
        [g, a_mat, ~, L] = builder();
        fprintf('    (designer returns no info struct)\n');
    catch
        [g, a_mat] = builder();
        fprintf('    (designer returns only g and a)\n');
    end
end
if isempty(L)
    L = filterbanklength(Ls, a_mat);
end
M = numel(g);

% --- hops as rationals -------------------------------------------------
if size(a_mat, 2) < 2
    a_mat = [a_mat(:,1), ones(size(a_mat,1), 1)];
end
a_rat = double(a_mat(:,1)) ./ double(a_mat(:,2));
N     = ceil(L ./ a_rat);
Nsum  = sum(N);

% --- centre frequencies, normalised so that 1 == Nyquist ---------------
if isfield(info, 'fc')
    fc_n = info.fc(:);
else
    fc_n = nan(M, 1);
end

% --- LTFAT's own TFR ---------------------------------------------------
tfr_v = nan(M, 1);
tfr_source = 'absent';
if isfield(info, 'tfr')
    if isa(info.tfr, 'function_handle')
        tfr_v = info.tfr(L);
        tfr_source = 'info.tfr(L)';
    else
        tfr_v = info.tfr(:);
        tfr_source = 'info.tfr';
    end
end
tfr_v = tfr_v(:);
if numel(tfr_v) == 1, tfr_v = repmat(tfr_v, M, 1); end
fprintf('    M=%d  L=%d  tfr from %s\n', M, L, tfr_source);

% --- the realised filters, so Python can verify it is the same bank ----
gf = filterbankfreqz(g, a_mat, L);      % L x M  (full-length responses)

% --- analysis ----------------------------------------------------------
c = filterbank(f, g, a_mat);
s_cell = cellfun(@abs, c, 'UniformOutput', false);

scal     = 1 ./ sqrt(a_rat);
s_scaled = cellfun(@(sEl, sc) sEl * sc, s_cell, num2cell(scal), ...
                   'UniformOutput', false);
abss     = vertcat(s_scaled{:});
abss_raw = vertcat(s_cell{:});

% --- neighbour graph ---------------------------------------------------
do_real = 1;
[NEIGH_raw, posInfo_raw] = comp_filterbankneighbors(a_rat, M, double(N), do_real);
NEIGH_c = int32(NEIGH_raw - 1);      % 0-based, -1 = absent

gderivweight = 0.5;
do_gabor     = 1;

% --- gradients at LTFAT's tfr, and at tfr = 1 --------------------------
% The pair pins gamma: tgrad depends on 1/gamma and fgrad on gamma, so
% Python can solve for the ratio without re-deriving anything.
have_tfr = all(isfinite(tfr_v)) && all(tfr_v > 0);
if have_tfr
    [tg_tfr, fg_tfr, logs_tfr] = comp_filterbankphasegradfrommag( ...
        abss, double(N), a_rat, M, sqrt(tfr_v), fc_n, ...
        NEIGH_c, posInfo_raw, gderivweight, do_gabor);
else
    tg_tfr = nan(Nsum,1); fg_tfr = nan(Nsum,1); logs_tfr = nan(Nsum,1);
end

[tg_one, fg_one, logs_one] = comp_filterbankphasegradfrommag( ...
    abss, double(N), a_rat, M, ones(M,1), fc_n, ...
    NEIGH_c, posInfo_raw, gderivweight, do_gabor);

% same, but on UNSCALED magnitudes -- isolates the 1/sqrt(a) convention
[tg_one_raw, fg_one_raw, ~] = comp_filterbankphasegradfrommag( ...
    abss_raw, double(N), a_rat, M, ones(M,1), fc_n, ...
    NEIGH_c, posInfo_raw, gderivweight, do_gabor);

% --- full pipeline, if the designer gave us a tfr ----------------------
newphase_cp = nan(Nsum,1); usedmask_cp = nan(Nsum,1); c_cp_flat = nan(Nsum,1);
if have_tfr
    try
        [c_cp, newphase, usedmask] = filterbankconstphase(s_cell, a_mat, fc_n, tfr_v);
        c_cp_flat   = vertcat(c_cp{:});
        newphase_cp = vertcat(newphase{:});
        usedmask_cp = double(vertcat(usedmask{:}));
    catch err
        fprintf('    filterbankconstphase failed: %s\n', err.message);
    end
end

% --- save --------------------------------------------------------------
r = struct();
r.designer     = name;
r.fs           = fs;
r.Ls           = Ls;
r.L            = L;
r.M            = M;
r.N            = int32(N);
r.a_num        = int32(a_mat(:,1));
r.a_den        = int32(a_mat(:,2));
r.a_rat        = a_rat;
r.fc_n         = fc_n;
r.tfr          = tfr_v;
r.tfr_source   = tfr_source;
r.info_fields  = fieldnames(info);
r.f            = f;
r.gf           = gf;
r.scal         = scal;
r.abss_scaled  = abss;
r.abss_raw     = abss_raw;
r.NEIGH        = NEIGH_c.';
r.posInfo      = posInfo_raw.';
r.tgrad_tfr    = tg_tfr;
r.fgrad_tfr    = fg_tfr;
r.logs_tfr     = logs_tfr;
r.tgrad_one    = tg_one;
r.fgrad_one    = fg_one;
r.logs_one     = logs_one;
r.tgrad_one_raw = tg_one_raw;
r.fgrad_one_raw = fg_one_raw;
r.constphase_c        = c_cp_flat;
r.constphase_newphase = newphase_cp;
r.constphase_usedmask = usedmask_cp;
r.gderivweight = gderivweight;
r.do_gabor     = do_gabor;

save(fullfile(outDir, sprintf('sqtfr_%s.mat', name)), '-struct', 'r', '-v7');
fprintf('    saved sqtfr_%s.mat\n', name);
end
