function compare_paths_ltfat()
%COMPARE_PATHS_LTFAT  Does LTFAT's own magnitude path agree with its own signal path?
%
%   compare_paths_ltfat()
%
%   WHY THIS EXISTS
%   ---------------
%   cool-frames has two ways to get the phase gradients that PGHI integrates:
%
%     signal path     filterbankphasegrad(f, g, a, L) -- derivative filters,
%                     exact, needs the complex coefficients and NO gamma.
%     magnitude path  comp_filterbankphasegradfrommag(...) -- estimated from
%                     |c| alone, which is the whole point of magnitude-only
%                     phase retrieval, and needs sqtfr.
%
%   On cool-frames' banks the two disagree on INTERIOR channels under every
%   gamma tried. That is what limits magnitude-only reconstruction there, and
%   it is reported as "structural, not a parameter choice".
%
%   What has NOT been established is whether the disagreement is inherited
%   from LTFAT or introduced by the port. The cross-language work settled that
%   the two implementations of the MAGNITUDE path are identical -- fed LTFAT's
%   own magnitudes, hops, centre frequencies and sqtfr, cool-frames reproduces
%   LTFAT's gradients at a median ratio of 1.00000000 on every channel, under
%   edge_mode='ltfat'. The signal path was never compared.
%
%   So the question reduces to: inside MATLAB, with no Python anywhere, do
%   LTFAT's two paths agree with each other? This script answers exactly that.
%
%     - If they disagree here too, the behaviour is LTFAT's and cool-frames
%       reproduces it faithfully. The limitation is in the method, and the
%       paper can say so.
%     - If they agree here, the port has a defect in the signal path, and the
%       magnitude-path parity result does not cover it.
%
%   HOW TO RUN
%   ----------
%       >> addpath(genpath('/path/to/ltfat')); ltfatstart;
%       >> cd /path/to/cool-frames/tests
%       >> compare_paths_ltfat
%
%   Writes reference_data/pathcompare_<designer>.mat so the Python side can
%   read the same numbers without re-deriving them.

thisDir = fileparts(mfilename('fullpath'));
outDir  = fullfile(thisDir, 'reference_data');
if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('LTFAT: %s\n\n', which('ltfatstart'));

fs = 8000;
Ls = 4096;
t  = (0 : Ls - 1)' / fs;
% Same probe as export_sqtfr_reference, so the two exports are comparable.
f = 0.5 * sin(2*pi*(100*t + (3000 - 100)/(2*(Ls/fs)) * t.^2)) ...
  + sin(2*pi*440*t) + 0.5*sin(2*pi*1000*t) + 0.3*sin(2*pi*2500*t);
f = f / max(abs(f));

designers = {
    'audfilters', @() audfilters(fs, Ls)
    'cqtfilters', @() cqtfilters(fs, 50, fs/2 - 100, 12, Ls)
};

for k = 1 : size(designers, 1)
    name = designers{k, 1};
    try
        compare_one(outDir, name, designers{k, 2}, fs, Ls, f);
    catch err
        fprintf('[FAIL] %-14s %s\n', name, err.message);
    end
end
end


% =========================================================================
function compare_one(outDir, name, builder, fs, Ls, f)

fprintf('=== %s ===\n', name);
[g, a_mat, ~, L, info] = builder();
M = numel(g);
if size(a_mat, 2) < 2
    a_mat = [a_mat(:,1), ones(size(a_mat,1), 1)];
end
a_rat = double(a_mat(:,1)) ./ double(a_mat(:,2));
N     = ceil(L ./ a_rat);

if ~isfield(info, 'tfr')
    fprintf('    no info.tfr; skipping\n');
    return
end
if isa(info.tfr, 'function_handle'), tfr_v = info.tfr(L); else, tfr_v = info.tfr; end
tfr_v = tfr_v(:);
if numel(tfr_v) == 1, tfr_v = repmat(tfr_v, M, 1); end
fc_n = info.fc(:);

% --- signal path: derivative filters, no gamma ------------------------
[tgrad_sig, fgrad_sig, ~] = filterbankphasegrad(f, g, a_mat, L);
tg_sig = vertcat(tgrad_sig{:});
fg_sig = vertcat(fgrad_sig{:});

% --- magnitude path: LTFAT's own estimator at LTFAT's own sqtfr -------
c        = filterbank(f, g, a_mat);
s_cell   = cellfun(@abs, c, 'UniformOutput', false);
scal     = 1 ./ sqrt(a_rat);
s_scaled = cellfun(@(sEl, sc) sEl * sc, s_cell, num2cell(scal), ...
                   'UniformOutput', false);
abss     = vertcat(s_scaled{:});

[NEIGH_raw, posInfo] = comp_filterbankneighbors(a_rat, M, double(N), 1);
[tg_mag, fg_mag, ~]  = comp_filterbankphasegradfrommag( ...
    abss, double(N), a_rat, M, sqrt(tfr_v), fc_n, ...
    int32(NEIGH_raw - 1), posInfo, 0.5, 1);

% --- compare, weighted by magnitude ------------------------------------
% An unweighted comparison is dominated by cells where |c| is at the noise
% floor and the phase gradient is meaningless in both paths.  Restrict to
% cells carrying real energy -- that is where a disagreement would actually
% cost reconstruction quality.
w    = vertcat(s_cell{:});
live = w > 0.01 * max(w);
chan = repelem((1:M).', double(N));

fprintf('    M = %d, L = %d, %d of %d cells above -40 dB\n', ...
        M, L, sum(live), numel(w));

report('tgrad', tg_sig, tg_mag, live, chan, M);
report('fgrad', fg_sig, fg_mag, live, chan, M);

r = struct('designer', name, 'fs', fs, 'Ls', Ls, 'L', L, 'M', M, ...
           'N', int32(N), 'a_rat', a_rat, 'fc_n', fc_n, 'tfr', tfr_v, ...
           'f', f, 'tgrad_signal', tg_sig, 'fgrad_signal', fg_sig, ...
           'tgrad_mag', tg_mag, 'fgrad_mag', fg_mag, 'abss_scaled', abss, ...
           'mag', w);
save(fullfile(outDir, sprintf('pathcompare_%s.mat', name)), '-struct', 'r', '-v7');
fprintf('    saved pathcompare_%s.mat\n\n', name);
end


% =========================================================================
function report(label, xs, xm, live, chan, M)
% Relative disagreement, on the live cells, overall and split
% interior/edge -- the split matters because the two implementations are
% known to differ by 2x on the one-sided edge channels by convention, and
% that is NOT the disagreement being investigated here.
xsl   = xs(live);
xml   = xm(live);
rel   = abs(xsl - xml) ./ max(abs(xsl), eps);
cl    = chan(live);
inner = (cl > 1) & (cl < M);

fprintf('    %-6s  median rel |signal - magnitude| : all %8.3f | inner %8.3f | edge %8.3f\n', ...
        label, median(rel), median(rel(inner)), median(rel(~inner)));

% Correlation, computed without the Statistics Toolbox so this runs anywhere.
fprintf('    %-6s  correlation(signal, magnitude)  : all %8.4f | inner %8.4f\n', ...
        label, pearson(xsl, xml), pearson(xsl(inner), xml(inner)));
end


% =========================================================================
function r = pearson(x, y)
x = x(:) - mean(x(:));
y = y(:) - mean(y(:));
d = norm(x) * norm(y);
if d == 0
    r = NaN;
else
    r = (x' * y) / d;
end
end
