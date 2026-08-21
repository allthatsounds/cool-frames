%RUN_LTFAT_TESTS  Run the full LTFAT filterbank unit-test suite and
%                 serialise results for later reference from Python.
%
%   Usage (from the filterbank/ directory or with filterbank/ on path):
%       run_ltfat_tests
%
%   Output files (written to the same directory as this script):
%       test_results_current.mat   -- overwritten on every run
%       test_results_reference.mat -- written only if it does not yet exist
%                                     (golden reference; never auto-overwritten)
%
%   Python loading example:
%       import scipy.io, pprint
%       d = scipy.io.loadmat('test_results_current.mat', squeeze_me=True)
%       pprint.pprint(d['results'].item())
%
%   The .mat file uses format v7 (default), which is fully compatible with
%   scipy.io.loadmat.  Every field is either a scalar double, a numeric
%   array, or a char array (MATLAB strings → bytes in Python).
%
%   -------------------------------------------------------------------------
%   File structure
%   -------------------------------------------------------------------------
%   results.meta
%       .timestamp          char  'dd-mmm-yyyy HH:MM:SS'
%       .matlab_version     char  version string
%       .platform           char  computer() string
%   results.tests           1×N struct array
%       .name               char  test class + method
%       .passed             double  1 = pass, 0 = fail
%       .duration_s         double  wall-clock seconds
%       .message            char  failure detail (empty when passed)
%   results.numerical
%       .reconstruction_err_dual    scalar  relative error, dual recon
%       .reconstruction_err_iter    scalar  relative residual, iterative recon
%       .frame_bounds_erb           1×2     [A, B] for ERB bank
%       .audfilter_fc               M×1     center freqs (Hz) from audfilters
%       .freqtoaud_erb              1×4     freqtoaud([100,500,1000,4000],'erb')
%       .audtofreq_erb              1×4     audtofreq([10,20,30,40],'erb')
%       .audfiltbw_erb              1×4     audfiltbw([100,500,1000,4000],'erb')
%       .filterbankresponse_first10 1×10    first 10 values of filterbankresponse
%       .firwin_hann32              32×1    firwin('hann', 32)
%       .cqt_fc_12bins              M×1     center freqs from cqtfilters(8000,100,4000,12,1024)

% ── 0. Setup ──────────────────────────────────────────────────────────────
this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);
addpath(parent_dir);
addpath(this_dir);

fprintf('\n=== LTFAT Filterbank Test Suite ===\n');
fprintf('MATLAB %s  |  %s\n\n', version, computer());

% ── 1. Run all test classes ───────────────────────────────────────────────
test_classes = { ...
    'TestAnalysisSynthesis', ...
    'TestFrameMath',         ...
    'TestAdvancedFilters',   ...
    'TestPostProcessing',    ...
    'TestUtilities'          ...
};

all_results = matlab.unittest.TestResult.empty();

for k = 1 : numel(test_classes)
    class_name = test_classes{k};
    fprintf('Running %s ...', class_name);
    try
        tr = runtests(fullfile(this_dir, [class_name '.m']));
        fprintf('  %d/%d passed\n', sum([tr.Passed]), numel(tr));
        all_results = [all_results, tr]; %#ok<AGROW>
    catch ME
        fprintf('  ERROR loading class: %s\n', ME.message);
    end
end

% ── 2. Print summary table ────────────────────────────────────────────────
n_total  = numel(all_results);
n_passed = sum([all_results.Passed]);
n_failed = n_total - n_passed;

fprintf('\n--- Summary ---\n');
fprintf('Total:  %d\n', n_total);
fprintf('Passed: %d\n', n_passed);
fprintf('Failed: %d\n', n_failed);

if n_failed > 0
    fprintf('\nFailed tests:\n');
    for k = 1 : numel(all_results)
        if ~all_results(k).Passed
            fprintf('  FAIL  %s\n', all_results(k).Name);
            if ~isempty(all_results(k).Details)
                disp(all_results(k).Details);
            end
        end
    end
end

% ── 3. Collect per-test data for serialisation ────────────────────────────
n = numel(all_results);
tests_struct(n) = struct('name','','passed',0,'duration_s',0,'message','');
for k = 1 : n
    tests_struct(k).name       = all_results(k).Name;
    tests_struct(k).passed     = double(all_results(k).Passed);
    tests_struct(k).duration_s = all_results(k).Duration;
    if ~all_results(k).Passed && ~isempty(all_results(k).Details)
        tests_struct(k).message = evalc('disp(all_results(k).Details)');
    else
        tests_struct(k).message = '';
    end
end

% ── 4. Collect numerical reference values ────────────────────────────────
fprintf('\nCollecting numerical reference values ...\n');

rng(42);   % same seed as make_test_params
fs = 8000;
Ls = 1024;
t  = (0:Ls-1)' / fs;
f_ref = randn(Ls, 1);   % reference signal (same as noise_mono)

num = struct();

try
    [g_erb, a_erb, fc_erb, L_erb] = audfilters(fs, Ls);
    M_erb = numel(g_erb);

    % --- Reconstruction error (dual) ---
    gd_erb   = filterbankdual(g_erb, a_erb, L_erb);
    c_erb    = filterbank(f_ref, g_erb, a_erb);
    f_rec    = ifilterbank(c_erb, gd_erb, a_erb);
    num.reconstruction_err_dual = norm(f_rec(1:Ls) - f_ref) / norm(f_ref);

    % --- Reconstruction residual (iterative) ---
    [~, relres, ~]          = ifilterbankiter(c_erb, g_erb, a_erb);
    num.reconstruction_err_iter = relres;

    % --- Frame bounds ---
    [AF, BF]              = filterbankbounds(g_erb, a_erb, L_erb);
    num.frame_bounds_erb  = [AF, BF];

    % --- Center frequencies ---
    num.audfilter_fc      = fc_erb;

    % --- Auditory scale conversions ---
    test_freqs            = [100, 500, 1000, 4000];
    num.freqtoaud_erb     = freqtoaud(test_freqs, 'erb');
    num.audtofreq_erb     = audtofreq([10, 20, 30, 40], 'erb');
    num.audfiltbw_erb     = audfiltbw(test_freqs, 'erb');

    % --- Filterbank response (first 10 bins) ---
    R = filterbankresponse(g_erb, a_erb, L_erb);
    num.filterbankresponse_first10 = R(1:10)';

    % --- FIR window ---
    num.firwin_hann32     = firwin('hann', 32);

    % --- CQT center frequencies ---
    [~, ~, fc_cqt, ~]    = cqtfilters(fs, 100, 4000, 12, Ls);
    num.cqt_fc_12bins     = fc_cqt;

    fprintf('Numerical collection: OK\n');

catch ME
    warning('run_ltfat_tests:numericalCollection', ...
        'Numerical collection failed: %s', ME.message);
    % Fill in NaN sentinels so the struct is still saveable.
    default_fields = { ...
        'reconstruction_err_dual', 'reconstruction_err_iter', ...
        'frame_bounds_erb', 'audfilter_fc', 'freqtoaud_erb', ...
        'audtofreq_erb', 'audfiltbw_erb', 'filterbankresponse_first10', ...
        'firwin_hann32', 'cqt_fc_12bins' };
    for fk = default_fields
        if ~isfield(num, fk{1})
            num.(fk{1}) = NaN;
        end
    end
end

% ── 5. Assemble results struct ────────────────────────────────────────────
results.meta.timestamp       = datestr(now, 'dd-mmm-yyyy HH:MM:SS');
results.meta.matlab_version  = version;
results.meta.platform        = computer();
results.tests                = tests_struct;
results.numerical            = num;

% ── 6. Save current run ───────────────────────────────────────────────────
current_path   = fullfile(this_dir, 'test_results_current.mat');
reference_path = fullfile(this_dir, 'test_results_reference.mat');

save(current_path, 'results', '-v7');
fprintf('\nSaved current results  → %s\n', current_path);

% ── 7. Save reference (only if it does not exist yet) ─────────────────────
if ~exist(reference_path, 'file')
    save(reference_path, 'results', '-v7');
    fprintf('Saved reference (NEW) → %s\n', reference_path);
else
    fprintf('Reference already exists; not overwritten:\n  %s\n', reference_path);
    fprintf('  Delete it manually to reset the golden reference.\n');
end

fprintf('\nDone.\n');
