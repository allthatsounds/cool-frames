% RUN_LAYER2_TESTS  Run all Layer 2 (filterbank API) unit and property tests.
%
%   Includes the existing test suite (TestAnalysisSynthesis, TestFrameMath,
%   TestAdvancedFilters, TestUtilities) plus the new property-based tests.
%
%   Usage:
%     run_layer2_tests
%     run_layer2_tests('save')   % also save reference .mat

function results = run_layer2_tests(varargin)

    do_save = any(strcmpi(varargin, 'save'));

    thisDir   = fileparts(mfilename('fullpath'));
    sharedDir = fullfile(thisDir, '..', 'shared');
    addpath(sharedDir);
    setup_filterbank_paths();

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('  LAYER 2 — Filterbank API Tests\n');
    fprintf('%s\n\n', repmat('=', 1, 70));

    % ── Unit tests (existing suite) ───────────────────────────────────────────
    fprintf('--- Unit tests ---\n');
    unit_results = runtests(fullfile(thisDir, 'unit'), 'Recursively', false);
    print_summary(unit_results, 'Unit');

    % ── Property tests ────────────────────────────────────────────────────────
    fprintf('\n--- Property tests ---\n');
    prop_results = runtests(fullfile(thisDir, 'property'), 'Recursively', false);
    print_summary(prop_results, 'Property');

    % ── Combined ──────────────────────────────────────────────────────────────
    results = [unit_results, prop_results];
    n_total = numel(results);
    n_pass  = sum(~[results.Failed] & ~[results.Incomplete]);
    n_fail  = sum([results.Failed]);

    fprintf('\n%s\n', repmat('-', 1, 70));
    fprintf('  LAYER 2 TOTAL: %d/%d passed', n_pass, n_total);
    if n_fail > 0
        failed = results([results.Failed]);
        for k = 1:numel(failed)
            fprintf('\n    * %s', failed(k).Name);
        end
    end
    fprintf('\n%s\n\n', repmat('-', 1, 70));

    % ── Reference values ──────────────────────────────────────────────────────
    if do_save
        ref = collect_layer2_reference();
        save_reference(ref, fullfile(thisDir, 'reference', 'layer2_reference'));
    end
end

function ref = collect_layer2_reference()
    setup_filterbank_paths();
    rng(42);
    fs = 8000;  Ls = 1024;

    x  = randn(Ls, 1);
    xc = randn(Ls, 1) + 1i*randn(Ls, 1);

    % ERB filterbank
    [g, a, fc, L] = audfilters(fs, Ls);
    gd = filterbankdual(g, a, L);
    [A, B] = filterbankbounds(g, a, L);

    ref.audfilters_fc          = fc(:);
    ref.frame_bounds_erb       = [A, B];
    ref.n_filters_erb          = numel(g);

    % Perfect reconstruction error (real noise)
    c  = filterbank(x, g, a);
    xr = ifilterbank(c, gd, a, L);
    ref.reconstruction_err_dual = norm(x - xr) / norm(x);

    % Energy ratio (with 1/a_m weighting)
    a_vals = a(:, 1);
    energy_Tx = sum(cellfun(@(cm, am) (1/am)*norm(cm)^2, c, num2cell(a_vals)));
    ref.energy_ratio_erb = energy_Tx / norm(x)^2;

    % CQT filterbank
    [gcq, acq, fccq] = cqtfilters(fs, 55, 3500, 12, Ls);
    [Acq, Bcq] = filterbankbounds(gcq, acq, Ls);
    ref.cqt_fc_12bins    = fccq(:);
    ref.frame_bounds_cqt = [Acq, Bcq];

    % Auditory scale
    freqs = linspace(100, 3500, 32)';
    ref.freqtoaud_erb  = freqtoaud(freqs, 'erb');
    ref.audtofreq_erb  = audtofreq(freqtoaud(freqs,'erb'), 'erb');

    fprintf('  Layer 2 reference values collected.\n');
end

function print_summary(res, label)
    n     = numel(res);
    n_ok  = sum(~[res.Failed] & ~[res.Incomplete]);
    n_fail= sum([res.Failed]);
    fprintf('  %s: %d/%d passed', label, n_ok, n);
    if n_fail > 0
        failed = res([res.Failed]);
        for k = 1:numel(failed)
            fprintf('\n    FAIL: %s', failed(k).Name);
        end
    end
    fprintf('\n');
end
