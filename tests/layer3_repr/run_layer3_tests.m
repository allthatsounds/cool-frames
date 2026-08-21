% RUN_LAYER3_TESTS  Run all Layer 3 (output representation / phase processing) tests.
%
%   Covers: filterbankphasegrad, filterbanksynchrosqueeze, filterbankconstphase,
%           filterbankreassign, filterbankresponse, filterbankfreqz, plotfilterbank,
%           rtpghifbwl (Real-Time Phase Gradient Heap Integration for filter banks).
%
%   Usage:
%     run_layer3_tests
%     run_layer3_tests('save')   % also save reference .mat

function results = run_layer3_tests(varargin)

    do_save = any(strcmpi(varargin, 'save'));

    thisDir   = fileparts(mfilename('fullpath'));
    sharedDir = fullfile(thisDir, '..', 'shared');
    addpath(sharedDir);
    setup_filterbank_paths();

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('  LAYER 3 — Output Representation / Phase Processing Tests\n');
    fprintf('%s\n\n', repmat('=', 1, 70));

    % ── Unit tests ────────────────────────────────────────────────────────────
    fprintf('--- Unit tests ---\n');
    unit_results = runtests(fullfile(thisDir, 'unit'), 'Recursively', false);
    print_summary(unit_results, 'Unit');

    % ── Property tests ────────────────────────────────────────────────────────
    fprintf('--- Property tests ---\n');
    prop_results = runtests(fullfile(thisDir, 'property'), 'Recursively', false);
    print_summary(prop_results, 'Property');

    results = [unit_results, prop_results];

    n_total = numel(results);
    n_pass  = sum(~[results.Failed] & ~[results.Incomplete]);
    n_fail  = sum([results.Failed]);

    fprintf('\n%s\n', repmat('-', 1, 70));
    fprintf('  LAYER 3 TOTAL: %d/%d passed', n_pass, n_total);
    if n_fail > 0
        failed = results([results.Failed]);
        for k = 1 : numel(failed)
            fprintf('\n    * %s', failed(k).Name);
        end
    end
    fprintf('\n%s\n\n', repmat('-', 1, 70));

    if do_save
        ref = collect_layer3_reference();
        save_reference(ref, fullfile(thisDir, 'reference', 'layer3_reference'));
    end
end

% ── Reference collection ─────────────────────────────────────────────────────

function ref = collect_layer3_reference()
    rng(42);
    fs = 8000;  Ls = 1024;
    x  = randn(Ls, 1);

    [g, a, ~, L, info] = audfilters(fs, Ls);
    c = filterbank(x, g, a);

    % Phase gradient
    [tgrad, fgrad, s, c2] = filterbankphasegrad(x, g, a, L);
    ref.phasegrad_tgrad1  = tgrad{1}(1:10);
    ref.phasegrad_fgrad1  = fgrad{1}(1:10);
    ref.phasegrad_s1      = s{1}(1:10);

    % Constphase
    fc_n = info.fc;
    tfr  = info.tfr(L);
    cr   = filterbankconstphase(c, a, fc_n, tfr);
    ref.constphase_abs1   = abs(cr{1}(1:10));

    % RTPGHIFB (wavelet filterbank)
    scales_ref = 2.^(linspace(5, -2, 32));
    [g_wl, a_wl, ~, ~, info_wl] = waveletfilters(Ls, scales_ref, 'repeat', 'uniform');
    corig_wl   = ufilterbank(x, g_wl, a_wl);
    s_wl       = abs(corig_wl.');
    [c_rtpghi, ~, tg_rtpghi, fg_rtpghi] = ...
        rtpghifbwl(s_wl, a_wl(1), info_wl.fc, info_wl.tfr);
    ref.rtpghifb_magabs   = norm(abs(c_rtpghi(:)) - s_wl(:));   % must be ~0
    ref.rtpghifb_tgrad1   = tg_rtpghi(1:5, 1);
    ref.rtpghifb_fgrad1   = fg_rtpghi(1:5, 1);

    % Filterbank response
    gf = filterbankresponse(g, a, L);
    ref.filterbankresponse_first32 = real(gf(1:32));

    fprintf('  Layer 3 reference values collected.\n');
end

% ── Helper ────────────────────────────────────────────────────────────────────

function print_summary(res, label)
    n      = numel(res);
    n_ok   = sum(~[res.Failed] & ~[res.Incomplete]);
    n_fail = sum([res.Failed]);
    fprintf('  %s: %d/%d passed', label, n_ok, n);
    if n_fail > 0
        failed = res([res.Failed]);
        for k = 1 : numel(failed)
            fprintf('\n    FAIL: %s', failed(k).Name);
        end
    end
    fprintf('\n');
end
