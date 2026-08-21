% RUN_LAYER0_TESTS  Run all Layer 0 (primitive ops) unit and property tests.
%
%   Runs both the unit and property sub-suites, prints a summary, and
%   optionally saves numerical reference values to reference/layer0_reference.mat.
%
%   Usage (from MATLAB):
%     run_layer0_tests            % run tests, skip reference save
%     run_layer0_tests('save')    % run tests and save reference
%
%   The reference .mat can be loaded in Python:
%     import scipy.io
%     ref = scipy.io.loadmat('reference/layer0_reference_current.mat', squeeze_me=True)

function results = run_layer0_tests(varargin)

    do_save = any(strcmpi(varargin, 'save'));

    thisDir  = fileparts(mfilename('fullpath'));
    sharedDir= fullfile(thisDir, '..', 'shared');
    fbRoot   = fullfile(thisDir, '..', '..', '..');
    addpath(fbRoot);
    addpath(sharedDir);

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('  LAYER 0 — Primitive Ops Tests\n');
    fprintf('%s\n\n', repmat('=', 1, 70));

    % ── Unit tests ────────────────────────────────────────────────────────────
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
    fprintf('  LAYER 0 TOTAL: %d/%d passed', n_pass, n_total);
    if n_fail > 0
        fprintf('  |  %d FAILED:', n_fail);
        failed = results([results.Failed]);
        for k = 1:numel(failed)
            fprintf('\n    * %s', failed(k).Name);
        end
    end
    fprintf('\n%s\n\n', repmat('-', 1, 70));

    % ── Reference values ──────────────────────────────────────────────────────
    if do_save
        ref = collect_layer0_reference(fbRoot);
        save_reference(ref, fullfile(thisDir, 'reference', 'layer0_reference'));
    end
end

% ── Collect numerical reference values for cross-validation with Python ───────
function ref = collect_layer0_reference(fbRoot)
    addpath(fbRoot);
    rng(42);
    fs = 8000;  Ls = 1024;

    % Primitive utilities
    N = 32;
    ref.fftindex_N32    = fftindex(N);
    ref.fftindex_N33    = fftindex(33);
    ref.floor23_1000    = floor23(1000);
    ref.floor23_8000    = floor23(8000);

    x = randn(Ls, 1) + 1i*randn(Ls, 1);
    ref.involute_real   = real(involute(x));
    ref.involute_imag   = imag(involute(x));
    ref.modcent_test    = modcent(linspace(-3, 3, 64)', 2);

    % Padding
    h32 = firwin('hann', 32);
    ref.fir2long_h32    = fir2long(h32, 128);
    ref.postpad_test    = postpad(h32, 64);
    ref.middlepad_test  = middlepad(h32, 64);

    % Signal extension (periodic, one example)
    xshort = randn(64, 1);
    ref.extBoundary_per = comp_extBoundary(xshort, 16, 'per');
    ref.extBoundary_zpd = comp_extBoundary(xshort, 16, 'zpd');

    % Subsampling
    xn = randn(Ls, 1);
    ref.downs_a2 = comp_downs(xn, 2);
    ref.ups_a2   = comp_ups(xn, 2);

    fprintf('  Layer 0 reference values collected.\n');
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
