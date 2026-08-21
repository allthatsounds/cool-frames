% RUN_LAYER1_TESTS  Run all Layer 1 (filter objects) unit and property tests.
%
%   Layer 1 tests cover filter constructors and frequency response computation:
%   blfilter, freqfilter, freqwavelet, firwin, freqwin, firfilter,
%   freqtoaud, audtofreq, audfiltbw, audspace.
%
%   STATUS: Unit test stubs present. Full test suite to be added when
%           Layer 1 Python classes are ported.
%
%   Usage:
%     run_layer1_tests
%     run_layer1_tests('save')   % also save reference .mat

function results = run_layer1_tests(varargin)

    do_save = any(strcmpi(varargin, 'save'));

    thisDir  = fileparts(mfilename('fullpath'));
    sharedDir= fullfile(thisDir, '..', 'shared');
    addpath(sharedDir);
    setup_filterbank_paths();

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('  LAYER 1 — Filter Objects Tests\n');
    fprintf('%s\n\n', repmat('=', 1, 70));

    unit_files = dir(fullfile(thisDir, 'unit', 'Test*.m'));
    prop_files = dir(fullfile(thisDir, 'property', 'Prop*.m'));

    if isempty(unit_files) && isempty(prop_files)
        fprintf('  (No test files yet — placeholder for Python port phase)\n\n');
        results = [];
        return;
    end

    results = [];
    if ~isempty(unit_files)
        fprintf('--- Unit tests ---\n');
        unit_results = runtests(fullfile(thisDir, 'unit'), 'Recursively', false);
        print_summary(unit_results, 'Unit');
        results = [results, unit_results];
    end
    if ~isempty(prop_files)
        fprintf('\n--- Property tests ---\n');
        prop_results = runtests(fullfile(thisDir, 'property'), 'Recursively', false);
        print_summary(prop_results, 'Property');
        results = [results, prop_results];
    end

    if do_save && ~isempty(results)
        ref = collect_layer1_reference([]);
        save_reference(ref, fullfile(thisDir, 'reference', 'layer1_reference'));
    end
end

function ref = collect_layer1_reference(~)
    % paths already set by setup_filterbank_paths() in the runner
    fs = 8000;  Ls = 1024;

    % Filter frequency responses at key points
    [g, a, fc] = audfilters(fs, Ls);
    ref.audfilters_fc         = fc(:);
    ref.audfilters_n_filters  = numel(g);

    % Auditory scale functions
    freqs = linspace(100, 3500, 64)';
    ref.freqtoaud_erb  = freqtoaud(freqs, 'erb');
    ref.audtofreq_erb  = audtofreq(freqtoaud(freqs, 'erb'), 'erb');
    ref.audfiltbw_erb  = audfiltbw(freqs, 'erb');

    % Window
    ref.firwin_hann32  = firwin('hann', 32);

    fprintf('  Layer 1 reference values collected.\n');
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
