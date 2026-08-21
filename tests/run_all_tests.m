% RUN_ALL_TESTS  Master runner: runs all layers and produces a unified report.
%
%   Usage (from the tests/ directory or any path):
%     run_all_tests              % run all layers
%     run_all_tests('save')      % run all layers and save all reference .mat files
%     run_all_tests('layer', 0)  % run only Layer 0
%     run_all_tests('layer', 3)  % run only Layer 3
%
%   Reference files are written to:
%     layer0_ops/reference/layer0_reference.mat        (golden)
%     layer0_ops/reference/layer0_reference_current.mat
%     ... (same pattern for layers 1, 2, 3)
%
%   Python cross-validation:
%     import scipy.io
%     ref = scipy.io.loadmat('layer2_filterbank/reference/layer2_reference_current.mat',
%                            squeeze_me=True)
%
%   Layer status:
%     Layer 0  (primitive ops)      -- full test suite
%     Layer 1  (filter objects)     -- full test suite
%     Layer 2  (filterbank API)     -- full test suite
%     Layer 3  (output repr)        -- full test suite
%     Layer 4  (auditory model)     -- placeholder (Python port target)
%     Layer 5  (ML interface)       -- placeholder (Python only)

function all_results = run_all_tests(varargin)

    % ---- Parse arguments ----------------------------------------------------
    do_save    = any(strcmpi(varargin, 'save'));
    layer_idx  = find(strcmpi(varargin, 'layer'));
    only_layer = [];
    if ~isempty(layer_idx) && numel(varargin) > layer_idx
        only_layer = varargin{layer_idx + 1};
    end

    % ---- Setup paths --------------------------------------------------------
    thisDir   = fileparts(mfilename('fullpath'));
    sharedDir = fullfile(thisDir, 'shared');
    addpath(sharedDir);
    setup_filterbank_paths();

    % ---- Ensure signal battery exists ---------------------------------------
    batteryFile = fullfile(sharedDir, 'signals', 'signal_battery.mat');
    if ~exist(batteryFile, 'file')
        fprintf('Generating signal battery...\n');
        make_signal_battery('save', true);
    end

    % ---- Print header -------------------------------------------------------
    fprintf('\n%s\n', repmat('#', 1, 70));
    fprintf('  LTFAT FILTERBANK -- Full Test Suite\n');
    fprintf('  %s\n', datestr(now));
    fprintf('%s\n\n', repmat('#', 1, 70));

    save_arg = {};
    if do_save, save_arg = {'save'}; end

    all_results  = [];
    layer_totals = struct();

    % ---- Layer 0 ------------------------------------------------------------
    if isempty(only_layer) || only_layer == 0
        res = run_layer_safe(@() run_layer0_tests(save_arg{:}), 'Layer 0', thisDir);
        all_results = [all_results, res];
        layer_totals.layer0 = summarise(res);
    end

    % ---- Layer 1 ------------------------------------------------------------
    if isempty(only_layer) || only_layer == 1
        res = run_layer_safe(@() run_layer1_tests(save_arg{:}), 'Layer 1', thisDir);
        all_results = [all_results, res];
        layer_totals.layer1 = summarise(res);
    end

    % ---- Layer 2 ------------------------------------------------------------
    if isempty(only_layer) || only_layer == 2
        res = run_layer_safe(@() run_layer2_tests(save_arg{:}), 'Layer 2', thisDir);
        all_results = [all_results, res];
        layer_totals.layer2 = summarise(res);
    end

    % ---- Layer 3 ------------------------------------------------------------
    if isempty(only_layer) || only_layer == 3
        res = run_layer_safe(@() run_layer3_tests(save_arg{:}), 'Layer 3', thisDir);
        all_results = [all_results, res];
        layer_totals.layer3 = summarise(res);
    end

    % ---- Layer 4 (placeholder) ----------------------------------------------
    if isempty(only_layer) || only_layer == 4
        fprintf('\n%s\n', repmat('=', 1, 70));
        fprintf('  LAYER 4 -- Auditory nonlinear stages (placeholder -- no tests yet)\n');
        fprintf('%s\n\n', repmat('=', 1, 70));
        layer_totals.layer4 = struct('pass', 0, 'fail', 0, 'total', 0);
    end

    % ---- Layer 5 (placeholder) ----------------------------------------------
    if isempty(only_layer) || only_layer == 5
        res = run_layer_safe(@() run_layer5_tests(save_arg{:}), 'Layer 5', thisDir);
        all_results = [all_results, res];
        layer_totals.layer5 = summarise(res);
    end

    % ---- Final summary ------------------------------------------------------
    fprintf('\n%s\n', repmat('#', 1, 70));
    fprintf('  FINAL SUMMARY\n');
    fprintf('%s\n', repmat('#', 1, 70));
    fprintf('  %-10s  %6s  %6s  %6s\n', 'Layer', 'Pass', 'Fail', 'Total');
    fprintf('  %s\n', repmat('-', 1, 40));

    layers = fieldnames(layer_totals);
    grand_pass = 0;  grand_fail = 0;  grand_total = 0;
    for k = 1 : numel(layers)
        s = layer_totals.(layers{k});
        fprintf('  %-10s  %6d  %6d  %6d\n', layers{k}, s.pass, s.fail, s.total);
        grand_pass  = grand_pass  + s.pass;
        grand_fail  = grand_fail  + s.fail;
        grand_total = grand_total + s.total;
    end
    fprintf('  %s\n', repmat('-', 1, 40));
    fprintf('  %-10s  %6d  %6d  %6d\n', 'TOTAL', grand_pass, grand_fail, grand_total);
    fprintf('%s\n\n', repmat('#', 1, 70));
end

% ---- Helpers ----------------------------------------------------------------

function results = run_layer_safe(fn, label, testsDir)
    label_lc = lower(strrep(label, ' ', ''));
    layer_dirs = dir(fullfile(testsDir, [label_lc '*']));
    for k = 1 : numel(layer_dirs)
        addpath(fullfile(testsDir, layer_dirs(k).name));
    end
    try
        results = fn();
        if isempty(results)
            results = matlab.unittest.TestResult.empty();
        end
    catch ME
        warning('run_all_tests:layerError', '%s failed to run: %s', label, ME.message);
        results = matlab.unittest.TestResult.empty();
    end
end

function s = summarise(results)
    if isempty(results)
        s.pass = 0; s.fail = 0; s.total = 0;
        return;
    end
    s.total = numel(results);
    s.fail  = sum([results.Failed]);
    s.pass  = s.total - s.fail - sum([results.Incomplete]);
end
