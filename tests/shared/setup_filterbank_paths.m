function setup_filterbank_paths()
%SETUP_LAYER2_PATHS  Add all directories needed to run the filterbank tests.
%
%   setup_filterbank_paths() adds every layer subdirectory, utility folder, and
%   legacy support directory to the MATLAB path.  It is idempotent – safe to
%   call repeatedly.
%
%   Directory layout (all MATLAB code now lives under filterbank/):
%
%     ltfat_filterbank/
%       filterbank/              ← project root  (fbProject)
%         layer0/                ← primitive ops
%           analysis/  synthesis/  frame_math/  fir_conversion/
%           signal_extension/  subsampling/  math_utils/
%         layer1/                ← filter objects
%           filter_constructors/  filter_design/  filter_prep/
%           window_functions/  auditory_scales/
%         layer2/                ← filterbank module
%           analysis_synthesis/  dispatch/  format/  frame/  visualization/
%         layer3/                ← output representation / phase processing
%           phase_processing/  reassignment/
%         layer4/                ← auditory nonlinear stages (placeholder)
%         layer5/                ← ML interface (placeholder)
%         utils/
%           arg_parsers/  assertions/  helpers/
%           legacy/
%             gabor/             ← LTFAT Gabor/DGT (required by some filterbank internals)
%             frames/            ← LTFAT general frame machinery (required by ifilterbankiter)
%         signals/               ← test signals
%         tests/
%           shared/              ← this file lives here

    thisFile  = fileparts(mfilename('fullpath'));   % tests/shared/
    fbProject = fullfile(thisFile, '..', '..');     % filterbank/

    % ── Layers 0–3 (recursively adds every sub-folder) ───────────────────
    for layer = {'layer0', 'layer1', 'layer2', 'layer3'}
        p = fullfile(fbProject, layer{1});
        if exist(p, 'dir')
            addpath(genpath(p));
        end
    end

    % ── Utilities (arg_parsers, assertions, helpers) ──────────────────────
    utilsDir = fullfile(fbProject, 'utils');
    if exist(utilsDir, 'dir')
        addpath(genpath(utilsDir));
    end

    % ── Signals ───────────────────────────────────────────────────────────
    sigDir = fullfile(fbProject, 'signals');
    if exist(sigDir, 'dir')
        addpath(sigDir);
    end

    % ── Shared test utilities ─────────────────────────────────────────────
    addpath(thisFile);

end
